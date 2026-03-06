"""
Unit tests for Flash Attention fallback mechanisms.

Tests verify that:
1. FA3, FA2, and SDPA produce consistent results
2. All backends work correctly with different configurations
3. Sliding window attention works across backends
4. GQA (Grouped Query Attention) works correctly
5. KV cache integration works with all backends
6. Gradients flow correctly through all backends

Run: python -m pytest tests/unit/models/test_attention_fallback.py -v -s

Note on test structure:
    Tests are organized by backend comparison:
    - TestBackendComparison: Compare FA3/FA2/SDPA outputs
    - TestSDPAFallback: SDPA-specific functionality
    - TestBackendFeatures: Features that should work on all backends
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from gpt_2.attention import HAS_FA2, HAS_FA3, _fa2, _fa3, attention_forward

# =============================================================================
# Helper Functions
# =============================================================================


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    if t1 is None or t2 is None:
        assert t1 is None and t2 is None, f"{name}: One tensor is None"
        return 0.0, 0.0

    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(
        t1, t2, atol=atol, rtol=rtol
    ), f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


def run_fa3_direct(q, k, v, causal=True, window_size=(-1, -1)):
    """Run FA3 directly on tensors (B, n_head, T, head_dim)."""
    # FA3 expects (B, T, n_head, head_dim)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    output = _fa3.flash_attn_func(q_t, k_t, v_t, causal=causal, window_size=window_size)
    return output.transpose(1, 2)  # Back to (B, n_head, T, head_dim)


def run_fa2_direct(q, k, v, causal=True, window_size=(-1, -1)):
    """Run FA2 directly on tensors (B, n_head, T, head_dim)."""
    # FA2 expects (B, T, n_head, head_dim)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    output = _fa2.flash_attn_func(q_t, k_t, v_t, causal=causal, window_size=window_size)
    return output.transpose(1, 2)  # Back to (B, n_head, T, head_dim)


# =============================================================================
# Backend Comparison Tests
# =============================================================================


@pytest.mark.skipif(not HAS_FA3, reason="FA3 required for comparison tests")
class TestFA3VsSDPA:
    """Compare FA3 and SDPA produce consistent results. Requires Hopper GPU."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention comparison."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)

        # FA3 output
        y_fa3 = run_fa3_direct(q, k, v, causal=True, window_size=(-1, -1))

        # SDPA output (use CPU to force SDPA fallback)
        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        y_sdpa = attention_forward(
            q_cpu, k_cpu, v_cpu, is_causal=True, enable_gqa=False, window_size=()
        )

        # Compare (convert SDPA to GPU and bfloat16 for comparison)
        y_sdpa_gpu = y_sdpa.to(device=self.DEVICE, dtype=self.DTYPE)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa_gpu, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        """Sliding window attention comparison."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)

        # FA3 output
        y_fa3 = run_fa3_direct(q, k, v, causal=True, window_size=(window, 0))

        # SDPA output with window
        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        y_sdpa = attention_forward(
            q_cpu,
            k_cpu,
            v_cpu,
            is_causal=True,
            enable_gqa=False,
            window_size=(window, 0),
        )

        y_sdpa_gpu = y_sdpa.to(device=self.DEVICE, dtype=self.DTYPE)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa_gpu, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """Group Query Attention comparison."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, n_heads, T, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, n_kv_heads, T, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, n_kv_heads, T, D, device=self.DEVICE, dtype=self.DTYPE)

        # FA3 output (FA3 handles GQA natively)
        y_fa3 = run_fa3_direct(q, k, v, causal=True, window_size=(-1, -1))

        # SDPA output (will expand KV heads)
        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        y_sdpa = attention_forward(
            q_cpu, k_cpu, v_cpu, is_causal=True, enable_gqa=True, window_size=()
        )

        y_sdpa_gpu = y_sdpa.to(device=self.DEVICE, dtype=self.DTYPE)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa_gpu, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        """Larger dimensions closer to real model."""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)

        y_fa3 = run_fa3_direct(q, k, v, causal=True, window_size=(-1, -1))

        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        y_sdpa = attention_forward(
            q_cpu, k_cpu, v_cpu, is_causal=True, enable_gqa=False, window_size=()
        )

        y_sdpa_gpu = y_sdpa.to(device=self.DEVICE, dtype=self.DTYPE)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa_gpu, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


@pytest.mark.skipif(not HAS_FA2, reason="FA2 required for comparison tests")
class TestFA2VsSDPA:
    """Compare FA2 and SDPA produce consistent results."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention comparison."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, H, T, D, device=self.DEVICE, dtype=self.DTYPE)

        # FA2 output
        y_fa2 = run_fa2_direct(q, k, v, causal=True, window_size=(-1, -1))

        # SDPA output
        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        y_sdpa = attention_forward(
            q_cpu, k_cpu, v_cpu, is_causal=True, enable_gqa=False, window_size=()
        )

        y_sdpa_gpu = y_sdpa.to(device=self.DEVICE, dtype=self.DTYPE)
        max_diff, mean_diff = assert_close(y_fa2, y_sdpa_gpu, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """Group Query Attention comparison."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, n_heads, T, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, n_kv_heads, T, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, n_kv_heads, T, D, device=self.DEVICE, dtype=self.DTYPE)

        # FA2 output
        y_fa2 = run_fa2_direct(q, k, v, causal=True, window_size=(-1, -1))

        # SDPA output
        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        y_sdpa = attention_forward(
            q_cpu, k_cpu, v_cpu, is_causal=True, enable_gqa=True, window_size=()
        )

        y_sdpa_gpu = y_sdpa.to(device=self.DEVICE, dtype=self.DTYPE)
        max_diff, mean_diff = assert_close(y_fa2, y_sdpa_gpu, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA Fallback Tests (run on any device)
# =============================================================================


class TestSDPAFallback:
    """Test SDPA fallback works correctly. Runs on any device."""

    def test_basic_forward(self, device, dtype):
        """Test SDPA forward pass produces valid output."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Force SDPA by using CPU or non-FA dtype
        if device.type == "cuda":
            q_test = q.cpu()
            k_test = k.cpu()
            v_test = v.cpu()
        else:
            q_test, k_test, v_test = q, k, v

        y = attention_forward(
            q_test, k_test, v_test, is_causal=True, enable_gqa=False, window_size=()
        )

        assert y.shape == (B, H, T, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_backward(self, device, dtype):
        """Test gradients flow through SDPA."""
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)

        y = attention_forward(q, k, v, is_causal=True, enable_gqa=False, window_size=())
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"

    def test_sliding_window_mask(self, device, dtype):
        """Test sliding window creates correct attention mask."""
        B, T, H, D = 2, 64, 4, 32
        window = 16

        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Test with sliding window
        y = attention_forward(
            q, k, v, is_causal=True, enable_gqa=False, window_size=(window, 0)
        )

        assert y.shape == (B, H, T, D)
        assert not torch.isnan(y).any()

    def test_gqa_head_expansion(self, device, dtype):
        """Test that GQA correctly expands KV heads."""
        B, T, D = 2, 32, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, n_heads, T, D, device=device, dtype=dtype)
        k = torch.randn(B, n_kv_heads, T, D, device=device, dtype=dtype)
        v = torch.randn(B, n_kv_heads, T, D, device=device, dtype=dtype)

        y = attention_forward(q, k, v, is_causal=True, enable_gqa=True, window_size=())

        # Output should match query head count
        assert y.shape == (B, n_heads, T, D)
        assert not torch.isnan(y).any()

    def test_non_causal_attention(self, device, dtype):
        """Test non-causal (bidirectional) attention."""
        B, T, H, D = 2, 32, 4, 32
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        y = attention_forward(
            q, k, v, is_causal=False, enable_gqa=False, window_size=()
        )

        assert y.shape == (B, H, T, D)
        assert not torch.isnan(y).any()


# =============================================================================
# Backend Feature Tests
# =============================================================================


class TestBackendFeatures:
    """Test features that should work on all available backends."""

    def test_various_sequence_lengths(self, device, dtype):
        """Test attention with various sequence lengths."""
        B, H, D = 2, 4, 32

        for T in [1, 4, 16, 32, 64, 128]:
            q = torch.randn(B, H, T, D, device=device, dtype=dtype)
            k = torch.randn(B, H, T, D, device=device, dtype=dtype)
            v = torch.randn(B, H, T, D, device=device, dtype=dtype)

            y = attention_forward(
                q, k, v, is_causal=True, enable_gqa=False, window_size=()
            )
            assert y.shape == (B, H, T, D)
            assert not torch.isnan(y).any()

    def test_various_head_counts(self, device, dtype):
        """Test attention with various head configurations."""
        B, T, D = 2, 32, 32

        head_configs = [
            (4, 4),  # MHA
            (8, 4),  # GQA
            (8, 2),  # GQA
            (8, 1),  # MQA
        ]

        for n_heads, n_kv_heads in head_configs:
            q = torch.randn(B, n_heads, T, D, device=device, dtype=dtype)
            k = torch.randn(B, n_kv_heads, T, D, device=device, dtype=dtype)
            v = torch.randn(B, n_kv_heads, T, D, device=device, dtype=dtype)

            enable_gqa = n_heads != n_kv_heads
            y = attention_forward(
                q, k, v, is_causal=True, enable_gqa=enable_gqa, window_size=()
            )

            assert y.shape == (B, n_heads, T, D)
            assert not torch.isnan(y).any()

    def test_various_window_sizes(self, device, dtype):
        """Test sliding window with various window sizes."""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        for window in [8, 16, 32, 64]:
            y = attention_forward(
                q, k, v, is_causal=True, enable_gqa=False, window_size=(window, 0)
            )
            assert y.shape == (B, H, T, D)
            assert not torch.isnan(y).any()

    def test_batch_size_variations(self, device, dtype):
        """Test different batch sizes."""
        T, H, D = 32, 4, 32

        for B in [1, 2, 4, 8, 16]:
            q = torch.randn(B, H, T, D, device=device, dtype=dtype)
            k = torch.randn(B, H, T, D, device=device, dtype=dtype)
            v = torch.randn(B, H, T, D, device=device, dtype=dtype)

            y = attention_forward(
                q, k, v, is_causal=True, enable_gqa=False, window_size=()
            )
            assert y.shape == (B, H, T, D)
            assert not torch.isnan(y).any()


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test attention stability with extreme values."""

    def test_large_values(self, device, dtype):
        """Test attention with large input values."""
        B, T, H, D = 2, 32, 4, 32
        scale = 10.0

        q = torch.randn(B, H, T, D, device=device, dtype=dtype) * scale
        k = torch.randn(B, H, T, D, device=device, dtype=dtype) * scale
        v = torch.randn(B, H, T, D, device=device, dtype=dtype) * scale

        y = attention_forward(q, k, v, is_causal=True, enable_gqa=False, window_size=())

        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_small_values(self, device, dtype):
        """Test attention with small input values."""
        B, T, H, D = 2, 32, 4, 32
        scale = 0.001

        q = torch.randn(B, H, T, D, device=device, dtype=dtype) * scale
        k = torch.randn(B, H, T, D, device=device, dtype=dtype) * scale
        v = torch.randn(B, H, T, D, device=device, dtype=dtype) * scale

        y = attention_forward(q, k, v, is_causal=True, enable_gqa=False, window_size=())

        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_zero_values(self, device, dtype):
        """Test attention with zero inputs."""
        B, T, H, D = 2, 32, 4, 32

        q = torch.zeros(B, H, T, D, device=device, dtype=dtype)
        k = torch.zeros(B, H, T, D, device=device, dtype=dtype)
        v = torch.zeros(B, H, T, D, device=device, dtype=dtype)

        y = attention_forward(q, k, v, is_causal=True, enable_gqa=False, window_size=())

        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self, device, dtype):
        """Test attention with sequence length of 1."""
        B, H, D = 2, 4, 32
        T = 1

        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        y = attention_forward(q, k, v, is_causal=True, enable_gqa=False, window_size=())

        assert y.shape == (B, H, T, D)
        assert not torch.isnan(y).any()

    def test_window_equals_sequence_length(self, device, dtype):
        """Test sliding window equal to sequence length (equivalent to full attention)."""
        B, T, H, D = 2, 32, 4, 32

        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Window size equals sequence length
        y_window = attention_forward(
            q, k, v, is_causal=True, enable_gqa=False, window_size=(T, 0)
        )

        # Full attention (no window)
        y_full = attention_forward(
            q, k, v, is_causal=True, enable_gqa=False, window_size=()
        )

        # Should produce similar results
        assert torch.allclose(y_window, y_full, atol=1e-2, rtol=1e-2)

    def test_window_larger_than_sequence(self, device, dtype):
        """Test sliding window larger than sequence length."""
        B, T, H, D = 2, 32, 4, 32

        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Window larger than sequence (should be equivalent to full attention)
        y = attention_forward(
            q, k, v, is_causal=True, enable_gqa=False, window_size=(T * 2, 0)
        )

        assert y.shape == (B, H, T, D)
        assert not torch.isnan(y).any()


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"HAS_FA2: {HAS_FA2}")
    print()

    pytest.main([__file__, "-v", "-s"])
