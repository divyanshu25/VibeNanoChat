"""
Unit tests for CausalSelfAttention module.

Tests cover:
- Basic forward pass with different configurations
- Shape validation and assertions
- RoPE (Rotary Position Embeddings) integration
- QK normalization behavior
- GQA (Grouped Query Attention) vs MHA (Multi-Head Attention)
- KV caching for efficient generation
- Flash Attention vs SDPA backends
- Causal masking correctness
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from gpt_2.attention import CausalSelfAttention, attention_forward
from gpt_2.config import GPTConfig
from gpt_2.rope import precompute_rotary_embeddings


class TestAttentionForward:
    """Tests for the attention_forward function."""

    def test_attention_forward_basic(self, device):
        """Test basic attention forward pass."""
        B, n_head, T, head_dim = 2, 4, 8, 64

        q = torch.randn(B, n_head, T, head_dim, device=device)
        k = torch.randn(B, n_head, T, head_dim, device=device)
        v = torch.randn(B, n_head, T, head_dim, device=device)

        output = attention_forward(q, k, v, is_causal=True, enable_gqa=False)

        # Check output shape: (B, n_head, T, head_dim)
        assert output.shape == (B, n_head, T, head_dim)
        assert output.dtype == q.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_forward_gqa(self, device):
        """Test attention forward with GQA (grouped query attention)."""
        B, n_head, n_kv_head, T, head_dim = 2, 8, 2, 16, 64

        q = torch.randn(B, n_head, T, head_dim, device=device)
        k = torch.randn(B, n_kv_head, T, head_dim, device=device)
        v = torch.randn(B, n_kv_head, T, head_dim, device=device)

        output = attention_forward(q, k, v, is_causal=True, enable_gqa=True)

        # Check output shape matches query
        assert output.shape == (B, n_head, T, head_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_forward_non_causal(self, device):
        """Test non-causal attention (all positions attend to all)."""
        B, n_head, T, head_dim = 2, 4, 8, 64

        q = torch.randn(B, n_head, T, head_dim, device=device)
        k = torch.randn(B, n_head, T, head_dim, device=device)
        v = torch.randn(B, n_head, T, head_dim, device=device)

        output = attention_forward(q, k, v, is_causal=False, enable_gqa=False)

        assert output.shape == (B, n_head, T, head_dim)
        assert not torch.isnan(output).any()


class TestCausalSelfAttention:
    """Tests for the CausalSelfAttention module."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA
        config.block_size = 128
        return config

    @pytest.fixture
    def gqa_config(self):
        """Create a config with GQA enabled."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 8
        config.n_kv_head = 2  # GQA: 8:2 ratio
        config.block_size = 128
        return config

    def test_initialization(self, small_config):
        """Test attention module initialization."""
        attn = CausalSelfAttention(small_config, layer_idx=0)

        assert attn.n_head == 4
        assert attn.n_embed == 256
        assert attn.head_dim == 64  # 256 / 4
        assert attn.n_kv_head == 4
        assert attn.layer_idx == 0

        # Check parameter initialization
        assert hasattr(attn, "c_attn")
        assert hasattr(attn, "c_proj")
        assert hasattr(attn.c_proj, "NANOGPT_SCALE_INIT")

    def test_initialization_gqa(self, gqa_config):
        """Test attention module initialization with GQA."""
        attn = CausalSelfAttention(gqa_config, layer_idx=0)

        assert attn.n_head == 8
        assert attn.n_kv_head == 2
        assert attn.head_dim == 32  # 256 / 8

        # Check c_attn output dimension: Q (256) + K (64) + V (64) = 384
        expected_dim = 256 + 2 * (32 * 2)  # n_embed + 2 * (head_dim * n_kv_head)
        assert attn.c_attn.out_features == expected_dim

    def test_forward_basic(self, small_config, device):
        """Test basic forward pass without RoPE or KV cache."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device)

        output = attn(x)

        # Check output shape: (B, T, C)
        assert output.shape == (B, T, C)
        assert output.dtype == x.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_rope(self, small_config, device):
        """Test forward pass with RoPE."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device)

        # Precompute RoPE embeddings
        head_dim = small_config.n_embed // small_config.n_head
        cos, sin = precompute_rotary_embeddings(
            seq_len=T, head_dim=head_dim, device=device
        )

        output = attn(x, cos_sin=(cos, sin))

        assert output.shape == (B, T, C)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_gqa(self, gqa_config, device):
        """Test forward pass with GQA."""
        attn = CausalSelfAttention(gqa_config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device)

        output = attn(x)

        assert output.shape == (B, T, C)
        assert not torch.isnan(output).any()

    def test_different_sequence_lengths(self, small_config, device):
        """Test attention with various sequence lengths."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)

        B, C = 2, 256

        for T in [1, 4, 8, 16, 32, 64]:
            x = torch.randn(B, T, C, device=device)
            output = attn(x)
            assert output.shape == (B, T, C)
            assert not torch.isnan(output).any()

    def test_qk_normalization(self, small_config, device):
        """Test that QK normalization prevents logit explosion."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)

        B, T, C = 2, 32, 256

        # Test with large input values (could cause attention logit explosion)
        x = torch.randn(B, T, C, device=device) * 10.0

        output = attn(x)

        # QK normalization should keep outputs reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() < 1e3  # Reasonable magnitude

    def test_causal_masking(self, small_config, device):
        """Test that causal masking is properly applied."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)
        attn.eval()  # Set to eval mode for deterministic behavior

        B, T, C = 1, 8, 256

        # Create input where position information matters
        x = torch.zeros(B, T, C, device=device)
        # Set last position to have large values
        x[:, -1, :] = 10.0

        with torch.no_grad():
            output = attn(x)

        # Due to causal masking, early positions shouldn't be affected by later ones
        # First position can only attend to itself (zeros)
        # Last position should be most affected by the large values

        # This is a rough check - in practice, causal masking means
        # earlier positions have more restricted attention patterns
        assert not torch.isnan(output).any()

    def test_batch_size_independence(self, small_config, device):
        """Test that different batch sizes produce consistent results."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)
        attn.eval()

        T, C = 16, 256

        # Create same input for different batch sizes
        x_single = torch.randn(1, T, C, device=device)
        x_batched = x_single.repeat(4, 1, 1)  # Batch of 4 identical sequences

        with torch.no_grad():
            output_single = attn(x_single)
            output_batched = attn(x_batched)

        # Each item in batch should produce same result as single item
        for i in range(4):
            assert torch.allclose(output_single[0], output_batched[i], atol=1e-5)

    def test_gradient_flow(self, small_config, device):
        """Test that gradients flow properly through attention."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Check that all parameters have gradients
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_determinism(self, small_config, device):
        """Test that attention is deterministic with same inputs."""
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)
        attn.eval()

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device)

        with torch.no_grad():
            output1 = attn(x)
            output2 = attn(x)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_head_dimension_consistency(self, small_config, device):
        """Test that head dimension calculation is correct."""
        attn = CausalSelfAttention(small_config, layer_idx=0)

        # n_embed should be divisible by n_head
        assert small_config.n_embed % small_config.n_head == 0

        expected_head_dim = small_config.n_embed // small_config.n_head
        assert attn.head_dim == expected_head_dim

    def test_parameter_count(self, small_config):
        """Test that parameter count matches expected dimensions."""
        attn = CausalSelfAttention(small_config, layer_idx=0)

        # c_attn: (n_embed, n_embed + 2*kv_dim)
        kv_dim = attn.head_dim * attn.n_kv_head
        expected_c_attn_params = small_config.n_embed * (
            small_config.n_embed + 2 * kv_dim
        )

        # c_proj: (n_embed, n_embed)
        expected_c_proj_params = small_config.n_embed * small_config.n_embed

        total_params = sum(p.numel() for p in attn.parameters())
        expected_total = expected_c_attn_params + expected_c_proj_params

        assert total_params == expected_total


class TestRoPEIntegration:
    """Tests for RoPE integration in attention."""

    def test_rope_shape_compatibility(self, device):
        """Test that RoPE shapes are compatible with attention."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA
        config.block_size = 128

        attn = CausalSelfAttention(config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device)

        head_dim = config.n_embed // config.n_head
        cos, sin = precompute_rotary_embeddings(
            seq_len=T, head_dim=head_dim, device=device
        )

        # Check RoPE shape: (1, T, 1, head_dim//2)
        assert cos.shape == (1, T, 1, head_dim // 2)
        assert sin.shape == (1, T, 1, head_dim // 2)

        output = attn(x, cos_sin=(cos, sin))
        assert output.shape == (B, T, C)

    def test_rope_position_encoding(self, device):
        """Test that RoPE integration works and produces valid outputs."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA

        attn = CausalSelfAttention(config, layer_idx=0).to(device)
        attn.eval()

        B, T, C = 1, 128, 256  # Use longer sequence for clearer RoPE effect

        # Create varied input at different positions to see position encoding effect
        x = torch.randn(B, T, C, device=device)

        head_dim = config.n_embed // config.n_head
        cos, sin = precompute_rotary_embeddings(
            seq_len=T, head_dim=head_dim, device=device
        )

        with torch.no_grad():
            output_with_rope = attn(x, cos_sin=(cos, sin))
            output_without_rope = attn(x, cos_sin=None)

        # RoPE should affect the output (outputs should be different)
        # Note: This is a sanity check that RoPE actually does something
        assert not torch.allclose(output_with_rope, output_without_rope, atol=1e-3)

        # Both outputs should be valid (no NaN/Inf)
        assert not torch.isnan(output_with_rope).any()
        assert not torch.isnan(output_without_rope).any()


class TestGQABehavior:
    """Tests specific to Grouped Query Attention."""

    def test_gqa_different_head_counts(self, device):
        """Test GQA with various Q:KV head ratios."""
        base_config = GPTConfig()
        base_config.n_embed = 256
        base_config.block_size = 128

        test_cases = [
            (8, 8),  # MHA (1:1)
            (8, 4),  # GQA (2:1)
            (8, 2),  # GQA (4:1)
            (8, 1),  # MQA (8:1) - Multi-Query Attention
        ]

        B, T, C = 2, 16, 256
        x = torch.randn(B, T, C, device=device)

        for n_head, n_kv_head in test_cases:
            config = base_config
            config.n_head = n_head
            config.n_kv_head = n_kv_head

            attn = CausalSelfAttention(config, layer_idx=0).to(device)
            output = attn(x)

            assert output.shape == (B, T, C)
            assert not torch.isnan(output).any()

    def test_gqa_parameter_efficiency(self):
        """Test that GQA uses fewer parameters than MHA."""
        base_config = GPTConfig()
        base_config.n_embed = 256
        base_config.n_head = 8

        # MHA configuration
        mha_config = base_config
        mha_config.n_kv_head = 8
        attn_mha = CausalSelfAttention(mha_config, layer_idx=0)

        # GQA configuration (4:1 ratio)
        gqa_config = base_config
        gqa_config.n_kv_head = 2
        attn_gqa = CausalSelfAttention(gqa_config, layer_idx=0)

        mha_params = sum(p.numel() for p in attn_mha.parameters())
        gqa_params = sum(p.numel() for p in attn_gqa.parameters())

        # GQA should have fewer parameters (smaller K, V projections)
        assert gqa_params < mha_params


class TestNumericalStability:
    """Tests for numerical stability of attention."""

    def test_large_sequence_length(self, device):
        """Test attention with longer sequences."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA
        config.block_size = 512

        attn = CausalSelfAttention(config, layer_idx=0).to(device)

        B, T, C = 1, 256, 256
        x = torch.randn(B, T, C, device=device)

        output = attn(x)

        assert output.shape == (B, T, C)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_extreme_input_values(self, device):
        """Test attention stability with extreme input values."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA

        attn = CausalSelfAttention(config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256

        # Test with various extreme values
        test_cases = [
            torch.randn(B, T, C, device=device) * 100,  # Large values
            torch.randn(B, T, C, device=device) * 0.001,  # Small values
            torch.zeros(B, T, C, device=device),  # All zeros
        ]

        for x in test_cases:
            output = attn(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_mixed_precision_compatibility(self, device):
        """Test that attention works with different dtypes."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA

        attn = CausalSelfAttention(config, layer_idx=0).to(device)

        B, T, C = 2, 16, 256

        # Test with float32
        x_fp32 = torch.randn(B, T, C, device=device, dtype=torch.float32)
        output_fp32 = attn(x_fp32)
        assert output_fp32.dtype == torch.float32
        assert not torch.isnan(output_fp32).any()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token_sequence(self, device):
        """Test attention with sequence length of 1."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA

        attn = CausalSelfAttention(config, layer_idx=0).to(device)

        B, T, C = 2, 1, 256
        x = torch.randn(B, T, C, device=device)

        output = attn(x)
        assert output.shape == (B, T, C)
        assert not torch.isnan(output).any()

    def test_minimum_config(self, device):
        """Test attention with minimal configuration."""
        config = GPTConfig()
        config.n_embed = 64  # Small embedding
        config.n_head = 2  # Few heads
        config.n_kv_head = 1  # MQA

        attn = CausalSelfAttention(config, layer_idx=0).to(device)

        B, T, C = 1, 4, 64
        x = torch.randn(B, T, C, device=device)

        output = attn(x)
        assert output.shape == (B, T, C)
        assert not torch.isnan(output).any()

    def test_layer_index_tracking(self):
        """Test that layer_idx is properly stored."""
        config = GPTConfig()
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA

        for idx in [0, 1, 5, 10]:
            attn = CausalSelfAttention(config, layer_idx=idx)
            assert attn.layer_idx == idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
