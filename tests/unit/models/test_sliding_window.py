"""
Unit tests for sliding window attention implementation.

Tests cover:
- Different window patterns (L, SL, SSSL, SSSSL)
- Window size verification
- Final layer full context requirement
- Forward pass with sliding windows
- Pattern validation
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT


class TestSlidingWindowAttention:
    """Tests for sliding window attention implementation."""

    @pytest.mark.parametrize("window_pattern", ["L", "SL", "SSSL", "SSSSL"])
    def test_window_patterns(self, window_pattern, device, dtype):
        """Test sliding window attention with different patterns."""
        # Create config with window pattern
        config = GPTConfig(
            vocab_size=50266,
            n_layer=6,
            n_head=4,
            n_embed=256,
            block_size=512,
            window_pattern=window_pattern,
        )

        # Create model
        model = GPT(config, master_process=False).to(device).to(dtype)

        # Verify final layer always has full context (block_size, 0)
        assert model.window_sizes[-1] == (
            config.block_size,
            0,
        ), f"Final layer must have full context ({config.block_size}, 0)!"

        # Test forward pass
        batch_size = 2
        seq_len = 64

        # Create dummy input
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        targets = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        # Forward pass
        logits, loss = model(input_ids, targets=targets)

        # Verify output shapes
        assert logits.shape == (
            batch_size,
            seq_len,
            config.vocab_size,
        ), f"Expected logits shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
        assert loss is not None and loss.numel() == 1, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_window_size_calculation(self, device, dtype):
        """Test that window sizes are calculated correctly."""
        config = GPTConfig(
            n_layer=6,
            n_head=4,
            n_embed=256,
            block_size=512,
            window_pattern="SSSL",  # Small, Small, Small, Large
        )

        model = GPT(config, master_process=False).to(device).to(dtype)

        # Check window sizes match pattern
        # Pattern "SSSL" repeats: S, S, S, L, S, S
        window_sizes = model.window_sizes
        assert len(window_sizes) == 6

        # Last layer should always have full context (block_size, 0)
        assert window_sizes[-1] == (
            config.block_size,
            0,
        ), f"Last layer must have full context ({config.block_size}, 0)"

        # Check that at least one layer has sliding window (left < block_size)
        short_window = config.block_size // 2
        has_sliding_window = any(ws[0] == short_window for ws in window_sizes[:-1])
        assert has_sliding_window, "Should have at least one sliding window layer"

    def test_final_layer_full_context(self, device, dtype):
        """Test that final layer always has full context regardless of pattern."""
        patterns = ["S", "SS", "SSS", "SSSS"]

        for pattern in patterns:
            config = GPTConfig(
                n_layer=4,
                n_head=4,
                n_embed=128,
                block_size=256,
                window_pattern=pattern,
            )

            model = GPT(config, master_process=False).to(device).to(dtype)

            # Final layer must have full context (block_size, 0)
            assert model.window_sizes[-1] == (
                config.block_size,
                0,
            ), f"Pattern '{pattern}': Final layer must have full context ({config.block_size}, 0)!"


class TestWindowPatternValidation:
    """Tests for window pattern validation."""

    @pytest.mark.parametrize(
        "valid_pattern", ["L", "S", "SL", "SSSL", "SSSSL", "LL", "SS"]
    )
    def test_valid_patterns_accepted(self, valid_pattern):
        """Test that valid window patterns are accepted."""
        config = GPTConfig(n_layer=4, window_pattern=valid_pattern)
        model = GPT(config, master_process=False)
        # Should not raise an error
        assert model is not None

    @pytest.mark.parametrize("invalid_pattern", ["X", "SLX", "123", "M"])
    def test_invalid_patterns_rejected(self, invalid_pattern):
        """Test that invalid window patterns are rejected."""
        config = GPTConfig(n_layer=4, window_pattern=invalid_pattern)
        with pytest.raises(AssertionError):
            GPT(config, master_process=False)

    def test_empty_pattern_uses_default(self):
        """Test that empty pattern uses default (full attention)."""
        config = GPTConfig(n_layer=4, window_pattern=None)
        model = GPT(config, master_process=False)

        # All layers should have full context (block_size, 0)
        for ws in model.window_sizes:
            assert ws == (
                config.block_size,
                0,
            ), f"Default pattern should have full context ({config.block_size}, 0) for all layers"


class TestSlidingWindowMemory:
    """Tests for memory efficiency of sliding windows."""

    def test_window_affects_attention_computation(self, device, dtype):
        """Test that sliding window actually restricts attention."""
        config_full = GPTConfig(
            n_layer=2,
            n_head=4,
            n_embed=128,
            block_size=256,
            window_pattern="L",  # Full attention
        )

        config_window = GPTConfig(
            n_layer=2,
            n_head=4,
            n_embed=128,
            block_size=256,
            window_pattern="S",  # Sliding window
        )

        model_full = GPT(config_full, master_process=False).to(device).to(dtype)
        model_window = GPT(config_window, master_process=False).to(device).to(dtype)

        # Verify different window sizes (block_size = full context, smaller = sliding window)
        assert model_full.window_sizes[0] == (
            config_full.block_size,
            0,
        ), "Full attention should be (block_size, 0)"
        assert (
            model_window.window_sizes[0][0] < config_window.block_size
        ), "Sliding window should have window size < block_size"

        # Both should produce valid outputs (different values expected)
        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(
            0, config_full.vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            logits_full, _ = model_full(input_ids)
            logits_window, _ = model_window(input_ids)

        assert logits_full.shape == logits_window.shape
        assert not torch.isnan(logits_full).any()
        assert not torch.isnan(logits_window).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
