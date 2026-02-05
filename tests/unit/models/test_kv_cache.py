"""
Unit tests for KV Cache implementation and generation equivalence.

This test suite verifies:
1. Basic KV cache operations (insert, prefill, reset)
2. CRITICAL: Generations with and without KV cache produce identical results
3. KV cache works correctly with model forward passes
4. Cache grows dynamically when needed
5. Edge cases and boundary conditions

The most important tests verify that KV caching is purely an optimization
that doesn't change generation outputs - with or without cache, the model
should generate exactly the same tokens given the same seed.
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
from gpt_2.kv_cache import KVCache


class TestKVCacheBasicOperations:
    """Tests for basic KV cache functionality."""

    def test_initialization(self):
        """Test KV cache initialization with specified dimensions."""
        batch_size = 2
        num_heads = 4
        seq_len = 128
        head_dim = 64
        num_layers = 6

        kv_cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_layers=num_layers,
        )

        # Check initial state
        assert kv_cache.pos == 0
        assert kv_cache.kv_cache is None  # Lazy initialization
        assert kv_cache.kv_shape == (
            num_layers,
            2,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
        )

    def test_reset(self, device):
        """Test cache reset functionality."""
        kv_cache = KVCache(
            batch_size=1, num_heads=4, seq_len=64, head_dim=64, num_layers=2
        )

        # Create some dummy K, V tensors to trigger initialization
        k = torch.randn(1, 4, 10, 64, device=device)
        v = torch.randn(1, 4, 10, 64, device=device)

        # Insert some data
        kv_cache.insert_kv(0, k, v)
        kv_cache.insert_kv(1, k, v)  # Last layer advances position
        assert kv_cache.pos == 10

        # Reset should clear position but keep cache allocated
        kv_cache.reset()
        assert kv_cache.pos == 0
        assert kv_cache.kv_cache is not None  # Cache memory still exists

    def test_lazy_initialization(self, device):
        """Test that cache is lazily initialized on first insert."""
        kv_cache = KVCache(
            batch_size=1, num_heads=4, seq_len=64, head_dim=64, num_layers=2
        )

        # Initially not allocated
        assert kv_cache.kv_cache is None

        # First insert triggers allocation
        k = torch.randn(1, 4, 5, 64, device=device)
        v = torch.randn(1, 4, 5, 64, device=device)
        kv_cache.insert_kv(0, k, v)

        # Now allocated with correct dtype and device
        assert kv_cache.kv_cache is not None
        assert kv_cache.kv_cache.dtype == k.dtype
        assert kv_cache.kv_cache.device == k.device

    def test_insert_and_retrieve(self, device):
        """Test inserting K,V pairs and retrieving them."""
        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32
        num_layers = 3

        kv_cache = KVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_layers=num_layers,
        )

        # Insert first batch of tokens (5 tokens)
        k1 = torch.randn(batch_size, num_heads, 5, head_dim, device=device)
        v1 = torch.randn(batch_size, num_heads, 5, head_dim, device=device)

        # Insert for each layer
        for layer_idx in range(num_layers):
            k_cached, v_cached = kv_cache.insert_kv(layer_idx, k1, v1)
            # Should return all cached K,V so far
            assert k_cached.shape == (batch_size, num_heads, 5, head_dim)
            assert v_cached.shape == (batch_size, num_heads, 5, head_dim)

        # Position should be updated after last layer
        assert kv_cache.pos == 5

        # Insert second batch (3 more tokens)
        k2 = torch.randn(batch_size, num_heads, 3, head_dim, device=device)
        v2 = torch.randn(batch_size, num_heads, 3, head_dim, device=device)

        for layer_idx in range(num_layers):
            k_cached, v_cached = kv_cache.insert_kv(layer_idx, k2, v2)
            # Should return all 8 tokens (5 + 3)
            assert k_cached.shape == (batch_size, num_heads, 8, head_dim)
            assert v_cached.shape == (batch_size, num_heads, 8, head_dim)

        assert kv_cache.pos == 8

    def test_dynamic_growth(self, device):
        """Test that cache grows dynamically when needed."""
        # Start with small cache
        initial_seq_len = 16
        kv_cache = KVCache(
            batch_size=1,
            num_heads=4,
            seq_len=initial_seq_len,
            head_dim=64,
            num_layers=2,
        )

        # Insert tokens that exceed initial capacity
        k = torch.randn(1, 4, 20, 64, device=device)  # 20 > 16
        v = torch.randn(1, 4, 20, 64, device=device)

        # Should automatically grow
        kv_cache.insert_kv(0, k, v)
        kv_cache.insert_kv(1, k, v)

        # Cache should have grown
        assert kv_cache.kv_cache.size(4) > initial_seq_len
        assert kv_cache.pos == 20

    def test_prefill_from_single_to_batch(self, device):
        """Test prefilling batch cache from single-sequence cache."""
        # Create single-batch cache and fill it
        src_cache = KVCache(
            batch_size=1, num_heads=4, seq_len=64, head_dim=32, num_layers=2
        )

        k = torch.randn(1, 4, 10, 32, device=device)
        v = torch.randn(1, 4, 10, 32, device=device)

        for layer_idx in range(2):
            src_cache.insert_kv(layer_idx, k, v)

        assert src_cache.pos == 10

        # Create larger batch cache
        dst_cache = KVCache(
            batch_size=4, num_heads=4, seq_len=64, head_dim=32, num_layers=2
        )

        # Prefill should copy data
        dst_cache.prefill(src_cache)

        # Check position was copied
        assert dst_cache.pos == 10

        # Check data was copied (broadcasting to all batch positions)
        assert dst_cache.kv_cache is not None
        assert dst_cache.kv_cache.shape[2] == 4  # batch_size=4

    def test_prefill_validation(self, device):
        """Test that prefill validates dimensions correctly."""
        src_cache = KVCache(
            batch_size=1, num_heads=4, seq_len=32, head_dim=64, num_layers=2
        )

        # Initialize source cache
        k = torch.randn(1, 4, 5, 64, device=device)
        v = torch.randn(1, 4, 5, 64, device=device)
        src_cache.insert_kv(0, k, v)
        src_cache.insert_kv(1, k, v)

        # Test mismatched dimensions
        with pytest.raises(AssertionError):
            # Different number of layers
            dst_cache = KVCache(
                batch_size=1, num_heads=4, seq_len=64, head_dim=64, num_layers=3
            )
            dst_cache.prefill(src_cache)

        with pytest.raises(AssertionError):
            # Different number of heads
            dst_cache = KVCache(
                batch_size=1, num_heads=8, seq_len=64, head_dim=64, num_layers=2
            )
            dst_cache.prefill(src_cache)

        with pytest.raises(AssertionError):
            # Different head dimension
            dst_cache = KVCache(
                batch_size=1, num_heads=4, seq_len=64, head_dim=32, num_layers=2
            )
            dst_cache.prefill(src_cache)

        with pytest.raises(AssertionError):
            # Destination too small to hold source data
            dst_cache = KVCache(
                batch_size=1, num_heads=4, seq_len=4, head_dim=64, num_layers=2
            )
            dst_cache.prefill(src_cache)


class TestKVCacheWithModel:
    """Tests for KV cache integration with GPT model."""

    @pytest.fixture
    def small_model(self, device):
        """Create a small GPT model for testing."""
        config = GPTConfig()
        config.vocab_size = 256  # Small vocab for testing
        config.n_embed = 128
        config.n_head = 4
        config.n_kv_head = 4  # MHA: same as n_head
        config.n_layer = 2
        config.block_size = 64
        config.use_qk_norm = True

        model = GPT(config, master_process=False, pad_vocab_size_to=64).to(device)
        model.eval()
        return model

    def test_model_forward_with_cache(self, small_model, device):
        """Test model forward pass with KV cache."""
        model = small_model
        config = model.config

        # Create KV cache (use n_kv_head for cache, not n_head)
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.block_size,
            head_dim=config.n_embed // config.n_head,
            num_layers=config.n_layer,
        )

        # First pass: process multiple tokens (prefill)
        input_ids = torch.randint(0, 256, (1, 10), device=device)

        with torch.no_grad():
            logits1, _ = model(input_ids, kv_cache=kv_cache)

        assert logits1.shape == (1, 10, config.vocab_size)
        assert kv_cache.pos == 10

        # Second pass: process one more token
        next_token = torch.randint(0, 256, (1, 1), device=device)

        with torch.no_grad():
            logits2, _ = model(next_token, kv_cache=kv_cache)

        assert logits2.shape == (1, 1, config.vocab_size)
        assert kv_cache.pos == 11

    def test_cache_position_tracking(self, small_model, device):
        """Test that cache position is tracked correctly across layers."""
        model = small_model
        config = model.config

        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.block_size,
            head_dim=config.n_embed // config.n_head,
            num_layers=config.n_layer,
        )

        # Process tokens one at a time
        for i in range(5):
            token = torch.randint(0, 256, (1, 1), device=device)
            with torch.no_grad():
                model(token, kv_cache=kv_cache)
            assert kv_cache.pos == i + 1


class TestGenerationEquivalence:
    """
    CRITICAL TESTS: Verify that KV cache produces identical generation results.

    These tests ensure that KV caching is purely a performance optimization
    that does not change the model's outputs. With the same seed, generations
    with and without KV cache must be exactly identical.
    """

    @pytest.fixture
    def test_model(self, device):
        """Create a model for generation testing."""
        config = GPTConfig()
        config.vocab_size = 512  # Reasonable vocab for testing
        config.n_embed = 256
        config.n_head = 4
        config.n_kv_head = 4  # MHA: same as n_head
        config.n_layer = 3
        config.block_size = 128
        config.use_qk_norm = True

        model = GPT(config, master_process=False, pad_vocab_size_to=64).to(device)
        model.eval()
        return model

    def generate_with_kv_cache(
        self, model, prompt_ids, max_tokens, temperature, seed, device
    ):
        """Generate text using KV cache (efficient)."""
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        config = model.config
        generated_ids = prompt_ids.clone()

        # Create KV cache (use n_kv_head for cache, not n_head)
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.block_size,
            head_dim=config.n_embed // config.n_head,
            num_layers=config.n_layer,
        )

        with torch.no_grad():
            # Prefill: process prompt all at once
            logits, _ = model(prompt_ids.unsqueeze(0), kv_cache=kv_cache)
            next_logits = logits[0, -1, : config.vocab_size]  # Crop to actual vocab

            # Sample first token
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_logits.argmax().item()

            generated_ids = torch.cat(
                [generated_ids, torch.tensor([next_token], device=device)]
            )

            # Decode: generate remaining tokens one at a time
            for _ in range(max_tokens - 1):
                token_tensor = torch.tensor([[next_token]], device=device)
                logits, _ = model(token_tensor, kv_cache=kv_cache)
                next_logits = logits[0, -1, : config.vocab_size]

                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = next_logits.argmax().item()

                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([next_token], device=device)]
                )

        return generated_ids

    def generate_without_kv_cache(
        self, model, prompt_ids, max_tokens, temperature, seed, device
    ):
        """Generate text without KV cache (recompute everything each step)."""
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        config = model.config
        generated_ids = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                # Reprocess entire sequence from scratch (inefficient but correct)
                logits, _ = model(generated_ids.unsqueeze(0), kv_cache=None)
                next_logits = logits[0, -1, : config.vocab_size]

                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = next_logits.argmax().item()

                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([next_token], device=device)]
                )

        return generated_ids

    @pytest.mark.parametrize("seed", [42, 123, 999, 2024])
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_generation_equivalence_different_seeds_and_temps(
        self, test_model, device, seed, temperature
    ):
        """
        CRITICAL TEST: Verify KV cache doesn't change generation outputs.

        With the same seed, generations with and without KV cache must produce
        exactly the same token sequence. This holds for various seeds and
        temperatures (including deterministic temperature=0.0).
        """
        model = test_model
        max_tokens = 20

        # Create random prompt
        prompt_ids = torch.randint(0, 512, (10,), device=device)

        # Generate with KV cache
        output_with_cache = self.generate_with_kv_cache(
            model, prompt_ids, max_tokens, temperature, seed, device
        )

        # Generate without KV cache (same seed)
        output_without_cache = self.generate_without_kv_cache(
            model, prompt_ids, max_tokens, temperature, seed, device
        )

        # CRITICAL: Outputs must be EXACTLY identical
        assert torch.equal(output_with_cache, output_without_cache), (
            f"Generation mismatch with seed={seed}, temperature={temperature}!\n"
            f"With cache:    {output_with_cache.tolist()}\n"
            f"Without cache: {output_without_cache.tolist()}\n"
            f"This means KV cache is changing model behavior!"
        )

    def test_generation_equivalence_various_prompt_lengths(self, test_model, device):
        """Test generation equivalence with different prompt lengths."""
        model = test_model
        seed = 42
        temperature = 0.8
        max_tokens = 15

        for prompt_len in [1, 5, 10, 20, 30]:
            prompt_ids = torch.randint(0, 512, (prompt_len,), device=device)

            output_with_cache = self.generate_with_kv_cache(
                model, prompt_ids, max_tokens, temperature, seed, device
            )

            output_without_cache = self.generate_without_kv_cache(
                model, prompt_ids, max_tokens, temperature, seed, device
            )

            assert torch.equal(
                output_with_cache, output_without_cache
            ), f"Mismatch with prompt_len={prompt_len}"

    def test_deterministic_generation_with_temp_zero(self, test_model, device):
        """
        Test that temperature=0.0 produces deterministic results.

        With temperature=0 (greedy decoding), outputs should be identical
        regardless of seed, both with and without KV cache.
        """
        model = test_model
        max_tokens = 15
        prompt_ids = torch.randint(0, 512, (10,), device=device)

        # Generate multiple times with different seeds but temp=0
        outputs_with_cache = []
        outputs_without_cache = []

        for seed in [1, 42, 123, 999]:
            outputs_with_cache.append(
                self.generate_with_kv_cache(
                    model, prompt_ids, max_tokens, 0.0, seed, device
                )
            )
            outputs_without_cache.append(
                self.generate_without_kv_cache(
                    model, prompt_ids, max_tokens, 0.0, seed, device
                )
            )

        # All outputs should be identical (deterministic greedy decoding)
        for i in range(len(outputs_with_cache) - 1):
            assert torch.equal(outputs_with_cache[i], outputs_with_cache[i + 1])
            assert torch.equal(outputs_without_cache[i], outputs_without_cache[i + 1])

        # And with-cache should match without-cache
        assert torch.equal(outputs_with_cache[0], outputs_without_cache[0])

    def test_stochastic_generation_produces_variation(self, test_model, device):
        """
        Test that temperature > 0 produces different outputs with different seeds.

        This is a sanity check that our sampling is actually working.
        """
        model = test_model
        max_tokens = 20
        temperature = 1.0
        prompt_ids = torch.randint(0, 512, (10,), device=device)

        outputs = []
        for seed in [1, 42, 123, 999, 2024, 2025, 2026, 2027]:
            output = self.generate_with_kv_cache(
                model, prompt_ids, max_tokens, temperature, seed, device
            )
            outputs.append(output)

        # With temperature > 0, different seeds should produce different outputs
        unique_outputs = len(set(tuple(o.tolist()) for o in outputs))
        assert unique_outputs > 1, (
            "All seeds produced identical outputs with temperature > 0. "
            "This suggests sampling is not working correctly."
        )

    def test_long_generation_equivalence(self, test_model, device):
        """Test equivalence for longer generation sequences."""
        model = test_model
        seed = 42
        temperature = 0.7
        max_tokens = 50  # Longer generation
        prompt_ids = torch.randint(0, 512, (5,), device=device)

        output_with_cache = self.generate_with_kv_cache(
            model, prompt_ids, max_tokens, temperature, seed, device
        )

        output_without_cache = self.generate_without_kv_cache(
            model, prompt_ids, max_tokens, temperature, seed, device
        )

        assert torch.equal(
            output_with_cache, output_without_cache
        ), "Mismatch in long generation (50 tokens)"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def tiny_model(self, device):
        """Create a tiny model for edge case testing."""
        config = GPTConfig()
        config.vocab_size = 64
        config.n_embed = 64
        config.n_head = 2
        config.n_kv_head = 2  # MHA: same as n_head
        config.n_layer = 1
        config.block_size = 32
        config.use_qk_norm = True

        model = GPT(config, master_process=False, pad_vocab_size_to=64).to(device)
        model.eval()
        return model

    def test_single_token_generation(self, tiny_model, device):
        """Test generation with single token at a time."""
        model = tiny_model
        config = model.config

        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.block_size,
            head_dim=config.n_embed // config.n_head,
            num_layers=config.n_layer,
        )

        # Process single tokens sequentially
        for i in range(5):
            token = torch.tensor([[i]], device=device)
            with torch.no_grad():
                logits, _ = model(token, kv_cache=kv_cache)
            assert logits.shape == (1, 1, config.vocab_size)
            assert kv_cache.pos == i + 1

    def test_cache_reuse_after_reset(self, tiny_model, device):
        """Test that cache can be reused after reset."""
        model = tiny_model
        config = model.config

        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=config.block_size,
            head_dim=config.n_embed // config.n_head,
            num_layers=config.n_layer,
        )

        # First generation
        tokens1 = torch.randint(0, 64, (1, 10), device=device)
        with torch.no_grad():
            model(tokens1, kv_cache=kv_cache)
        assert kv_cache.pos == 10

        # Reset and reuse
        kv_cache.reset()
        assert kv_cache.pos == 0

        # Second generation
        tokens2 = torch.randint(0, 64, (1, 8), device=device)
        with torch.no_grad():
            model(tokens2, kv_cache=kv_cache)
        assert kv_cache.pos == 8

    def test_empty_cache_state(self):
        """Test cache in empty/uninitialized state."""
        kv_cache = KVCache(
            batch_size=1, num_heads=4, seq_len=32, head_dim=64, num_layers=2
        )

        assert kv_cache.pos == 0
        assert kv_cache.kv_cache is None
        assert kv_cache.get_pos() == 0

        # Reset on empty cache should not fail
        kv_cache.reset()
        assert kv_cache.pos == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
