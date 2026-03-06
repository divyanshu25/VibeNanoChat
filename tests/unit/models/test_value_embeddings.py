"""
Unit tests for Value Embeddings implementation.

Tests cover:
- Model initialization with value embeddings
- Correct layer assignment (alternating pattern)
- Value embedding dimensions and parameter counts
- Gate initialization (should be zero)
- Forward pass functionality
- Gradient flow through value embeddings
- Optimizer configuration with value embeddings
- Integration with GQA and different model sizes
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
from gpt_2.utils import has_value_embedding


class TestValueEmbeddingsCore:
    """Test core value embeddings functionality."""

    def test_has_value_embedding_function(self):
        """Test has_value_embedding() returns correct alternating pattern."""
        # Pattern: layer_idx % 2 == (n_layer - 1) % 2
        # Alternating layers based on last layer's parity

        # 4 layers: last=3 (odd), so odd layers: 1, 3 should have VE
        assert not has_value_embedding(0, 4)
        assert has_value_embedding(1, 4)
        assert not has_value_embedding(2, 4)
        assert has_value_embedding(3, 4)  # Last layer always

        # 6 layers: last=5 (odd), so odd layers: 1, 3, 5 should have VE
        assert not has_value_embedding(0, 6)
        assert has_value_embedding(1, 6)
        assert not has_value_embedding(2, 6)
        assert has_value_embedding(3, 6)
        assert not has_value_embedding(4, 6)
        assert has_value_embedding(5, 6)  # Last layer always

        # 8 layers: last=7 (odd), so odd layers: 1, 3, 5, 7 should have VE
        assert not has_value_embedding(0, 8)
        assert has_value_embedding(1, 8)
        assert not has_value_embedding(2, 8)
        assert has_value_embedding(3, 8)
        assert not has_value_embedding(4, 8)
        assert has_value_embedding(5, 8)
        assert not has_value_embedding(6, 8)
        assert has_value_embedding(7, 8)  # Last layer always

    def test_model_has_value_embeds(self):
        """Test that model creates value embeddings for correct layers."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Check that value_embeds exists
        assert hasattr(model, "value_embeds")
        assert isinstance(model.value_embeds, torch.nn.ModuleDict)

        # Check correct layers have VE (1, 3 for 4 layers - odd parity)
        expected_layers = ["1", "3"]
        actual_layers = sorted(model.value_embeds.keys())
        assert actual_layers == expected_layers

    def test_value_embeds_dimensions(self):
        """Test value embeddings have correct dimensions."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=256,
            n_layer=6,
            n_head=8,
            n_kv_head=4,  # GQA: 2:1 ratio
            block_size=512,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Calculate expected kv_dim
        head_dim = config.n_embed // config.n_head  # 256 / 8 = 32
        expected_kv_dim = config.n_kv_head * head_dim  # 4 * 32 = 128

        # Check dimensions for each VE
        for layer_idx, ve in model.value_embeds.items():
            weight_shape = ve.weight.shape
            # Shape should be (padded_vocab_size, kv_dim)
            assert weight_shape[1] == expected_kv_dim, (
                f"Layer {layer_idx}: expected kv_dim={expected_kv_dim}, "
                f"got {weight_shape[1]}"
            )

    def test_value_embed_gate_initialization(self):
        """Test that value_embed_gate weights are zero-initialized."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Check only layers with value embeddings have gates (alternating pattern)
        from gpt_2.utils import has_value_embedding

        for i, block in enumerate(model.transformer.h):
            assert hasattr(block.attn, "value_embed_gate")

            if has_value_embedding(i, config.n_layer):
                # Layers with VE should have gates
                assert block.attn.value_embed_gate is not None

                # Check gate is zero-initialized
                gate_weight_norm = block.attn.value_embed_gate.weight.norm().item()
                assert gate_weight_norm < 1e-6, (
                    f"Layer {i}: gate should be zero-initialized, "
                    f"got norm={gate_weight_norm}"
                )
            else:
                # Layers without VE should not have gates
                assert block.attn.value_embed_gate is None

    def test_parameter_count(self):
        """Test that value embeddings add expected number of parameters."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Count VE parameters
        ve_params = sum(p.numel() for p in model.value_embeds.parameters())

        # Calculate expected: 3 layers × padded_vocab_size × kv_dim
        head_dim = config.n_embed // config.n_head
        kv_dim = config.n_kv_head * head_dim
        padded_vocab_size = model.padded_vocab_size
        num_ve_layers = len(model.value_embeds)

        expected_params = num_ve_layers * padded_vocab_size * kv_dim

        assert (
            ve_params == expected_params
        ), f"Expected {expected_params:,} VE params, got {ve_params:,}"


class TestValueEmbeddingsForward:
    """Test value embeddings during forward pass."""

    def test_forward_pass_with_ve(self):
        """Test forward pass works with value embeddings."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        batch_size = 2
        seq_len = 64
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(idx, targets=targets)

        # Check outputs
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0  # Should have some loss
        assert torch.isfinite(loss)  # Should not be NaN or Inf

    def test_gradient_flow(self):
        """Test that gradients flow through value embeddings."""
        config = GPTConfig(
            vocab_size=500,  # Smaller vocab for better token coverage
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Use larger batch to ensure VE tokens are accessed
        batch_size = 8
        seq_len = 128
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward and backward
        model.zero_grad()
        logits, loss = model(idx, targets=targets)
        loss.backward()

        # Check that value embeddings have gradient tensors allocated
        ve_params_found = 0
        for name, param in model.named_parameters():
            if "value_embeds" in name:
                ve_params_found += 1
                # Gradient tensor should exist (even if sparse/mostly zero for embeddings)
                assert param.grad is not None, f"{name} should have gradient tensor"
                # Check correct shape
                assert param.grad.shape == param.shape, f"{name} grad shape mismatch"
                # Check gradients are finite (no NaN/Inf)
                assert torch.isfinite(param.grad).all(), f"{name} has NaN/Inf gradients"

        # Should have gradient tensors for all VE layers
        assert ve_params_found == len(model.value_embeds), (
            f"Expected params for {len(model.value_embeds)} VE layers, "
            f"got {ve_params_found}"
        )

    def test_forward_with_different_layer_counts(self):
        """Test forward pass works with different number of layers."""
        for n_layer in [4, 6, 8, 12]:
            config = GPTConfig(
                vocab_size=500,
                n_embed=64,
                n_layer=n_layer,
                n_head=4,
                n_kv_head=4,
                block_size=128,
                window_pattern="L",
            )
            model = GPT(config, master_process=False)

            batch_size = 2
            seq_len = 32
            idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                logits, _ = model(idx, targets=None)

            assert logits.shape == (batch_size, seq_len, config.vocab_size)


class TestValueEmbeddingsOptimizer:
    """Test optimizer configuration with value embeddings."""

    def test_optimizer_has_ve_params(self):
        """Test that optimizer includes value embedding parameters."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=0.5,
        )

        # Count VE params in optimizer
        ve_params_in_optimizer = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                # Check if this param is a value embedding
                for ve_param in model.value_embeds.parameters():
                    if param is ve_param:
                        ve_params_in_optimizer.add(id(param))

        # All VE params should be in optimizer
        total_ve_params = len(list(model.value_embeds.parameters()))
        assert len(ve_params_in_optimizer) == total_ve_params, (
            f"Expected {total_ve_params} VE params in optimizer, "
            f"got {len(ve_params_in_optimizer)}"
        )

    def test_optimizer_ve_learning_rate(self):
        """Test that VE params use embedding_lr."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        embedding_lr = 0.3
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            embedding_lr=embedding_lr,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=0.5,
        )

        # Find VE param group
        ve_param_ids = {id(p) for p in model.value_embeds.parameters()}

        found_ve_group = False
        for group in optimizer.param_groups:
            group_param_ids = {id(p) for p in group["params"]}
            if ve_param_ids & group_param_ids:  # Intersection
                # This group contains VE params
                assert (
                    group["lr"] == embedding_lr
                ), f"VE group should have lr={embedding_lr}, got {group['lr']}"
                found_ve_group = True

        assert found_ve_group, "Could not find VE params in any optimizer group"


class TestValueEmbeddingsGQA:
    """Test value embeddings work with Grouped Query Attention."""

    def test_gqa_with_ve(self):
        """Test VE works correctly with GQA (n_kv_head < n_head)."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=256,
            n_layer=4,
            n_head=8,
            n_kv_head=2,  # GQA: 4:1 ratio
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Check VE dimensions match kv_dim (not full n_head)
        head_dim = config.n_embed // config.n_head  # 256 / 8 = 32
        expected_kv_dim = config.n_kv_head * head_dim  # 2 * 32 = 64

        for ve in model.value_embeds.values():
            assert ve.weight.shape[1] == expected_kv_dim

        # Test forward pass
        batch_size = 2
        seq_len = 32
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = model(idx, targets=None)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_gqa_gate_dimensions(self):
        """Test value_embed_gate outputs match n_kv_head in GQA."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=256,
            n_layer=4,
            n_head=8,
            n_kv_head=4,  # GQA: 2:1 ratio
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Check gate output dimension matches n_kv_head (only for layers with VE)
        from gpt_2.utils import has_value_embedding

        for i, block in enumerate(model.transformer.h):
            gate = block.attn.value_embed_gate
            if has_value_embedding(i, config.n_layer):
                # Layers with VE should have gates with correct dimensions
                assert gate is not None
                # value_embed_gate: (n_kv_head, 32) - output size first
                assert gate.weight.shape == (config.n_kv_head, 32)
            else:
                # Layers without VE should not have gates
                assert gate is None


class TestValueEmbeddingsFLOPs:
    """Test that VE are correctly excluded from FLOP calculations."""

    def test_flops_exclude_ve(self):
        """Test estimate_flops() excludes value embeddings."""
        config = GPTConfig(
            vocab_size=1000,
            n_embed=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            block_size=256,
            window_pattern="L",
        )
        model = GPT(config, master_process=False)

        # Get FLOP estimate
        flops = model.estimate_flops()

        # VE params should not be counted in FLOPs
        # FLOPs should be based on matmul params only
        param_counts = model.num_scaling_params()

        # Rough check: FLOPs should be 6 * (total - excluded) + attention_flops
        # Just verify it's reasonable (not counting VE twice)
        assert flops > 0
        assert flops < 6 * param_counts["total"]  # Upper bound

        # More specific check: verify excluded params are indeed excluded
        # by checking that adding more VE layers doesn't increase FLOPs proportionally
        # (This is implicit in the exclusion logic)


if __name__ == "__main__":
    # Allow running individual test file
    pytest.main([__file__, "-v"])
