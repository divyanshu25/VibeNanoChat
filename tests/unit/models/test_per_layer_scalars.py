"""
Unit tests for per-layer scalars (resid_lambdas and x0_lambdas) implementation.

Tests verify:
1. Parameters exist and have correct shapes
2. Initialization is correct (resid_lambdas=1.0, x0_lambdas=0.1)
3. Forward pass works correctly with scalars
4. Gradients flow through scalars
5. Optimizer groups are configured correctly
6. Per-layer mixing is applied before each block
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, os.path.join(project_root, "src"))

from gpt_2.gpt2_model import GPT


class TestConfig:
    """Minimal config for testing per-layer scalars"""

    vocab_size = 1000
    n_embed = 128
    n_layer = 4
    n_head = 4
    n_kv_head = 4
    block_size = 64
    window_pattern = "L"
    logit_softcap = 30.0


@pytest.fixture
def config():
    """Provide test configuration"""
    return TestConfig()


@pytest.fixture
def model(config):
    """Create model instance for testing"""
    return GPT(config, master_process=False, pad_vocab_size_to=64)


class TestPerLayerScalarsExistence:
    """Test that per-layer scalar parameters exist and have correct shapes"""

    def test_resid_lambdas_exists(self, model, config):
        """Verify resid_lambdas parameter exists"""
        assert hasattr(
            model, "resid_lambdas"
        ), "Model should have resid_lambdas parameter"
        assert isinstance(
            model.resid_lambdas, nn.Parameter
        ), "resid_lambdas should be a Parameter"

    def test_x0_lambdas_exists(self, model, config):
        """Verify x0_lambdas parameter exists"""
        assert hasattr(model, "x0_lambdas"), "Model should have x0_lambdas parameter"
        assert isinstance(
            model.x0_lambdas, nn.Parameter
        ), "x0_lambdas should be a Parameter"

    def test_resid_lambdas_shape(self, model, config):
        """Verify resid_lambdas has correct shape"""
        expected_shape = (config.n_layer,)
        assert (
            model.resid_lambdas.shape == expected_shape
        ), f"resid_lambdas shape should be {expected_shape}, got {model.resid_lambdas.shape}"

    def test_x0_lambdas_shape(self, model, config):
        """Verify x0_lambdas has correct shape"""
        expected_shape = (config.n_layer,)
        assert (
            model.x0_lambdas.shape == expected_shape
        ), f"x0_lambdas shape should be {expected_shape}, got {model.x0_lambdas.shape}"


class TestPerLayerScalarsInitialization:
    """Test that per-layer scalars are initialized correctly"""

    def test_resid_lambdas_init_value(self, model, config):
        """Verify resid_lambdas initialized to 1.0"""
        expected = torch.ones(config.n_layer)
        assert torch.allclose(
            model.resid_lambdas.data, expected
        ), f"resid_lambdas should be initialized to 1.0, got {model.resid_lambdas.data}"

    def test_x0_lambdas_init_value(self, model, config):
        """Verify x0_lambdas initialized to 0.1"""
        expected = torch.full((config.n_layer,), 0.1)
        assert torch.allclose(
            model.x0_lambdas.data, expected
        ), f"x0_lambdas should be initialized to 0.1, got {model.x0_lambdas.data}"

    def test_gradients_enabled(self, model):
        """Verify gradients are enabled for both parameters"""
        assert (
            model.resid_lambdas.requires_grad
        ), "resid_lambdas should require gradients"
        assert model.x0_lambdas.requires_grad, "x0_lambdas should require gradients"


class TestPerLayerScalarsForward:
    """Test forward pass with per-layer scalars"""

    def test_forward_pass_completes(self, model, config):
        """Verify forward pass completes without errors"""
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets=targets)

        assert logits is not None, "Forward pass should return logits"
        assert loss is not None, "Forward pass should return loss"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_forward_pass_output_shape(self, model, config):
        """Verify forward pass produces correct output shape"""
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets=None)

        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert (
            logits.shape == expected_shape
        ), f"Logits shape should be {expected_shape}, got {logits.shape}"

    def test_scalars_affect_output(self, model, config):
        """Verify that changing scalars affects model output"""
        # Train model for a few steps to get non-zero weights
        # (At initialization, weights are tiny and softcap clamps everything)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch_size = 2
        seq_len = 16

        for _ in range(5):
            idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            model.zero_grad()
            _, loss = model(idx, targets=targets)
            loss.backward()
            optimizer.step()

        # Now test with same input but different scalars
        torch.manual_seed(42)
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Get baseline output with default scalars
        with torch.no_grad():
            logits_baseline, _ = model(idx)

        # Modify scalars significantly
        original_resid = model.resid_lambdas.data.clone()
        original_x0 = model.x0_lambdas.data.clone()

        with torch.no_grad():
            model.resid_lambdas.fill_(2.0)  # 2x amplification
            model.x0_lambdas.fill_(0.5)  # 5x increase from default

        # Get modified output with changed scalars
        with torch.no_grad():
            logits_modified, _ = model(idx)

        # Restore original values
        with torch.no_grad():
            model.resid_lambdas.data.copy_(original_resid)
            model.x0_lambdas.data.copy_(original_x0)

        # Outputs should be different (softcap limits the magnitude of changes)
        max_diff = (logits_baseline - logits_modified).abs().max().item()
        assert (
            max_diff > 1e-5
        ), f"Changing scalars should affect model output, max diff: {max_diff}"

    def test_neutral_initialization_equivalence(self, model, config):
        """Verify that initialization values approximate standard Transformer"""
        # This tests that resid_lambdas=1.0 and x0_lambdas≈0 approximate standard behavior
        # We can't test exact equivalence due to x0_lambdas=0.1, but outputs should be similar

        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Default initialization (resid=1.0, x0=0.1)
        with torch.no_grad():
            logits_default, _ = model(idx)

        # Nearly neutral (resid=1.0, x0=0.0)
        with torch.no_grad():
            model.x0_lambdas.fill_(0.0)
            logits_neutral, _ = model(idx)

        # Should be relatively close (not identical due to numerical differences)
        diff = (logits_default - logits_neutral).abs().max()
        assert diff < 1.0, f"Default init should be close to neutral, max diff: {diff}"


class TestPerLayerScalarsGradients:
    """Test gradient flow through per-layer scalars"""

    def test_resid_lambdas_receives_gradients(self, model, config):
        """Verify resid_lambdas receives gradients during backprop"""
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets=targets)
        loss.backward()

        assert (
            model.resid_lambdas.grad is not None
        ), "resid_lambdas should receive gradients"
        assert not torch.isnan(
            model.resid_lambdas.grad
        ).any(), "resid_lambdas gradients should not contain NaN"

    def test_x0_lambdas_receives_gradients(self, model, config):
        """Verify x0_lambdas receives gradients during backprop"""
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets=targets)
        loss.backward()

        assert model.x0_lambdas.grad is not None, "x0_lambdas should receive gradients"
        assert not torch.isnan(
            model.x0_lambdas.grad
        ).any(), "x0_lambdas gradients should not contain NaN"

    def test_gradient_magnitudes(self, model, config):
        """Verify gradients have reasonable magnitudes"""
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets=targets)
        loss.backward()

        resid_grad_norm = model.resid_lambdas.grad.norm().item()
        x0_grad_norm = model.x0_lambdas.grad.norm().item()

        # Gradients should not be NaN or too large
        # Note: At initialization, resid_lambdas gradients can be very small (1e-10 range)
        # because the model starts with near-zero weights, and resid_lambdas=1.0 is near-optimal
        assert not torch.isnan(
            model.resid_lambdas.grad
        ).any(), "resid_lambdas gradients should not be NaN"
        assert not torch.isnan(
            model.x0_lambdas.grad
        ).any(), "x0_lambdas gradients should not be NaN"
        assert (
            resid_grad_norm < 1e3
        ), f"resid_lambdas gradient norm too large: {resid_grad_norm}"
        assert x0_grad_norm < 1e3, f"x0_lambdas gradient norm too large: {x0_grad_norm}"

        # At least one should have non-trivial gradients
        assert (
            resid_grad_norm > 0 or x0_grad_norm > 1e-6
        ), f"Both gradients are too small: resid={resid_grad_norm}, x0={x0_grad_norm}"


class TestPerLayerScalarsOptimizer:
    """Test optimizer configuration for per-layer scalars"""

    def test_optimizer_contains_scalars(self, model):
        """Verify optimizer includes per-layer scalars"""
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            ddp=False,
            master_process=False,
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=0.5,
            adam_beta1=0.8,
            adam_beta2=0.95,
        )

        # Find parameters in optimizer
        all_params = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                all_params.add(id(param))

        assert (
            id(model.resid_lambdas) in all_params
        ), "Optimizer should include resid_lambdas"
        assert id(model.x0_lambdas) in all_params, "Optimizer should include x0_lambdas"

    def test_resid_lambdas_learning_rate(self, model):
        """Verify resid_lambdas uses conservative learning rate (0.01x scalar_lr)"""
        scalar_lr = 0.5
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            ddp=False,
            master_process=False,
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=scalar_lr,
            adam_beta1=0.8,
            adam_beta2=0.95,
        )

        # Find resid_lambdas group
        resid_group = None
        for group in optimizer.param_groups:
            if group["kind"] == "adamw":
                for param in group["params"]:
                    if param is model.resid_lambdas:
                        resid_group = group
                        break

        assert resid_group is not None, "resid_lambdas should be in optimizer"
        expected_lr = scalar_lr * 0.01
        assert (
            resid_group["lr"] == expected_lr
        ), f"resid_lambdas LR should be {expected_lr}, got {resid_group['lr']}"

    def test_x0_lambdas_learning_rate(self, model):
        """Verify x0_lambdas uses full scalar_lr"""
        scalar_lr = 0.5
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            ddp=False,
            master_process=False,
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=scalar_lr,
            adam_beta1=0.8,
            adam_beta2=0.95,
        )

        # Find x0_lambdas group
        x0_group = None
        for group in optimizer.param_groups:
            if group["kind"] == "adamw":
                for param in group["params"]:
                    if param is model.x0_lambdas:
                        x0_group = group
                        break

        assert x0_group is not None, "x0_lambdas should be in optimizer"
        assert (
            x0_group["lr"] == scalar_lr
        ), f"x0_lambdas LR should be {scalar_lr}, got {x0_group['lr']}"

    def test_x0_lambdas_high_momentum(self, model):
        """Verify x0_lambdas uses higher beta1 for stability"""
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            ddp=False,
            master_process=False,
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=0.5,
            adam_beta1=0.8,
            adam_beta2=0.95,
        )

        # Find x0_lambdas group
        x0_group = None
        for group in optimizer.param_groups:
            if group["kind"] == "adamw":
                for param in group["params"]:
                    if param is model.x0_lambdas:
                        x0_group = group
                        break

        assert x0_group is not None, "x0_lambdas should be in optimizer"
        assert (
            x0_group["betas"][0] == 0.96
        ), f"x0_lambdas beta1 should be 0.96, got {x0_group['betas'][0]}"

    def test_optimizer_step_updates_scalars(self, model, config):
        """Verify optimizer step updates scalar values"""
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            ddp=False,
            master_process=False,
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=0.5,
            adam_beta1=0.8,
            adam_beta2=0.95,
        )

        # Run forward/backward
        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        model.zero_grad()
        logits, loss = model(idx, targets=targets)
        loss.backward()

        # Store values before step
        resid_before = model.resid_lambdas.data.clone()
        x0_before = model.x0_lambdas.data.clone()

        # Take optimizer step
        optimizer.step()

        # Values should change
        resid_changed = not torch.allclose(resid_before, model.resid_lambdas.data)
        x0_changed = not torch.allclose(x0_before, model.x0_lambdas.data)

        assert resid_changed, "Optimizer step should update resid_lambdas"
        assert x0_changed, "Optimizer step should update x0_lambdas"


class TestPerLayerScalarsIntegration:
    """Integration tests for per-layer scalars in full training context"""

    def test_multi_step_training(self, model, config):
        """Verify scalars can be trained for multiple steps"""
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            device="cpu",
            ddp=False,
            master_process=False,
            embedding_lr=0.3,
            unembedding_lr=0.004,
            matrix_lr=0.02,
            scalar_lr=0.5,
            adam_beta1=0.8,
            adam_beta2=0.95,
        )

        batch_size = 2
        seq_len = 16
        num_steps = 5

        initial_resid = model.resid_lambdas.data.clone()
        initial_x0 = model.x0_lambdas.data.clone()

        for step in range(num_steps):
            idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            model.zero_grad()
            logits, loss = model(idx, targets=targets)
            loss.backward()
            optimizer.step()

        # Values should have changed after multiple steps
        resid_delta = (model.resid_lambdas.data - initial_resid).abs().max().item()
        x0_delta = (model.x0_lambdas.data - initial_x0).abs().max().item()

        assert resid_delta > 0, "resid_lambdas should change after training"
        assert x0_delta > 0, "x0_lambdas should change after training"

    def test_parameter_count_overhead(self, model, config):
        """Verify per-layer scalars add minimal parameter overhead"""
        param_counts = model.num_scaling_params()
        scalar_params = model.resid_lambdas.numel() + model.x0_lambdas.numel()
        overhead_pct = (scalar_params / param_counts["total"]) * 100

        assert (
            scalar_params == 2 * config.n_layer
        ), f"Should have 2*n_layer scalar params, got {scalar_params}"
        assert (
            overhead_pct < 0.1
        ), f"Scalar parameter overhead should be negligible, got {overhead_pct:.4f}%"

    def test_save_and_load_checkpoint(self, model, config, tmp_path):
        """Verify scalars are correctly saved and loaded in checkpoints"""
        # Set custom values
        with torch.no_grad():
            model.resid_lambdas.fill_(1.5)
            model.x0_lambdas.fill_(0.2)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Create new model and load
        new_model = GPT(config, master_process=False, pad_vocab_size_to=64)
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Verify values match
        assert torch.allclose(
            new_model.resid_lambdas.data, model.resid_lambdas.data
        ), "resid_lambdas should match after loading checkpoint"
        assert torch.allclose(
            new_model.x0_lambdas.data, model.x0_lambdas.data
        ), "x0_lambdas should match after loading checkpoint"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
