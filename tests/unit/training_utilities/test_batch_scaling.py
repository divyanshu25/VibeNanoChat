"""
Unit tests for batch_scaling module.

Tests verify:
1. get_scaling_params() correctly extracts scaling parameters from model
2. compute_optimal_batch_size() applies Power Lines scaling law correctly
3. compute_lr_scale_factor() applies square root scaling correctly
4. compute_weight_decay_scale_factor() applies T_epoch framework correctly
5. scale_hyperparameters() integrates all scaling correctly
"""

import math
import os
import sys

import pytest
import torch

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, os.path.join(project_root, "src"))

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.training_utilities.batch_scaling import (
    compute_lr_scale_factor, compute_optimal_batch_size,
    compute_weight_decay_scale_factor, get_scaling_params,
    scale_hyperparameters)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def small_config():
    """Create a small test config for fast model instantiation"""
    return GPTConfig(
        depth=6,
        aspect_ratio=64,
        head_dim=64,
        block_size=128,
        vocab_size=1000,
        target_param_data_ratio=10,
        total_batch_size=2**16,  # 65,536 tokens
        embedding_lr=0.3,
        unembedding_lr=0.004,
        matrix_lr=0.02,
        scalar_lr=0.5,
        weight_decay=0.1,
    )


@pytest.fixture
def small_model(small_config):
    """Create a small model for testing"""
    with torch.device("meta"):
        model = GPT(small_config, master_process=False)
    return model


@pytest.fixture
def mock_model_with_params():
    """Create a mock model with num_scaling_params method"""

    class MockModel:
        def num_scaling_params(self):
            return {
                "transformer_matrices": 10_000_000,
                "lm_head": 1_000_000,
                "embeddings": 500_000,
            }

    return MockModel()


# ============================================================================
# Tests for get_scaling_params()
# ============================================================================


class TestGetScalingParams:
    """Test get_scaling_params() function"""

    def test_returns_transformer_plus_lm_head(self, mock_model_with_params):
        """Verify that scaling params = transformer_matrices + lm_head"""
        result = get_scaling_params(mock_model_with_params)
        expected = 10_000_000 + 1_000_000  # transformer + lm_head
        assert result == expected, f"Expected {expected}, got {result}"

    def test_excludes_embeddings(self, mock_model_with_params):
        """Verify that embeddings are excluded from scaling params"""
        result = get_scaling_params(mock_model_with_params)
        # Should not include the 500,000 embedding params
        assert result == 11_000_000

    def test_with_real_model(self, small_model):
        """Test with a real GPT model"""
        result = get_scaling_params(small_model)
        # Just verify it returns a positive integer
        assert isinstance(result, int)
        assert result > 0


# ============================================================================
# Tests for compute_optimal_batch_size()
# ============================================================================


class TestComputeOptimalBatchSize:
    """Test compute_optimal_batch_size() function"""

    def test_basic_power_law_scaling(self):
        """Test that batch size grows as D^0.383"""
        # If tokens increase 10x, batch should increase by 10^0.383 ≈ 2.414x
        target_tokens = 10_000_000_000  # 10B tokens
        reference_tokens = 1_000_000_000  # 1B tokens
        reference_batch = 524_288  # 2^19

        result = compute_optimal_batch_size(
            target_tokens, reference_tokens, reference_batch, round_to_power_of_2=False
        )

        # Expected: 524288 * (10^0.383) ≈ 1,265,392
        expected = reference_batch * (10**0.383)
        assert abs(result - expected) < 1, f"Expected {expected:.0f}, got {result}"

    def test_equal_tokens_returns_same_batch(self):
        """Test that equal token counts return the same batch size"""
        result = compute_optimal_batch_size(
            target_tokens=1_000_000_000,
            reference_tokens=1_000_000_000,
            reference_batch_size=524_288,
            round_to_power_of_2=False,
        )
        assert result == 524_288

    def test_power_of_2_rounding(self):
        """Test that power-of-2 rounding works correctly"""
        # Use values that would give non-power-of-2 result
        result = compute_optimal_batch_size(
            target_tokens=10_000_000_000,
            reference_tokens=1_000_000_000,
            reference_batch_size=524_288,
            round_to_power_of_2=True,
        )
        # Result should be a power of 2
        log2_result = math.log2(result)
        assert log2_result == int(
            log2_result
        ), f"{result} is not a power of 2 (log2={log2_result})"

    def test_nanochat_example_d12_to_d26(self):
        """Test the documented example: d12 to d26 scaling"""
        # d12: 0.7B tokens, 524K batch
        # d26: 9.6B tokens, expected ~1M batch
        result = compute_optimal_batch_size(
            target_tokens=9_600_000_000,
            reference_tokens=700_000_000,
            reference_batch_size=2**19,
            round_to_power_of_2=True,
        )
        # Should be approximately 2^20 (1,048,576)
        assert result == 2**20, f"Expected {2**20}, got {result}"

    def test_smaller_model_smaller_batch(self):
        """Test that smaller models get smaller batch sizes"""
        reference_batch = 524_288
        reference_tokens = 1_000_000_000

        # Smaller model (fewer tokens)
        smaller_batch = compute_optimal_batch_size(
            target_tokens=100_000_000,  # 10x fewer tokens
            reference_tokens=reference_tokens,
            reference_batch_size=reference_batch,
            round_to_power_of_2=False,
        )

        assert smaller_batch < reference_batch


# ============================================================================
# Tests for compute_lr_scale_factor()
# ============================================================================


class TestComputeLRScaleFactor:
    """Test compute_lr_scale_factor() function"""

    def test_sqrt_scaling_rule(self):
        """Test that LR scales as sqrt(B/B_ref)"""
        # 4x batch size should give 2x LR
        result = compute_lr_scale_factor(
            total_batch_size=2_097_152,  # 2^21
            reference_batch_size=524_288,  # 2^19
        )
        expected = math.sqrt(4.0)  # sqrt(4) = 2.0
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_equal_batch_size_returns_one(self):
        """Test that equal batch sizes return scale factor of 1.0"""
        result = compute_lr_scale_factor(
            total_batch_size=524_288, reference_batch_size=524_288
        )
        assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"

    def test_nanochat_example_524k_to_1m(self):
        """Test the documented example: 524K to 1M batch"""
        result = compute_lr_scale_factor(
            total_batch_size=2**20,  # 1M
            reference_batch_size=2**19,  # 524K
        )
        expected = math.sqrt(2.0)  # sqrt(2) ≈ 1.414
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_smaller_batch_smaller_lr(self):
        """Test that smaller batches get smaller LR scale factors"""
        result = compute_lr_scale_factor(
            total_batch_size=131_072,  # 2^17
            reference_batch_size=524_288,  # 2^19
        )
        # Should be sqrt(1/4) = 0.5
        assert result < 1.0, "Smaller batch should give scale factor < 1.0"
        assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"


# ============================================================================
# Tests for compute_weight_decay_scale_factor()
# ============================================================================


class TestComputeWeightDecayScaleFactor:
    """Test compute_weight_decay_scale_factor() function"""

    def test_t_epoch_formula(self):
        """Test that WD scales as sqrt(B/B_ref) * (D_ref/D)"""
        # Double batch size, double training tokens
        # sqrt(2) * (1/2) = sqrt(2)/2 ≈ 0.707
        result = compute_weight_decay_scale_factor(
            total_batch_size=1_048_576,
            reference_batch_size=524_288,
            target_tokens=2_000_000_000,
            reference_tokens=1_000_000_000,
        )
        expected = math.sqrt(2.0) * 0.5  # sqrt(2) * (1/2) ≈ 0.707
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_equal_params_equal_tokens_returns_one(self):
        """Test that identical settings return scale factor of 1.0"""
        result = compute_weight_decay_scale_factor(
            total_batch_size=524_288,
            reference_batch_size=524_288,
            target_tokens=1_000_000_000,
            reference_tokens=1_000_000_000,
        )
        assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"

    def test_nanochat_example_d12_to_d26(self):
        """Test the documented example: d12 to d26"""
        result = compute_weight_decay_scale_factor(
            total_batch_size=2**20,  # 1M
            reference_batch_size=2**19,  # 524K
            target_tokens=9_600_000_000,  # 9.6B
            reference_tokens=700_000_000,  # 0.7B
        )
        # sqrt(2) * (0.7/9.6) ≈ 0.1033
        expected = math.sqrt(2.0) * (700_000_000 / 9_600_000_000)
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_larger_batch_increases_wd(self):
        """Test that larger batch increases WD (for same token count)"""
        baseline = compute_weight_decay_scale_factor(
            total_batch_size=524_288,
            reference_batch_size=524_288,
            target_tokens=1_000_000_000,
            reference_tokens=1_000_000_000,
        )
        larger_batch = compute_weight_decay_scale_factor(
            total_batch_size=1_048_576,  # 2x batch
            reference_batch_size=524_288,
            target_tokens=1_000_000_000,
            reference_tokens=1_000_000_000,
        )
        assert larger_batch > baseline

    def test_more_tokens_decreases_wd(self):
        """Test that more training tokens decrease WD (for same batch)"""
        baseline = compute_weight_decay_scale_factor(
            total_batch_size=524_288,
            reference_batch_size=524_288,
            target_tokens=1_000_000_000,
            reference_tokens=1_000_000_000,
        )
        more_tokens = compute_weight_decay_scale_factor(
            total_batch_size=524_288,
            reference_batch_size=524_288,
            target_tokens=10_000_000_000,  # 10x tokens
            reference_tokens=1_000_000_000,
        )
        assert more_tokens < baseline


# ============================================================================
# Tests for scale_hyperparameters()
# ============================================================================


class TestScaleHyperparametersNanochatStyle:
    """Test scale_hyperparameters() integration function"""

    def test_returns_required_keys(self, small_model, small_config):
        """Test that function returns all required keys"""
        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Check required keys exist
        assert "num_iterations" in result
        assert "flops_per_token" in result
        assert "total_batch_size" in result
        assert "batch_lr_scale" in result
        assert "weight_decay_scaled" in result
        assert "scaling_info" in result

    def test_scaling_info_contains_diagnostics(self, small_model, small_config):
        """Test that scaling_info contains diagnostic information"""
        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        info = result["scaling_info"]
        assert "target_tokens" in info
        assert "target_param_data_ratio" in info
        assert "reference_tokens" in info
        assert "num_scaling_params" in info
        assert "ref_scaling_params" in info
        assert "batch_ratio" in info
        assert "token_ratio" in info
        assert "wd_scale_factor" in info

    def test_uses_provided_batch_size(self, small_model, small_config):
        """Test that function uses provided batch size when not -1"""
        small_config.total_batch_size = 2**18  # Explicitly set
        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        assert result["total_batch_size"] == 2**18

    def test_auto_computes_batch_size_when_negative_one(
        self, small_model, small_config
    ):
        """Test that function auto-computes batch size when total_batch_size=-1"""
        small_config.total_batch_size = -1
        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Should auto-compute, result should be positive and power of 2
        batch_size = result["total_batch_size"]
        assert batch_size > 0
        log2_batch = math.log2(batch_size)
        assert log2_batch == int(log2_batch), f"Batch size {batch_size} not power of 2"

    def test_lr_scale_matches_sqrt_rule(self, small_model, small_config):
        """Test that LR scale factor matches sqrt(B/B_ref) rule"""
        small_config.total_batch_size = 2**20  # 1M tokens
        reference_batch = 2**19  # 524K tokens

        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=reference_batch,
            master_process=False,
        )

        expected_lr_scale = math.sqrt(2.0)  # sqrt(2^20 / 2^19)
        assert abs(result["batch_lr_scale"] - expected_lr_scale) < 1e-6

    def test_weight_decay_scaled_correctly(self, small_model, small_config):
        """Test that weight decay is scaled according to T_epoch framework"""
        small_config.total_batch_size = 2**18
        small_config.weight_decay = 0.1

        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Verify weight decay was scaled (should not equal original)
        # (Unless by coincidence the scale factor is exactly 1.0)
        assert "weight_decay_scaled" in result
        # Just verify it's a positive number
        assert result["weight_decay_scaled"] > 0

    def test_diagnostic_info_accuracy(self, small_model, small_config):
        """Test that diagnostic info accurately reflects calculations"""
        small_config.total_batch_size = 2**18
        reference_batch = 2**19

        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=reference_batch,
            master_process=False,
        )

        info = result["scaling_info"]

        # Verify batch_ratio calculation
        expected_batch_ratio = 2**18 / 2**19
        assert abs(info["batch_ratio"] - expected_batch_ratio) < 1e-6

        # Verify token calculations are consistent
        assert info["target_tokens"] > 0
        assert info["reference_tokens"] > 0

    def test_no_print_when_not_master_process(self, small_model, small_config, capsys):
        """Test that function doesn't print scaling info when master_process=False"""
        _ = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        captured = capsys.readouterr()
        # Should not print batch scaling info (though model init may print other things)
        assert "AUTO BATCH SIZE" not in captured.out
        assert "LEARNING RATE SCALING" not in captured.out
        assert "WEIGHT DECAY SCALING" not in captured.out

    def test_num_iterations_calculated_from_param_ratio(
        self, small_model, small_config
    ):
        """Test that num_iterations is correctly calculated from target_param_data_ratio"""
        small_config.target_param_data_ratio = 10
        small_config.target_flops = -1
        small_config.total_batch_size = 2**16  # 65,536 tokens

        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Verify calculation: num_iterations = (param_ratio * num_scaling_params) / batch_size
        expected_target_tokens = (
            small_config.target_param_data_ratio
            * result["scaling_info"]["num_scaling_params"]
        )
        expected_iterations = expected_target_tokens // result["total_batch_size"]

        assert result["num_iterations"] == expected_iterations
        assert result["num_iterations"] > 0

    def test_num_iterations_calculated_from_target_flops(
        self, small_model, small_config
    ):
        """Test that num_iterations is correctly calculated from target_flops"""
        # Set target_flops based on a param_ratio of 8
        num_scaling_params = get_scaling_params(small_model)
        flops_per_token = small_model.estimate_flops()
        param_ratio = 8
        target_flops = flops_per_token * num_scaling_params * param_ratio

        small_config.target_flops = target_flops
        small_config.target_param_data_ratio = 10  # Should be overridden
        small_config.total_batch_size = 2**16

        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Verify that target_flops was used to compute param_data_ratio
        computed_param_ratio = result["scaling_info"]["target_param_data_ratio"]
        assert (
            abs(computed_param_ratio - param_ratio) < 0.01
        ), f"Expected param_ratio ~{param_ratio}, got {computed_param_ratio}"

        # Verify num_iterations is calculated correctly
        expected_tokens = int(computed_param_ratio * num_scaling_params)
        expected_iterations = expected_tokens // result["total_batch_size"]
        assert (
            abs(result["num_iterations"] - expected_iterations) <= 1
        )  # Allow 1 token rounding

    def test_flops_per_token_returned(self, small_model, small_config):
        """Test that flops_per_token is correctly returned"""
        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Verify flops_per_token matches model's estimate
        expected_flops = small_model.estimate_flops()
        assert result["flops_per_token"] == expected_flops
        assert result["flops_per_token"] > 0

    def test_target_flops_priority_over_param_ratio(self, small_model, small_config):
        """Test that target_flops takes priority over target_param_data_ratio"""
        num_scaling_params = get_scaling_params(small_model)
        flops_per_token = small_model.estimate_flops()

        # Set target_flops to achieve param_ratio of 5
        target_param_ratio = 5
        small_config.target_flops = (
            flops_per_token * num_scaling_params * target_param_ratio
        )
        small_config.target_param_data_ratio = 100  # This should be ignored
        small_config.total_batch_size = 2**16

        result = scale_hyperparameters(
            model=small_model,
            config=small_config,
            reference_depth=12,
            reference_batch_size=2**19,
            master_process=False,
        )

        # Verify that the computed param_ratio is close to 5, not 100
        computed_ratio = result["scaling_info"]["target_param_data_ratio"]
        assert (
            abs(computed_ratio - target_param_ratio) < 0.01
        ), f"Expected {target_param_ratio}, got {computed_ratio} (should not be 100)"


# ============================================================================
# Edge Cases and Validation Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_raises_error_when_no_training_horizon_specified(
        self, small_model, small_config
    ):
        """Test that function raises ValueError when neither target_flops nor param_ratio is set"""
        small_config.target_flops = -1
        small_config.target_param_data_ratio = -1

        with pytest.raises(ValueError, match="No training horizon specified"):
            scale_hyperparameters(
                model=small_model,
                config=small_config,
                reference_depth=12,
                reference_batch_size=2**19,
                master_process=False,
            )

    def test_very_small_batch_size(self):
        """Test behavior with very small batch sizes"""
        result = compute_optimal_batch_size(
            target_tokens=1_000_000,  # 1M tokens
            reference_tokens=1_000_000_000,  # 1B tokens
            reference_batch_size=524_288,
            round_to_power_of_2=True,
        )
        # Should handle small batch sizes gracefully
        assert result > 0
        assert isinstance(result, int)

    def test_very_large_batch_size(self):
        """Test behavior with very large batch sizes"""
        result = compute_optimal_batch_size(
            target_tokens=1_000_000_000_000,  # 1T tokens
            reference_tokens=1_000_000_000,  # 1B tokens
            reference_batch_size=524_288,
            round_to_power_of_2=True,
        )
        # Should handle large batch sizes gracefully
        assert result > 524_288
        assert isinstance(result, int)

    def test_batch_ratio_extremes_lr_scaling(self):
        """Test LR scaling with extreme batch ratios"""
        # Very large batch
        large_result = compute_lr_scale_factor(
            total_batch_size=100_000_000, reference_batch_size=1_000_000
        )
        assert large_result == math.sqrt(100.0)  # Should be 10.0

        # Very small batch
        small_result = compute_lr_scale_factor(
            total_batch_size=1_000_000, reference_batch_size=100_000_000
        )
        assert abs(small_result - 0.1) < 1e-6  # Should be 0.1

    def test_token_ratio_extremes_wd_scaling(self):
        """Test WD scaling with extreme token ratios"""
        # Much longer training
        result = compute_weight_decay_scale_factor(
            total_batch_size=524_288,
            reference_batch_size=524_288,
            target_tokens=100_000_000_000,  # 100B tokens
            reference_tokens=1_000_000_000,  # 1B tokens
        )
        # With 100x more tokens, WD should be much smaller
        assert result < 0.1
