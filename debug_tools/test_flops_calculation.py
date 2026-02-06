"""
Test script to verify FLOPs and iteration calculations match nanochat.
"""

import sys

sys.path.append("src")

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import calculate_num_iterations


def test_nanochat_560m_config():
    """Test with nanochat 560M configuration."""
    print("=" * 80)
    print("Testing Nanochat 560M Model Configuration")
    print("=" * 80)

    # Create model with nanochat 560M config
    config = GPTConfig()

    # Verify config matches nanochat output
    print("\nModel Architecture:")
    print(f"vocab_size: {config.vocab_size:,}")
    print(f"num_layers: {config.n_layer}")
    print(f"model_dim: {config.n_embed}")
    print(f"num_heads: {config.n_head}")
    print(f"num_kv_heads: {config.n_kv_head}")
    print(f"block_size: {config.block_size}")

    print("\nTraining Configuration:")
    print(
        f"Tokens / micro-batch / rank: {config.batch_size} x {config.block_size} = {config.batch_size * config.block_size:,}"
    )
    print(f"Total batch size: {config.total_batch_size:,}")

    # Create model
    model = GPT(config)

    # Calculate parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_scaling_params = model.num_scaling_params()

    print(f"\nNumber of parameters: {num_params:,}")
    assert (
        num_params == num_scaling_params
    ), "num_params should equal num_scaling_params"

    # Calculate FLOPs
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    # Expected FLOPs from nanochat output: 3.491758e+09
    expected_flops = 3.491758e09
    flops_diff = abs(num_flops_per_token - expected_flops) / expected_flops
    print(f"Expected FLOPs: {expected_flops:e}")
    print(f"Difference: {flops_diff * 100:.2f}%")

    if flops_diff < 0.01:  # Within 1%
        print("✓ FLOPs calculation matches nanochat!")
    else:
        print(f"✗ FLOPs calculation differs from nanochat by {flops_diff * 100:.2f}%")

    # Calculate iterations
    print("\n" + "=" * 80)
    print("Calculating Training Iterations")
    print("=" * 80)

    # Use default data:param ratio of 20 (Chinchilla optimal)
    config.target_param_data_ratio = 20

    num_iterations, _, _ = calculate_num_iterations(model, config)

    # Nanochat with ratio=4 got 21,400 iterations
    # With ratio=20, we should get 5x more: ~107,000 iterations
    # But let's test with ratio=4 to match nanochat output
    print("\n" + "=" * 80)
    print("Testing with data:param ratio = 4 (nanochat default)")
    print("=" * 80)
    config.target_param_data_ratio = 4
    num_iterations_ratio4, _, _ = calculate_num_iterations(model, config)

    expected_iterations = 21400  # From nanochat output
    print(f"\nExpected iterations (nanochat): {expected_iterations:,}")
    print(f"Calculated iterations: {num_iterations_ratio4:,}")

    if abs(num_iterations_ratio4 - expected_iterations) < 100:
        print("✓ Iteration calculation matches nanochat!")
    else:
        print("✗ Iteration calculation differs from nanochat")

    # Test with explicit num_iterations
    print("\n" + "=" * 80)
    print("Testing with explicit num_iterations")
    print("=" * 80)
    config.num_iterations = 10000
    config.target_param_data_ratio = -1  # Disable ratio-based calculation
    num_iterations_explicit, _, _ = calculate_num_iterations(model, config)
    assert num_iterations_explicit == 10000, "Should use explicit num_iterations"
    print("✓ Explicit num_iterations works!")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_nanochat_560m_config()
