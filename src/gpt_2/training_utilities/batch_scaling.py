"""
Batch size, learning rate, and weight decay scaling utilities.

This module implements the nanochat-style batch scaling approach based on:
1. Power Lines paper (Bergsma et al., arXiv:2505.13738): Bopt ∝ D^0.383
2. T_epoch framework (arXiv:2405.13698): Weight decay scaling for constant T_epoch
3. Square root learning rate scaling: η ∝ √(B/B_ref) for AdamW/Muon

These scaling laws allow us to automatically determine optimal hyperparameters
for models of different sizes while maintaining consistent training dynamics.
"""

import math

from gpt_2.utils import get_scaling_params


def compute_optimal_batch_size(
    target_tokens, reference_tokens, reference_batch_size, round_to_power_of_2=True
):
    """
     Calculate optimal batch size based on Power Lines paper (arXiv:2505.13738).

     The paper found that optimal batch size grows as Bopt ∝ D^0.383, where D is
     the number of training tokens (not parameters!). This means:
     - 10× more tokens → only 2.4× bigger batch
     - Batch size grows slowly, much slower than linear

     For nanochat's compute-optimal training where D ∝ N (via target-param-data-ratio),
     deeper models naturally want larger batches, but the growth is sublinear.

     Args:
         target_tokens (int): Training horizon in tokens for target model
         reference_tokens (int): Training horizon in tokens for reference model (e.g., d12)
         reference_batch_size (int): Known optimal batch size for reference model
         round_to_power_of_2 (bool): Round result to nearest power of 2 for efficiency

     Returns:
         int: Optimal batch size for target model

     Example:
         # d12 trains on 0.7B tokens with batch size 524K (2^19)
         # d26 trains on 9.6B tokens, what batch size?
         >>> compute_optimal_batch_size(
         ...     target_tokens=9_600_000_000,
         ...     reference_tokens=700_000_000,
         ...     reference_batch_size=2**19
         ... )
         1048576  # 2^20, approximately 1M tokens

          AUTO BATCH SIZE CALCULATION
    Target tokens: 1,799,470,400
    Reference tokens (d12): 1,235,692,800
    Token ratio: 1.46×
    Reference batch: 524,288
    → Auto-computed optimal batch size: 524,288 tokens



    """
    # Calculate the ratio of training horizons
    # This tells us how much longer the target model will train
    token_ratio = target_tokens / reference_tokens

    # Apply power law: Bopt ∝ D^0.383
    # The 0.383 exponent means batch size grows sublinearly with training length
    # This is because larger batches reduce gradient noise, but the benefit saturates
    predicted_batch_size = reference_batch_size * (token_ratio**0.383)

    # Round to nearest power of 2 for computational efficiency
    # Powers of 2 are preferred because:
    # - They make gradient accumulation math clean (always integer divisor)
    # - GPU kernels are often optimized for power-of-2 sizes
    if round_to_power_of_2:
        optimal_batch_size = 2 ** round(math.log2(predicted_batch_size))
    else:
        optimal_batch_size = int(predicted_batch_size)

    return optimal_batch_size


def compute_lr_scale_factor(total_batch_size, reference_batch_size):
    """
    Calculate learning rate scaling factor based on batch size.

    For second-order optimizers (AdamW, Muon), we use square root scaling:
    η ∝ √(B/B_ref)

    This is gentler than linear scaling because:
    - Larger batches provide more accurate gradient estimates
    - Second-order optimizers accumulate curvature information
    - The effective learning "speed" doesn't need to scale linearly with batch size

    Note: First-order optimizers like SGD use linear scaling (η ∝ B/B_ref),
    but nanochat uses AdamW + Muon, so we use sqrt scaling for both.

    Args:
        total_batch_size (int): Target batch size in tokens
        reference_batch_size (int): Reference batch size where base LR was tuned

    Returns:
        float: Multiplicative factor to scale all learning rates

    Example:
        # Tuned at 524K (2^19), now using 1M (2^20) → 2× batch size
        >>> compute_lr_scale_factor(
        ...     total_batch_size=2**20,
        ...     reference_batch_size=2**19
        ... )
        1.4142...  # sqrt(2) ≈ 1.414
    """
    # Calculate batch size ratio
    batch_ratio = total_batch_size / reference_batch_size

    # Apply square root scaling rule for second-order optimizers
    # η ∝ √(B/B_ref)
    lr_scale_factor = batch_ratio**0.5

    return lr_scale_factor


def compute_weight_decay_scale_factor(
    total_batch_size, reference_batch_size, target_tokens, reference_tokens
):
    """
    Calculate weight decay scaling factor using T_epoch framework.

    Based on "Scaling Laws for Gradient Descent Training" (arXiv:2405.13698).
    The key insight is that T_epoch = B/(η·λ·D) should remain constant across
    different model sizes to maintain consistent regularization strength.

    Since we scale learning rate as η ∝ √(B/B_ref), we can derive:
    λ = λ_ref · √(B/B_ref) · (D_ref/D)

    This ensures:
    - Larger batches → higher weight decay (compensates for fewer update steps)
    - More tokens → lower weight decay (longer training needs less regularization)
    - The product B/(η·λ·D) stays constant (preserves regularization dynamics)

    Note: This theory is studied for AdamW, but nanochat applies it to Muon too,
    assuming similar principles hold for Newton-Schulz preconditioned optimization.

    Args:
        total_batch_size (int): Target batch size in tokens
        reference_batch_size (int): Reference batch size where base WD was tuned
        target_tokens (int): Training horizon in tokens for target model
        reference_tokens (int): Training horizon in tokens for reference model

    Returns:
        float: Multiplicative factor to scale weight decay

    Example:
        # d12: 524K batch, 0.7B tokens, WD=0.2
        # d26: 1M batch, 9.6B tokens, WD=?
        >>> compute_weight_decay_scale_factor(
        ...     total_batch_size=2**20,       # 1M
        ...     reference_batch_size=2**19,   # 524K
        ...     target_tokens=9_600_000_000,  # 9.6B
        ...     reference_tokens=700_000_000  # 0.7B
        ... )
        0.1033...  # WD should be 0.2 * 0.1033 ≈ 0.021
    """
    # Calculate batch size ratio (B/B_ref)
    batch_ratio = total_batch_size / reference_batch_size

    # Calculate token horizon ratio (D_ref/D)
    token_ratio = reference_tokens / target_tokens

    # Apply T_epoch scaling formula: λ = λ_ref · √(B/B_ref) · (D_ref/D)
    # The sqrt term comes from LR scaling (η ∝ √(B/B_ref))
    # The token ratio ensures T_epoch = B/(η·λ·D) remains constant
    wd_scale_factor = (batch_ratio**0.5) * token_ratio

    return wd_scale_factor


def scale_hyperparameters(
    model,
    config,
    reference_depth=12,
    reference_batch_size=2**19,  # 524,288 tokens
    master_process=True,
):
    """
    Apply nanochat-style batch scaling to all hyperparameters.

    This function implements the complete scaling pipeline:
    1. Calculate num_iterations from target_flops or target_param_data_ratio
    2. Auto-compute optimal batch size (if not specified)
    3. Scale learning rates by √(B/B_ref)
    4. Scale weight decay by √(B/B_ref) · (D_ref/D)

    Args:
        model: GPT model with num_scaling_params() method
        config: GPTConfig with training hyperparameters
        reference_depth (int): Reference model depth for building reference model
        reference_batch_size (int): Batch size where base hyperparams were tuned
        master_process (bool): Whether to print scaling information

    Returns:
        dict: Scaled hyperparameters containing:
            - num_iterations: Number of training steps
            - flops_per_token: FLOPs per token for this model
            - total_batch_size: Possibly auto-computed optimal batch size
            - batch_lr_scale: LR multiplicative factor
            - weight_decay_scaled: Scaled weight decay value
            - scaling_info: Dict with diagnostic information
    """
    # ========== STEP 1: Calculate num_iterations and target tokens ==========

    # Get number of parameters used for scaling law calculations
    # (transformer matrices + lm_head, excludes embeddings)
    num_scaling_params = get_scaling_params(model)

    # Get FLOPs per token for this model
    flops_per_token = model.estimate_flops()

    # Calculate target_param_data_ratio from target_flops if specified
    # Priority: target_flops > target_param_data_ratio
    if config.target_flops > 0:
        # Backwards calculate the param_data_ratio from target FLOPs
        # FLOPs = flops_per_token × tokens
        # tokens = param_data_ratio × num_scaling_params
        # Therefore: param_data_ratio = FLOPs / (flops_per_token × num_scaling_params)
        target_param_data_ratio = config.target_flops / (
            flops_per_token * num_scaling_params
        )

        if master_process:
            print("\n📊 COMPUTING PARAM RATIO FROM TARGET FLOPS")
            print(f"   Target FLOPs: {config.target_flops:.2e}")
            print(f"   FLOPs per token: {flops_per_token:.2e}")
            print(f"   Scaling params: {num_scaling_params:,}")
            print(f"   → Computed param_data_ratio: {target_param_data_ratio:.2f}")
    else:
        # Use the specified param_data_ratio
        target_param_data_ratio = config.target_param_data_ratio

        if target_param_data_ratio <= 0:
            raise ValueError(
                "No training horizon specified. Set either target_flops or target_param_data_ratio"
            )

    # Calculate optimal training horizon in tokens
    # This is compute-optimal: Tokens = target_param_data_ratio × Params
    # e.g., Chinchilla used ratio=20, nanochat often uses 8-10
    target_tokens = int(target_param_data_ratio * num_scaling_params)

    # Build reference model (d12) to get its parameter count
    # We build it on 'meta' device so it's just shapes/dtypes, no actual memory
    from gpt_2.config import GPTConfig as GPTConfigClass
    from gpt_2.gpt2_model import GPT

    # Create reference model config (d=12) using same architecture settings
    # vocab_size must match target config since lm_head size depends on it
    ref_config = GPTConfigClass(
        depth=reference_depth,
        aspect_ratio=config.aspect_ratio,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
    )

    # Build reference model on meta device (no memory allocation)
    import torch

    with torch.device("meta"):
        ref_model = GPT(ref_config, master_process=False)

    # Get reference model's scaling parameters
    ref_scaling_params = get_scaling_params(ref_model)

    # Calculate reference model's training horizon
    reference_tokens = int(target_param_data_ratio * ref_scaling_params)

    # ========== STEP 2: Auto-compute optimal batch size (if requested) ==========

    total_batch_size = config.total_batch_size

    # If total_batch_size == -1, auto-compute using Power Lines scaling law
    if total_batch_size == -1:
        total_batch_size = compute_optimal_batch_size(
            target_tokens=target_tokens,
            reference_tokens=reference_tokens,
            reference_batch_size=reference_batch_size,
            round_to_power_of_2=True,
        )

        if master_process:
            print("\n📊 AUTO BATCH SIZE CALCULATION")
            print(f"   Target tokens: {target_tokens:,}")
            print(f"   Reference tokens (d{reference_depth}): {reference_tokens:,}")
            print(f"   Token ratio: {target_tokens/reference_tokens:.2f}×")
            print(f"   Reference batch: {reference_batch_size:,}")
            print(f"   → Auto-computed optimal batch size: {total_batch_size:,} tokens")

    # ========== STEP 3: Calculate learning rate scaling ==========

    batch_lr_scale = compute_lr_scale_factor(
        total_batch_size=total_batch_size, reference_batch_size=reference_batch_size
    )

    if master_process and batch_lr_scale != 1.0:
        print("\n📊 LEARNING RATE SCALING")
        batch_ratio = total_batch_size / reference_batch_size
        print(
            f"   Batch size: {total_batch_size:,} (reference: {reference_batch_size:,})"
        )
        print(f"   Batch ratio: {batch_ratio:.4f}×")
        print(f"   LR scale: {batch_lr_scale:.4f}× (sqrt rule: η ∝ √(B/B_ref))")
        print("   Scaled LRs:")
        print(
            f"      embedding_lr:    {config.embedding_lr:.6f} → {config.embedding_lr * batch_lr_scale:.6f}"
        )
        print(
            f"      unembedding_lr:  {config.unembedding_lr:.6f} → {config.unembedding_lr * batch_lr_scale:.6f}"
        )
        print(
            f"      matrix_lr:       {config.matrix_lr:.6f} → {config.matrix_lr * batch_lr_scale:.6f}"
        )
        print(
            f"      scalar_lr:       {config.scalar_lr:.6f} → {config.scalar_lr * batch_lr_scale:.6f}"
        )

    # ========== STEP 4: Calculate weight decay scaling ==========

    wd_scale_factor = compute_weight_decay_scale_factor(
        total_batch_size=total_batch_size,
        reference_batch_size=reference_batch_size,
        target_tokens=target_tokens,
        reference_tokens=reference_tokens,
    )

    weight_decay_scaled = config.weight_decay * wd_scale_factor

    if master_process and weight_decay_scaled != config.weight_decay:
        print("\n📊 WEIGHT DECAY SCALING (T_epoch framework)")
        print("   Formula: λ = λ_ref · √(B/B_ref) · (D_ref/D)")
        print(f"   Batch ratio: {total_batch_size / reference_batch_size:.4f}×")
        print(f"   Token ratio (inverse): {reference_tokens / target_tokens:.4f}×")
        print(f"   WD scale: {wd_scale_factor:.6f}×")
        print(f"   Weight decay: {config.weight_decay:.6f} → {weight_decay_scaled:.6f}")

    # ========== STEP 5: Calculate num_iterations ==========

    # We need total_batch_size to compute num_iterations
    # So this must happen after batch size is determined
    num_iterations = target_tokens // total_batch_size

    if master_process:
        print("\n📊 TRAINING ITERATIONS")
        print(f"   Target tokens: {target_tokens:,}")
        print(f"   Total batch size: {total_batch_size:,}")
        print(f"   → Number of iterations: {num_iterations:,}")

    # ========== STEP 6: Return scaled hyperparameters ==========

    return {
        "num_iterations": num_iterations,
        "flops_per_token": flops_per_token,
        "total_batch_size": total_batch_size,
        "batch_lr_scale": batch_lr_scale,
        "weight_decay_scaled": weight_decay_scaled,
        "scaling_info": {
            "target_tokens": target_tokens,
            "target_param_data_ratio": target_param_data_ratio,
            "reference_tokens": reference_tokens,
            "num_scaling_params": num_scaling_params,
            "ref_scaling_params": ref_scaling_params,
            "batch_ratio": total_batch_size / reference_batch_size,
            "token_ratio": target_tokens / reference_tokens,
            "wd_scale_factor": wd_scale_factor,
        },
    }
