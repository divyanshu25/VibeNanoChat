"""Hyperparameter setup utilities for training."""

import torch

from gpt_2.training_utilities.batch_scaling import scale_hyperparameters
from gpt_2.utils import get_peak_flops


def setup_hyperparameters(
    config,
    raw_model,
    sft_training,
    ddp_world_size,
    device,
    eval_interval_override,
    core_eval_interval_override,
    master_process,
):
    """
    Configure training hyperparameters based on training mode.

    Args:
        config: GPTConfig instance
        raw_model: Unwrapped GPT model (for parameter counting)
        sft_training: Whether doing SFT training
        ddp_world_size: Number of distributed processes
        device: Device for computation
        eval_interval_override: Override for validation eval interval
        core_eval_interval_override: Override for core eval interval
        master_process: Whether this is the master process

    Returns:
        dict: Dictionary containing all computed hyperparameters:
            - num_epochs: Number of training epochs
            - grad_accumulation_steps: Gradient accumulation steps
            - total_batch_size: Total batch size across all devices
            - max_steps: Maximum steps per epoch (num_iterations)
            - flops_per_token: FLOPs per token
            - peak_flops: Peak FLOPs of device
            - run_evals_after: Validation eval interval
            - run_core_evals_after: Core eval interval
            - batch_lr_scale: Learning rate scaling factor (pretrain only)
            - weight_decay_scaled: Scaled weight decay (pretrain only)
            - embedding_lr: Scaled embedding LR (pretrain only)
            - unembedding_lr: Scaled unembedding LR (pretrain only)
            - matrix_lr: Scaled matrix LR (pretrain only)
            - scalar_lr: Scaled scalar LR (pretrain only)
    """
    num_epochs = config.num_epochs
    # Note: run_evals_after will be set adaptively after we know max_steps

    # Apply nanochat-style batch scaling for pretraining
    # (SFT doesn't use batch scaling)
    batch_lr_scale = 1.0
    weight_decay_scaled = config.weight_decay
    embedding_lr = config.embedding_lr
    unembedding_lr = config.unembedding_lr
    matrix_lr = config.matrix_lr
    scalar_lr = config.scalar_lr

    # Batch size and gradient accumulation
    if sft_training:
        # For SFT, we count examples (conversations) not tokens
        # No gradient accumulation - process each batch immediately
        grad_accumulation_steps = 1

        # Update config with computed batch size
        config.total_batch_size = config.batch_size * ddp_world_size

        # For SFT, calculate num_iterations from dataset size (will be set by trainer)
        # For now, we don't compute it here since SFT doesn't use batch_scaling
        num_iterations = None  # Will be set by trainer based on dataset size
        flops_per_token = raw_model.estimate_flops()
    else:
        # For pretrain: Apply nanochat-style batch scaling
        # This handles auto batch size, num_iterations, LR scaling, and weight decay scaling
        scaling_result = scale_hyperparameters(
            model=raw_model,
            config=config,
            reference_depth=12,
            reference_batch_size=2**19,  # 524,288 tokens (nanochat reference)
            master_process=master_process,
        )

        # Extract computed values from batch scaling
        num_iterations = scaling_result["num_iterations"]
        flops_per_token = scaling_result["flops_per_token"]
        config.total_batch_size = scaling_result["total_batch_size"]
        batch_lr_scale = scaling_result["batch_lr_scale"]
        weight_decay_scaled = scaling_result["weight_decay_scaled"]

        # Apply LR scaling to all parameter groups
        embedding_lr = config.embedding_lr * batch_lr_scale
        unembedding_lr = config.unembedding_lr * batch_lr_scale
        matrix_lr = config.matrix_lr * batch_lr_scale
        scalar_lr = config.scalar_lr * batch_lr_scale

        # Calculate gradient accumulation steps with final batch size
        world_tokens_per_fwdbwd = config.batch_size * config.block_size * ddp_world_size
        grad_accumulation_steps = config.total_batch_size // world_tokens_per_fwdbwd

        assert (
            config.total_batch_size % world_tokens_per_fwdbwd == 0
        ), f"Total batch size {config.total_batch_size} must be divisible by {world_tokens_per_fwdbwd}"

    if master_process:
        print(f"\n{'='*80}")
        print("HYPERPARAMETERS SUMMARY")
        print(f"{'='*80}")
        print(f"Total batch size: {config.total_batch_size:,}")
        print(f"Grad accumulation steps: {grad_accumulation_steps}")
        if not sft_training:
            if num_iterations is not None:
                print(f"Num iterations: {num_iterations:,}")
            if batch_lr_scale != 1.0:
                print(f"LR scale factor: {batch_lr_scale:.6f}")
                print(f"Weight decay (scaled): {weight_decay_scaled:.6f}")
        print(f"{'='*80}\n")

    # Set max_steps (for SFT, this will be None and set by trainer)
    max_steps = num_iterations

    # Initialize peak FLOPs for MFU calculation
    if device.startswith("cuda"):
        device_name = torch.cuda.get_device_name(device)
        peak_flops = get_peak_flops(device_name)
        if master_process:
            print(f"GPU: {device_name}")
            print(f"Peak FLOPS (BF16): {peak_flops:.2e}\n")
    else:
        peak_flops = float("inf")  # MFU not meaningful for non-CUDA devices

    # Set adaptive eval intervals based on total training steps
    # Val loss: frequent (good learning curve), Core evals: sparse (expensive benchmarks)
    total_steps = max_steps * num_epochs

    # Validation loss interval (faster, more frequent)
    if eval_interval_override is not None:
        run_evals_after = eval_interval_override
    else:
        # Adaptive: target ~8-12 val loss measurements for smooth learning curve
        adaptive_val_interval = max(100, total_steps // 10)  # min 100 steps
        adaptive_val_interval = min(
            adaptive_val_interval, 500
        )  # max 500 steps (don't spam)
        run_evals_after = adaptive_val_interval

    # Core evaluation interval (slower, less frequent)
    if core_eval_interval_override is not None:
        run_core_evals_after = core_eval_interval_override
    else:
        # Adaptive: target ~3-5 core eval measurements (expensive, 80s each)
        adaptive_core_interval = max(250, total_steps // 4)  # min 250 steps
        adaptive_core_interval = min(
            adaptive_core_interval, total_steps // 2
        )  # at least 2 evals
        run_core_evals_after = adaptive_core_interval

    return {
        "num_epochs": num_epochs,
        "grad_accumulation_steps": grad_accumulation_steps,
        "total_batch_size": config.total_batch_size,
        "max_steps": max_steps,
        "flops_per_token": flops_per_token,
        "peak_flops": peak_flops,
        "run_evals_after": run_evals_after,
        "run_core_evals_after": run_core_evals_after,
        # Batch scaling results (for pretrain)
        "batch_lr_scale": batch_lr_scale,
        "weight_decay_scaled": weight_decay_scaled,
        "embedding_lr": embedding_lr,
        "unembedding_lr": unembedding_lr,
        "matrix_lr": matrix_lr,
        "scalar_lr": scalar_lr,
    }
