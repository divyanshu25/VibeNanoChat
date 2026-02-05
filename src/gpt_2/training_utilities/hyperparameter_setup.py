"""Hyperparameter setup utilities for training."""

import torch

from gpt_2.utils import calculate_num_iterations, get_peak_flops


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
            - max_learning_rate: Maximum learning rate
            - min_learning_rate: Minimum learning rate
            - max_steps: Maximum steps per epoch
            - flops_per_token: FLOPs per token
            - peak_flops: Peak FLOPs of device
            - warmup_steps: Number of warmup steps
            - run_evals_after: Validation eval interval
            - run_core_evals_after: Core eval interval
    """
    num_epochs = config.num_epochs
    # Note: run_evals_after will be set adaptively after we know max_steps

    # Batch size and gradient accumulation
    if sft_training:
        # For SFT, we count examples (conversations) not tokens
        # No gradient accumulation - process each batch immediately
        grad_accumulation_steps = 1
        total_batch_size = config.batch_size * ddp_world_size
    else:
        # For pretrain/midtrain, we count tokens
        total_batch_size = config.total_batch_size
        grad_accumulation_steps = total_batch_size // (
            config.batch_size * config.block_size * ddp_world_size
        )

        assert (
            total_batch_size % (config.batch_size * config.block_size * ddp_world_size)
            == 0
        ), "Total batch size must be divisible by batch_size * block_size * world_size"

    if master_process:
        print(f"Total batch size: {total_batch_size}")
        print(f"Grad accumulation steps: {grad_accumulation_steps}")

    # Learning rate scheduling parameters
    max_learning_rate = config.max_learning_rate
    min_learning_rate = max_learning_rate * config.min_lr_ratio

    # Apply nanochat-style depth-aware LR scaling if using depth mode
    # LR ∝ 1/√model_dim (tuned at model_dim=768, depth=12)
    if config._depth_mode:
        # Learning rate scaling: LR ∝ 1/√model_dim
        # Reference: tuned at model_dim=768 (depth=12, aspect_ratio=64)
        reference_dim = 768
        lr_scale = (config.n_embed / reference_dim) ** -0.5
        max_learning_rate *= lr_scale
        min_learning_rate *= lr_scale

        if master_process:
            print(f"\n📐 DEPTH-AWARE SCALING (depth={config.depth})")
            print(f"   n_layer: {config.n_layer}")
            print(
                f"   n_embed: {config.n_embed} (base: {config._base_dim}, nudge: {config._nudge:+d})"
            )
            print(f"   n_head: {config.n_head}")
            print(f"   head_dim: {config.n_embed // config.n_head}")
            print(
                f"   LR scaling: {lr_scale:.6f} (∝ 1/√({config.n_embed}/{reference_dim}))"
            )
            print(
                f"   Max LR: {config.max_learning_rate:.2e} → {max_learning_rate:.2e}"
            )

    # Automatically calculate steps based on config settings for all phases
    if master_process:
        print("\n" + "=" * 80)
        if sft_training:
            print("📊 CALCULATING SFT TRAINING STEPS")
        else:
            print("📊 CALCULATING PRETRAINING STEPS")
        print("=" * 80)

    num_iterations, flops_per_token, _ = calculate_num_iterations(
        raw_model, config, master_process
    )
    max_steps = num_iterations

    if master_process:
        print("=" * 80 + "\n")

    # Initialize peak FLOPs for MFU calculation
    if device.startswith("cuda"):
        device_name = torch.cuda.get_device_name(device)
        peak_flops = get_peak_flops(device_name)
        if master_process:
            print(f"GPU: {device_name}")
            print(f"Peak FLOPS (BF16): {peak_flops:.2e}\n")
    else:
        peak_flops = float("inf")  # MFU not meaningful for non-CUDA devices

    # Set warmup steps based on training phase (calculated as % of max_steps)
    if sft_training:
        warmup_steps = int(max_steps * config.lr_warmup_ratio_sft)
    else:
        warmup_steps = int(max_steps * config.lr_warmup_ratio_pretrain)

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
        "total_batch_size": total_batch_size,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "max_steps": max_steps,
        "flops_per_token": flops_per_token,
        "peak_flops": peak_flops,
        "warmup_steps": warmup_steps,
        "run_evals_after": run_evals_after,
        "run_core_evals_after": run_core_evals_after,
    }
