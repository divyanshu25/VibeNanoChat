"""Model setup utilities for training."""

import torch

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT


def setup_model(
    depth_override,
    aspect_ratio_override,
    head_dim_override,
    target_flops_override,
    param_data_ratio_override,
    eval_interval_override,
    device,
    ddp,
    master_process,
):
    """
    Initialize GPT model and wrap with torch.compile if needed.

    Args:
        depth_override: Override for model depth (None to use default)
        aspect_ratio_override: Override for aspect ratio (None to use default)
        head_dim_override: Override for head dimension (None to use default)
        target_flops_override: Override for target FLOPs (None to use default)
        param_data_ratio_override: Override for param:data ratio (None to use default)
        eval_interval_override: Override for evaluation interval (None to use default)
        device: Device to place model on
        ddp: Whether using distributed data parallel
        master_process: Whether this is the master process

    Returns:
        tuple: (config, raw_model, model)
            - config: GPTConfig instance with all overrides applied
            - raw_model: Unwrapped GPT model
            - model: torch.compiled model (or raw_model if compile disabled)
    """
    config = GPTConfig()

    # Override architecture parameters using depth-based parameterization
    if depth_override is not None:
        if master_process:
            print(f"🔢 Using depth-based architecture: depth={depth_override}")
        config.depth = depth_override
        if aspect_ratio_override is not None:
            config.aspect_ratio = aspect_ratio_override
        if head_dim_override is not None:
            config.head_dim = head_dim_override
        # __post_init__ will auto-calculate n_layer, n_embed, n_head
        config.__post_init__()

        # Auto-scale batch size based on model size to avoid OOM
        # Small models can use larger batch sizes for faster training
        # Large models need smaller batch sizes to fit in memory
        # Reference: tuned on 2x H100 80GB
        if depth_override <= 8:
            auto_batch_size = 32  # ~100M params: high throughput
        elif depth_override <= 10:
            auto_batch_size = 32  # ~100M params: high throughput
        elif depth_override <= 14:
            auto_batch_size = 16  # ~150M-210M params: balanced
        elif depth_override <= 18:
            auto_batch_size = 8  # ~280M-360M params: conservative
        elif depth_override <= 22:
            auto_batch_size = 4  # ~560M-700M params: tight fit
        else:
            auto_batch_size = 2  # ~1B+ params: minimal batch

        if master_process:
            if auto_batch_size != config.batch_size:
                print(
                    f"   Auto-scaling batch_size: {config.batch_size} → {auto_batch_size} (for depth={depth_override})"
                )
            else:
                print(
                    f"   Batch size: {auto_batch_size} (optimal for depth={depth_override})"
                )
        config.batch_size = auto_batch_size
    else:
        # Use default config values
        if master_process:
            print(
                f"📝 Using default architecture: n_layer={config.n_layer}, n_embed={config.n_embed}"
            )

    # Override target_flops if provided
    if target_flops_override is not None:
        if master_process:
            print(
                f"🎯 Overriding target_flops: {config.target_flops:.2e} → {target_flops_override:.2e}"
            )
        config.target_flops = target_flops_override

    # Override param_data_ratio if provided
    if param_data_ratio_override is not None:
        if master_process:
            print(
                f"🎯 Overriding param_data_ratio: {config.target_param_data_ratio} → {param_data_ratio_override}"
            )
        config.target_param_data_ratio = param_data_ratio_override

    # Override eval_interval if provided
    if eval_interval_override is not None:
        if master_process:
            print(
                f"⏱️  Overriding eval_interval: {config.eval_interval} → {eval_interval_override}"
            )
        config.eval_interval = eval_interval_override

    # Create raw model and keep reference BEFORE any wrapping
    # This reference will always point to the unwrapped model with updated weights
    raw_model = GPT(config, master_process=master_process)
    raw_model.to(device)

    # Wrap with torch.compile for faster training
    # Use dynamic=False because input shapes never change during training (nanochat-style)
    # Requires PyTorch 2.9.0+ to avoid none_dealloc crash
    model = torch.compile(raw_model, dynamic=False)
    # model = raw_model

    # Nanochat-style: NO DDP wrapper when using combined optimizer with Muon
    # The DistMuonAdamW optimizer handles gradient synchronization internally
    # This provides better overlap of communication with computation
    #
    # Nanochat-style: NO DDP wrapper when using Muon optimizer
    # The DistMuonAdamW optimizer handles gradient synchronization internally
    if ddp:
        if master_process:
            print(
                "Using DistMuonAdamW optimizer - skipping DDP wrapper (nanochat-style)"
            )

    # Note: raw_model stays unchanged and shares parameters with model
    # This allows clean checkpoint saving without unwrapping

    return config, raw_model, model
