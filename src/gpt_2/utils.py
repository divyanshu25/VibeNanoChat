"""
Shared utilities for NanoGPT project.

This module provides common functionality used across the project,
including the custom tokenizer with special tokens for chat format,
learning rate scheduling, and checkpoint management.
"""

import math
import os

import tiktoken
import torch


def get_special_tokens():
    """
    Define special tokens for chat format and tool calling.

    Token IDs start at 50257 (right after GPT-2's vocab which ends at 50256).

    Returns:
        dict: Mapping of special token strings to their token IDs
    """
    return {
        "<|bos|>": 50257,  # Beginning of sequence - marks start of conversation
        "<|user_start|>": 50258,  # Marks start of user message
        "<|user_end|>": 50259,  # Marks end of user message
        "<|assistant_start|>": 50260,  # Marks start of assistant response
        "<|assistant_end|>": 50261,  # Marks end of assistant response
        # Tool calling tokens (for GSM8K calculator calls)
        "<|python|>": 50262,  # Marks start of Python/calculator expression
        "<|python_end|>": 50263,  # Marks end of Python/calculator expression
        "<|output_start|>": 50264,  # Marks start of calculator output
        "<|output_end|>": 50265,  # Marks end of calculator output
    }


def get_custom_tokenizer():
    """
    Create a custom tiktoken encoder with our special tokens registered.

    This extends the GPT-2 tokenizer by adding our chat-format and tool-calling special tokens.
    The custom encoder can then handle these tokens natively via encode/decode.

    Returns:
        tuple: (custom_encoder, special_tokens_dict)
    """
    # Get the base GPT-2 encoding
    base_enc = tiktoken.get_encoding("gpt2")

    # Define our custom special tokens
    special_tokens = get_special_tokens()

    # Create a new encoding that includes both:
    # 1. GPT-2's existing special tokens (like <|endoftext|>)
    # 2. Our new chat-format special tokens
    enc = tiktoken.Encoding(
        name="nano_chat",  # Custom name for our extended tokenizer
        pat_str=base_enc._pat_str,  # Use same regex pattern for tokenization
        mergeable_ranks=base_enc._mergeable_ranks,  # Use same BPE merges
        special_tokens={
            **base_enc._special_tokens,  # Keep GPT-2's <|endoftext|> (id=50256)
            **special_tokens,  # Add our chat format + tool calling special tokens
        },
    )

    return enc, special_tokens


def calculate_num_iterations(
    model,
    config,
    master_process: bool = True,
) -> tuple[int, float, int]:
    """
    Calculate the number of training iterations based on nanochat-style logic.

    Priority order (first available is used):
    1. config.num_iterations (if > 0): explicit number of steps
    2. config.target_flops (if > 0): calculate from target FLOPs
    3. config.target_param_data_ratio (if > 0): calculate from data:param ratio

    Args:
        model: The GPT model (needs estimate_flops() and num_scaling_params() methods)
        config: Configuration object with training parameters
        master_process: If True, print logging information (default: True)

    Returns:
        tuple: (num_iterations, num_flops_per_token, num_scaling_params)

    Raises:
        ValueError: If no training horizon is specified
    """
    num_flops_per_token = model.estimate_flops()
    num_scaling_params = model.num_scaling_params()

    # Priority 1: Explicit num_iterations
    if config.num_iterations > 0:
        num_iterations = config.num_iterations
        if master_process:
            print(f"Using user-provided number of iterations: {num_iterations:,}")

    # Priority 2: Target FLOPs
    elif config.target_flops > 0:
        num_iterations = round(
            config.target_flops / (num_flops_per_token * config.total_batch_size)
        )
        if master_process:
            print(
                f"Calculated number of iterations from target FLOPs: {num_iterations:,}"
            )

    # Priority 3: Data:param ratio (default for nanochat)
    elif config.target_param_data_ratio > 0:
        target_tokens = config.target_param_data_ratio * num_scaling_params
        num_iterations = target_tokens // config.total_batch_size
        if master_process:
            print(
                f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
            )

    else:
        raise ValueError(
            "No training horizon specified. Set one of: num_iterations, target_flops, or target_param_data_ratio"
        )

    return num_iterations, num_flops_per_token, num_scaling_params


def get_lr_multiplier(
    global_step: int,
    max_steps: int,
    num_epochs: int,
    training_phase: str = "pretrain",
    warmup_ratio: float = 0.0,
    warmdown_ratio: float = 0.4,
    final_lr_frac: float = 0.0,
) -> float:
    """
    Compute learning rate multiplier for separate learning rates (nanochat-style).

    This returns a multiplier that gets applied to each parameter group's base
    learning rate, preserving the relative ratios between groups while following
    a schedule appropriate for the training phase.

    Training Phase Schedules (matching nanochat exactly):
    - pretrain: warmup -> constant -> warmdown to final_lr_frac
    - midtrain: constant for 80% of training -> linear decay to 0
    - sft/rl: linear decay from 1.0 to 0 (no warmup)

    Args:
        global_step: Current global training step (across all epochs)
        max_steps: Maximum steps per epoch
        num_epochs: Total number of epochs
        training_phase: One of "pretrain", "midtrain", "sft", "rl"
        warmup_ratio: Ratio of iterations for LR warmup (pretrain only, default 0.0)
        warmdown_ratio: Ratio of iterations for LR warmdown (pretrain only, default 0.4)
        final_lr_frac: Final LR as fraction of initial LR (pretrain only, default 0.0)

    Returns:
        float: Learning rate multiplier for current step
    """
    total_steps = max_steps * num_epochs

    if training_phase == "pretrain":
        # Nanochat base_train logic: warmup -> constant -> warmdown
        warmup_iters = round(warmup_ratio * total_steps)
        warmdown_iters = round(warmdown_ratio * total_steps)

        if global_step < warmup_iters:
            # Linear warmup
            return (global_step + 1) / warmup_iters
        elif global_step <= total_steps - warmdown_iters:
            # Constant LR
            return 1.0
        else:
            # Linear warmdown to final_lr_frac
            progress = (total_steps - global_step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    elif training_phase == "midtrain":
        # Nanochat mid_train logic: constant for 80%, then linear decay to 0
        progress = global_step / total_steps
        return 1.0 if progress < 0.8 else 1.0 - (progress - 0.8) / 0.2

    elif training_phase in ["sft", "rl"]:
        # Nanochat chat_sft/chat_rl logic: linear decay from 1.0 to 0
        return 1.0 - global_step / total_steps

    else:
        raise ValueError(
            f"Unknown training_phase: {training_phase}. Must be one of: pretrain, midtrain, sft, rl"
        )


def get_lr(
    global_step: int,
    warmup_steps: int,
    max_steps: int,
    num_epochs: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    """
    Implement learning rate scheduling with warmup and cosine annealing.

    - Warmup: Linear increase from 0 to max_lr over warmup_steps
    - Cosine annealing: Smooth decay from max_lr to min_lr using cosine function
    - Constant: min_lr after total_steps

    Args:
        global_step: Current global training step (across all epochs)
        warmup_steps: Number of steps for linear warmup
        max_steps: Maximum steps per epoch
        num_epochs: Total number of epochs
        max_learning_rate: Peak learning rate
        min_learning_rate: Minimum learning rate after decay

    Returns:
        float: Learning rate for current step
    """
    total_steps = max_steps * num_epochs

    if global_step < warmup_steps:
        # Linear warmup: gradually increase learning rate
        lr = max_learning_rate * (global_step + 1) / warmup_steps
    elif global_step > total_steps:
        # After total steps, use minimum learning rate
        lr = min_learning_rate
    else:
        # Cosine annealing: smooth decay using cosine function
        decay_ratio = (global_step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
    return lr


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer = None,
    master_process: bool = True,
) -> dict:
    """
    Load model weights and optionally optimizer state from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        device: Device to map the checkpoint to
        optimizer: Optional optimizer to load state into (for resuming training)
        master_process: Whether this is the master process (for logging)

    Returns:
        dict: Dictionary with 'start_step', 'start_epoch', 'start_global_step', 'config', 'val_loss'
    """
    if master_process:
        print(f"\n{'='*80}")
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*80}\n")

    # weights_only=False required for loading custom classes like GPTConfig
    # (PyTorch 2.6+ defaults to weights_only=True for security)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check for vocabulary size mismatch
    checkpoint_state = checkpoint["model"]
    checkpoint_vocab_size = checkpoint_state["transformer.wte.weight"].shape[0]
    model_vocab_size = model.transformer.wte.weight.shape[0]

    if checkpoint_vocab_size != model_vocab_size:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ VOCABULARY SIZE MISMATCH ERROR\n"
            f"{'='*80}\n"
            f"Checkpoint vocab size: {checkpoint_vocab_size}\n"
            f"Model vocab size:      {model_vocab_size}\n"
            f"Difference:            {abs(model_vocab_size - checkpoint_vocab_size)} tokens\n"
            f"\n"
            f"The checkpoint was saved with a different vocabulary size than the\n"
            f"current model configuration. This will cause index out of bounds errors.\n"
            f"\n"
            f"Possible solutions:\n"
            f"1. Use a checkpoint with matching vocab_size={model_vocab_size}\n"
            f"2. Modify your model config to use vocab_size={checkpoint_vocab_size}\n"
            f"3. Retrain from scratch with the correct vocabulary size\n"
            f"{'='*80}\n"
        )
        raise ValueError(error_msg)

    # Load model state
    model.load_state_dict(checkpoint_state)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if master_process:
            print("✅ Optimizer state loaded")

    # Extract training state
    result = {
        "config": checkpoint.get("config"),
        "start_epoch": checkpoint.get("epoch", 0),
        "start_step": 0,
        "start_global_step": 0,
        "val_loss": checkpoint.get("val_loss"),
    }

    # Load step if available (for resuming)
    if "step" in checkpoint:
        result["start_step"] = checkpoint["step"] + 1

    # Load global_step if available
    if "global_step" in checkpoint:
        result["start_global_step"] = checkpoint["global_step"] + 1

    if master_process:
        print(
            f"✅ Resuming from epoch {result['start_epoch']}, step {result['start_step']}"
        )
        print(f"   Global step: {result['start_global_step']}")
        if result["val_loss"] is not None:
            print(f"   Checkpoint val_loss: {result['val_loss']:.4f}")
        print(f"{'='*80}\n")

    return result


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    global_step: int,
    val_loss: float,
    checkpoint_dir: str,
    ddp: bool = False,
    checkpoint_interval: int = 200,
    max_steps: int = 17234,
    num_epochs: int = 3,
    master_process: bool = True,
    mid_training: bool = False,
    sft_training: bool = False,
) -> None:
    """
    Save model checkpoint at specified intervals.

    Args:
        model: The model to save (can be DDP wrapped)
        optimizer: The optimizer to save state from
        step: Current step within epoch
        epoch: Current epoch
        global_step: Global step across all epochs
        val_loss: Validation loss for this checkpoint
        checkpoint_dir: Directory to save checkpoints
        ddp: Whether using distributed data parallel
        checkpoint_interval: Save checkpoint every N steps
        max_steps: Maximum steps per epoch
        num_epochs: Total number of epochs
        master_process: Whether this is the master process
        mid_training: Whether this is mid-training mode
        sft_training: Whether this is SFT training mode
    """
    total_steps = max_steps * num_epochs
    should_save = (global_step > 0 and global_step % checkpoint_interval == 0) or (
        global_step == total_steps - 1
    )

    if not (master_process and should_save):
        return

    # Get the underlying model (unwrap DDP if needed)
    model_to_save = model.module if ddp else model
    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_to_save.config,
        "step": step,
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    if sft_training:
        checkpoint_suffix = "_sft"
    elif mid_training:
        checkpoint_suffix = "_midtraining"
    else:
        checkpoint_suffix = "_pretraining"

    checkpoint_path = (
        f"{checkpoint_dir}/model_checkpoint_global{global_step}{checkpoint_suffix}.pt"
    )

    torch.save(checkpoint, checkpoint_path)
    print(f"\n💾 Checkpoint saved: {checkpoint_path}\n")


def accumulate_bpb(per_token_loss, targets, token_bytes):
    """
    Accumulate total nats and bytes for BPB calculation.

    Handles:
    - Special tokens (token_bytes == 0): excluded from calculation
    - Ignored targets (negative indices): excluded from calculation

    Args:
        per_token_loss: Per-token loss tensor, shape (B*T,)
        targets: Target token IDs, shape (B, T) or (B*T,)
        token_bytes: Tensor mapping token ID to byte length, shape (vocab_size,)

    Returns:
        tuple: (nats_sum, bytes_sum) to add to running totals
    """
    y = targets.view(-1)

    if (y.int() < 0).any():
        # Handle ignored targets (e.g. -1)
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        num_bytes = torch.where(
            valid,
            token_bytes[y_safe],
            torch.zeros_like(y, dtype=token_bytes.dtype),
        )
    else:
        # Fast path: no ignored targets
        num_bytes = token_bytes[y]

    # Exclude special tokens (num_bytes == 0) from loss sum
    nats_sum = (per_token_loss * (num_bytes > 0)).sum()
    bytes_sum = num_bytes.sum()

    return nats_sum, bytes_sum


def get_peak_flops(device_name: str) -> float:
    """
    Get theoretical peak FLOPs for bfloat16 on various GPU architectures.

    Returns the theoretical peak FLOPS (floating point operations per second)
    for bfloat16 computation on the specified device. Used to calculate MFU
    (Model FLOPs Utilization).

    Inspired by:
    - torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
    - nanochat: https://github.com/karpathy/nanochat

    Args:
        device_name: Name of the GPU device (e.g., "NVIDIA H100")

    Returns:
        float: Peak FLOPS for bfloat16 operations. Returns inf if unknown device
               (so MFU shows as 0% rather than an incorrect value)
    """
    name = device_name.lower()

    # --- NVIDIA Blackwell ---
    if "gb200" in name or "grace blackwell" in name:
        return 2.5e15
    if "b200" in name:
        return 2.25e15
    if "b100" in name:
        return 1.8e15

    # --- NVIDIA Hopper (H100/H200/H800) ---
    if "h200" in name:
        if "nvl" in name or "pcie" in name:
            return 836e12
        return 989e12  # H200 SXM
    if "h100" in name:
        if "nvl" in name:
            return 835e12
        if "pcie" in name:
            return 756e12
        return 989e12  # H100 SXM
    if "h800" in name:
        if "nvl" in name:
            return 989e12
        return 756e12  # H800 PCIe

    # --- NVIDIA Ampere data center ---
    if "a100" in name or "a800" in name:
        return 312e12
    if "a40" in name:
        return 149.7e12
    if "a30" in name:
        return 165e12

    # --- NVIDIA Ada data center ---
    if "l40s" in name or "l40-s" in name or "l40 s" in name:
        return 362e12
    if "l4" in name:
        return 121e12

    # --- AMD CDNA accelerators ---
    if "mi355" in name:
        return 2.5e15
    if "mi325" in name or "mi300x" in name:
        return 1.3074e15
    if "mi300a" in name:
        return 980.6e12
    if "mi250x" in name:
        return 383e12
    if "mi250" in name:
        return 362.1e12

    # --- Consumer RTX (for hobbyists) ---
    if "5090" in name:
        return 209.5e12
    if "4090" in name:
        return 165.2e12
    if "3090" in name:
        return 71e12

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    print(
        f"WARNING: Peak FLOPS undefined for device: {device_name}, MFU will show as 0%"
    )
    return float("inf")
