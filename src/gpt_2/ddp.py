"""
DDP (Distributed Data Parallel) Training Script for GPT-2

This module orchestrates distributed training across multiple GPUs using PyTorch's
DistributedDataParallel (DDP). It supports three training modes:
    - pretraining: Train a model from scratch on general data
    - mid-training: Continue training from a checkpoint on specialized data
    - all: Run pretraining followed by mid-training automatically

Usage Examples:
    # Single GPU pretraining
    python ddp.py --mode pretraining

    # Multi-GPU pretraining with torchrun (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 ddp.py --mode pretraining

    # Mid-training from a checkpoint
    torchrun --nproc_per_node=4 ddp.py --mode mid-training --checkpoint /path/to/checkpoint.pt

    # Full pipeline (pretrain → mid-train)
    torchrun --nproc_per_node=4 ddp.py --mode all
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path Setup: Ensure parent directory is in sys.path for local imports
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
# ---------------------------------------------------------------------------
# PyTorch Distributed Imports
# ---------------------------------------------------------------------------
from torch.distributed import destroy_process_group, init_process_group

from gpt_2.trainer import Trainer


def setup_distributed():
    """
    Initialize the distributed training environment.

    This function detects whether we're running in a DDP context (launched via torchrun)
    or single-process mode and configures the environment accordingly.

    DDP Environment Variables (set by torchrun):
        - RANK: Global rank of this process across all nodes
        - LOCAL_RANK: Rank within the current node (used for GPU assignment)
        - WORLD_SIZE: Total number of processes across all nodes

    Returns:
        tuple: (ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device)
            - ddp (bool): Whether DDP is enabled
            - ddp_rank (int): Global rank of this process
            - ddp_local_rank (int): Local rank for GPU assignment
            - ddp_world_size (int): Total number of processes
            - master_process (bool): True if this is the main process (rank 0)
            - device (str): Device string (e.g., "cuda:0", "cpu", "mps")
    """
    # Check if running in DDP mode by looking for RANK environment variable
    # torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE automatically
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        # -----------------------------------------------------------------------
        # DDP Mode: Initialize process group for multi-GPU training
        # -----------------------------------------------------------------------
        print(f"Initializing DDP at rank: {os.environ['RANK']}")
        assert torch.cuda.is_available(), "CUDA is not available"

        # Extract rank information from environment
        ddp_rank = int(os.environ["RANK"])  # Global rank across all nodes
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Rank within this node
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))  # Total processes

        # Assign this process to a specific GPU based on local rank
        # e.g., rank 0 → cuda:0, rank 1 → cuda:1, etc.
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

        # Initialize NCCL backend (optimized for NVIDIA GPU communication)
        # This sets up communication primitives like all_reduce, broadcast, etc.
        # Pass device_id to avoid warnings about device not being specified
        init_process_group(backend="nccl", device_id=torch.device(device))

        # Only rank 0 should handle logging, checkpointing, and evaluation
        master_process = ddp_rank == 0
    else:
        # -----------------------------------------------------------------------
        # Single Process Mode: No DDP, use best available device
        # -----------------------------------------------------------------------
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True  # Single process is always the master

        # Auto-detect best available device: CUDA > MPS > CPU
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
        print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Set random seeds for reproducibility
    # Each process uses the same seed to ensure consistent initialization
    # -----------------------------------------------------------------------
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def run_pretraining(
    ddp,
    ddp_rank,
    ddp_local_rank,
    ddp_world_size,
    master_process,
    device,
    run_evals,
    run_core_evals=False,
    run_chatcore_evals=False,
    checkpoint_path=None,
):
    """
    Execute the pretraining phase.

    Pretraining trains the model from scratch (or resumes from checkpoint) on a
    large general-purpose dataset (typically OpenWebText or similar). This
    establishes the model's base language understanding capabilities.

    Args:
        ddp (bool): Whether DDP is enabled
        ddp_rank (int): Global process rank
        ddp_local_rank (int): Local rank for GPU assignment
        ddp_world_size (int): Total number of processes
        master_process (bool): Whether this is the main process
        device (str): Device to train on
        run_evals (bool): Whether to run evaluations during training
        checkpoint_path (str, optional): Path to checkpoint to resume from.
            If None, starts from scratch.

    Returns:
        str: Path to the final checkpoint saved after pretraining
    """
    # Only master process prints status messages to avoid duplicate output
    if master_process:
        print("\n" + "=" * 80)
        if checkpoint_path:
            print("🔄 RESUMING PRETRAINING PHASE")
            print(f"📂 From checkpoint: {checkpoint_path}")
        else:
            print("🚀 STARTING PRETRAINING PHASE")
        print("=" * 80 + "\n")

    # Create trainer with mid_training=False for pretraining mode
    # This tells the trainer to use pretraining hyperparameters and data
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer = Trainer(
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        master_process=master_process,
        device=device,
        run_evals=run_evals,
        run_core_evals=run_core_evals,
        run_chatcore_evals=run_chatcore_evals,
        mid_training=False,  # Use pretraining configuration
        checkpoint_path=checkpoint_path,  # Resume from checkpoint if provided
        checkpoint_dir=checkpoint_dir,
        token_bytes_path="/mnt/localssd/NanoGPT/data/token_bytes.pt",
    )
    trainer.train()

    # Construct path to the final checkpoint for potential mid-training continuation
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints"
    final_checkpoint = os.path.join(
        checkpoint_dir, f"model_checkpoint_{trainer.max_steps-1}_pretraining.pt"
    )

    if master_process:
        print(f"\n✅ Pretraining complete! Final checkpoint: {final_checkpoint}\n")

    return final_checkpoint


def run_midtraining(
    ddp,
    ddp_rank,
    ddp_local_rank,
    ddp_world_size,
    master_process,
    device,
    run_evals,
    run_core_evals=False,
    run_chatcore_evals=False,
    checkpoint_path=None,
):
    """
    Execute the mid-training phase.

    Mid-training continues from a pretrained checkpoint and trains on
    specialized data (e.g., task mixtures, domain-specific data). This
    adapts the model's capabilities for specific downstream tasks while
    retaining general language understanding.

    Args:
        ddp (bool): Whether DDP is enabled
        ddp_rank (int): Global process rank
        ddp_local_rank (int): Local rank for GPU assignment
        ddp_world_size (int): Total number of processes
        master_process (bool): Whether this is the main process
        device (str): Device to train on
        run_evals (bool): Whether to run evaluations during training
        checkpoint_path (str): Path to pretrained checkpoint to resume from

    Raises:
        ValueError: If checkpoint_path is not provided
        FileNotFoundError: If the checkpoint file doesn't exist
    """
    if master_process:
        print("\n" + "=" * 80)
        print("🔄 STARTING MID-TRAINING PHASE")
        print("=" * 80 + "\n")

    # Validate checkpoint path
    if not checkpoint_path:
        raise ValueError(
            "Checkpoint path is required for mid-training. Use --checkpoint flag."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create trainer with mid_training=True to use mid-training configuration
    # This loads the checkpoint and uses mid-training hyperparameters/data
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/midtrain_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer = Trainer(
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        master_process=master_process,
        device=device,
        run_evals=run_evals,
        run_core_evals=run_core_evals,
        run_chatcore_evals=run_chatcore_evals,
        mid_training=True,  # Use mid-training configuration
        checkpoint_path=checkpoint_path,  # Resume from this checkpoint
        checkpoint_dir=checkpoint_dir,
        token_bytes_path="/mnt/localssd/NanoGPT/data/token_bytes.pt",
    )
    trainer.train()

    if master_process:
        print("\n✅ Mid-training complete!\n")


def run_trainer(args):
    """
    Main training orchestration function.

    This function handles the high-level training flow based on the selected mode:
        - pretraining: Train from scratch
        - mid-training: Continue from checkpoint on specialized data
        - all: Run pretraining, then automatically continue to mid-training

    Args:
        args: Parsed command-line arguments containing:
            - mode: Training mode (pretraining/mid-training/all)
            - checkpoint: Path to checkpoint (for mid-training)
            - run_evals: Whether to run evaluations
    """
    # Initialize distributed training environment
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = (
        setup_distributed()
    )

    try:
        # -----------------------------------------------------------------------
        # Mode: Pretraining Only
        # Train from scratch (or resume from checkpoint) on general data
        # -----------------------------------------------------------------------
        if args.mode == "pretraining":
            run_pretraining(
                ddp,
                ddp_rank,
                ddp_local_rank,
                ddp_world_size,
                master_process,
                device,
                args.run_evals,
                run_core_evals=args.run_core_evals,
                run_chatcore_evals=args.run_chatcore_evals,
                checkpoint_path=args.checkpoint,  # Optional: resume from checkpoint
            )

        # -----------------------------------------------------------------------
        # Mode: Mid-training Only
        # Continue from checkpoint on specialized data
        # -----------------------------------------------------------------------
        elif args.mode == "mid-training":
            run_midtraining(
                ddp,
                ddp_rank,
                ddp_local_rank,
                ddp_world_size,
                master_process,
                device,
                args.run_evals,
                run_core_evals=args.run_core_evals,
                run_chatcore_evals=args.run_chatcore_evals,
                checkpoint_path=args.checkpoint,
            )

        # -----------------------------------------------------------------------
        # Mode: Full Pipeline
        # Run pretraining, then automatically continue to mid-training
        # -----------------------------------------------------------------------
        elif args.mode == "all":
            if master_process:
                print("\n" + "=" * 80)
                print("🎯 RUNNING FULL PIPELINE: PRETRAINING → MID-TRAINING")
                print("=" * 80 + "\n")

            # Phase 1: Pretraining
            # Returns the path to the final checkpoint for mid-training
            checkpoint_path = run_pretraining(
                ddp,
                ddp_rank,
                ddp_local_rank,
                ddp_world_size,
                master_process,
                device,
                args.run_evals,
                run_core_evals=args.run_core_evals,
                run_chatcore_evals=args.run_chatcore_evals,
            )

            # Transition message
            if master_process:
                print("\n" + "=" * 80)
                print("🔄 TRANSITIONING TO MID-TRAINING")
                print(f"📂 Using checkpoint: {checkpoint_path}")
                print("=" * 80 + "\n")

            # Phase 2: Mid-training
            # Uses the checkpoint from pretraining
            run_midtraining(
                ddp,
                ddp_rank,
                ddp_local_rank,
                ddp_world_size,
                master_process,
                device,
                args.run_evals,
                run_core_evals=args.run_core_evals,
                run_chatcore_evals=args.run_chatcore_evals,
                checkpoint_path=checkpoint_path,
            )

            if master_process:
                print("\n" + "=" * 80)
                print("🎉 FULL PIPELINE COMPLETE!")
                print("=" * 80 + "\n")

        else:
            raise ValueError(
                f"Invalid mode: {args.mode}. Choose from: pretraining, mid-training, all"
            )

    finally:
        # -----------------------------------------------------------------------
        # Cleanup: Destroy the process group to release resources
        # This is important to avoid hanging processes on exit
        # -----------------------------------------------------------------------
        if ddp:
            destroy_process_group()


# ===========================================================================
# Entry Point
# ===========================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Argument Parser Setup
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="GPT-2 Training with DDP - Supports pretraining, mid-training, or both"
    )

    # Training mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="pretraining",
        choices=["pretraining", "mid-training", "all"],
        help="Training mode: 'pretraining' (only pretrain), 'mid-training' (only mid-train), or 'all' (pretrain then mid-train)",
    )

    # Checkpoint path for resuming or mid-training
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (required for mid-training, optional for pretraining to resume)",
    )

    # Evaluation toggle (default: enabled)
    parser.add_argument(
        "--no-evals",
        action="store_true",
        help="Disable evaluations during training (faster training)",
    )

    # CORE evaluation toggle (default: disabled)
    parser.add_argument(
        "--run-core-evals",
        action="store_true",
        help="Enable CORE benchmark evaluations during training (recommended for tracking model quality)",
    )

    # ChatCore evaluation toggle (default: disabled)
    parser.add_argument(
        "--run-chatcore-evals",
        action="store_true",
        help="Enable ChatCore evaluations during training",
    )

    # Parse and process arguments
    args = parser.parse_args()
    args.run_evals = not args.no_evals  # Convert --no-evals flag to positive boolean

    # -----------------------------------------------------------------------
    # Argument Validation
    # Mid-training mode requires a checkpoint
    # -----------------------------------------------------------------------
    if args.mode == "mid-training" and not args.checkpoint:
        parser.error("--checkpoint is required when using --mode mid-training")

    # Start training
    run_trainer(args)
