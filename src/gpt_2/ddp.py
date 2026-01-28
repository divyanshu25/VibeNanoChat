"""
DDP (Distributed Data Parallel) Training Script for GPT-2
==========================================================

This module orchestrates distributed training across multiple GPUs using PyTorch's
DistributedDataParallel (DDP). It provides a flexible training pipeline with support
for nanochat-style depth parameterization and hybrid AdamW+Muon optimization.

Training Modes
--------------
    - pretraining:  Train from scratch on general data (e.g., FineWeb-Edu)
    - mid-training: Continue from checkpoint on specialized data (e.g., task mixtures)
    - sft:          Supervised fine-tuning with conversation data (multiplex dataloader)
    - all:          Full pipeline (pretrain → mid-train → sft) automatically

Architecture Parameterization
-----------------------------
Uses nanochat-style depth × aspect_ratio parameterization for scaling law experiments:
    - model_dim = depth × aspect_ratio
    - n_layer = depth
    - n_heads = model_dim // head_dim
    - Example: depth=12, aspect_ratio=64, head_dim=128 → 12 layers, 768d model, 6 heads

Optimization
------------
    - Default: Hybrid AdamW+Muon (nanochat-style)
      • Muon: Transformer weight matrices (W_qkv, W_o, W_fc1, W_fc2)
      • AdamW: Embeddings, output head, and low-dimensional parameters
    - Use --no-muon flag to switch to AdamW-only

Usage Examples
--------------
    # Single GPU pretraining (Muon enabled by default)
    python ddp.py --mode pretraining --depth 12

    # Multi-GPU pretraining with 4 GPUs
    torchrun --nproc_per_node=4 ddp.py --mode pretraining --depth 12 --target-flops 1e18

    # Mid-training from checkpoint with AdamW-only
    torchrun --nproc_per_node=4 ddp.py --mode mid-training --checkpoint path/to/ckpt.pt --no-muon

    # Full pipeline with CORE evaluations
    torchrun --nproc_per_node=4 ddp.py --mode all --depth 12 --run-core-evals

Module Structure
----------------
    1. setup_distributed()    - Initialize DDP environment
    2. run_pretraining()      - Execute pretraining phase
    3. run_midtraining()      - Execute mid-training phase
    4. run_sft()              - Execute SFT phase
    5. run_trainer()          - Main orchestration function
    6. Argument parser        - Command-line interface
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

# ===========================================================================
# Distributed Training Setup
# ===========================================================================


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


# ===========================================================================
# Training Phase Functions
# ===========================================================================


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
    depth=None,
    aspect_ratio=None,
    head_dim=None,
    target_flops=None,
    eval_interval=None,
    core_eval_interval=None,
    use_muon=True,
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
        run_core_evals (bool): Whether to run CORE benchmark evaluations
        run_chatcore_evals (bool): Whether to run ChatCORE evaluations
        checkpoint_path (str, optional): Path to checkpoint to resume from
        depth (int, optional): Model depth for nanochat-style architecture
        aspect_ratio (int, optional): Aspect ratio for depth mode
        head_dim (int, optional): Target head dimension
        target_flops (float, optional): Target total FLOPs for training
        eval_interval (int, optional): Steps between validation evaluations
        core_eval_interval (int, optional): Steps between CORE evaluations
        use_muon (bool): Use hybrid AdamW+Muon optimizer (default: True)

    Returns:
        str: Path to the final checkpoint saved after pretraining
    """
    # -----------------------------------------------------------------------
    # Status Message (Master Process Only)
    # -----------------------------------------------------------------------
    if master_process:
        print("\n" + "=" * 80)
        if checkpoint_path:
            print("🔄 RESUMING PRETRAINING PHASE")
            print(f"📂 From checkpoint: {checkpoint_path}")
        else:
            print("🚀 STARTING PRETRAINING PHASE")
        print("=" * 80 + "\n")

    # -----------------------------------------------------------------------
    # Initialize Trainer for Pretraining
    # -----------------------------------------------------------------------
    # mid_training=False tells the trainer to use:
    #   - Pretraining data (FineWeb-Edu)
    #   - Pretraining hyperparameters (LR schedule, warmup ratio, etc.)
    #   - Pretraining checkpoint interval
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
        depth=depth,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        target_flops=target_flops,
        eval_interval=eval_interval,
        core_eval_interval=core_eval_interval,
        use_muon=use_muon,
    )
    trainer.train()

    # Construct path to the final checkpoint for potential mid-training continuation
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints"
    # The actual filename format: model_checkpoint_global{global_step}_pretraining.pt
    final_global_step = (trainer.max_steps * trainer.num_epochs) - 1
    final_checkpoint = os.path.join(
        checkpoint_dir, f"model_checkpoint_global{final_global_step}_pretraining.pt"
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
    depth=None,
    aspect_ratio=None,
    head_dim=None,
    eval_interval=None,
    core_eval_interval=None,
    use_muon=True,
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
        run_core_evals (bool): Whether to run CORE benchmark evaluations
        run_chatcore_evals (bool): Whether to run ChatCORE evaluations
        checkpoint_path (str): Path to pretrained checkpoint to resume from
        depth (int, optional): Model depth for nanochat-style architecture
        aspect_ratio (int, optional): Aspect ratio for depth mode
        head_dim (int, optional): Target head dimension
        eval_interval (int, optional): Steps between validation evaluations
        core_eval_interval (int, optional): Steps between CORE evaluations
        use_muon (bool): Use hybrid AdamW+Muon optimizer (default: True)

    Returns:
        str: Path to the final checkpoint saved after mid-training

    Raises:
        ValueError: If checkpoint_path is not provided
        FileNotFoundError: If the checkpoint file doesn't exist
    """
    # -----------------------------------------------------------------------
    # Status Message (Master Process Only)
    # -----------------------------------------------------------------------
    if master_process:
        print("\n" + "=" * 80)
        print("🔄 STARTING MID-TRAINING PHASE")
        print("=" * 80 + "\n")

    # -----------------------------------------------------------------------
    # Checkpoint Validation
    # -----------------------------------------------------------------------
    # Mid-training REQUIRES a pretrained checkpoint to continue from
    if not checkpoint_path:
        raise ValueError(
            "Checkpoint path is required for mid-training. Use --checkpoint flag."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # Initialize Trainer for Mid-Training
    # -----------------------------------------------------------------------
    # mid_training=True tells the trainer to use:
    #   - Mid-training data (specialized task mixtures)
    #   - Mid-training hyperparameters (LR schedule, warmup ratio, etc.)
    #   - Mid-training checkpoint interval
    #   - Loads model weights from the provided checkpoint
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
        depth=depth,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        eval_interval=eval_interval,
        core_eval_interval=core_eval_interval,
        use_muon=use_muon,
    )
    trainer.train()

    # Construct path to the final checkpoint
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/midtrain_checkpoints"
    final_global_step = (trainer.max_steps * trainer.num_epochs) - 1
    final_checkpoint = os.path.join(
        checkpoint_dir, f"model_checkpoint_global{final_global_step}_midtraining.pt"
    )

    if master_process:
        print(f"\n✅ Mid-training complete! Final checkpoint: {final_checkpoint}\n")

    return final_checkpoint


def run_sft(
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
    depth=None,
    aspect_ratio=None,
    head_dim=None,
    eval_interval=None,
    core_eval_interval=None,
    use_muon=True,
):
    """
    Execute the SFT (Supervised Fine-Tuning) phase.

    SFT continues from a mid-training checkpoint and trains on conversation data
    using the multiplex dataloader. This adapts the model for chat and instruction
    following capabilities.

    Args:
        ddp (bool): Whether DDP is enabled
        ddp_rank (int): Global process rank
        ddp_local_rank (int): Local rank for GPU assignment
        ddp_world_size (int): Total number of processes
        master_process (bool): Whether this is the main process
        device (str): Device to train on
        run_evals (bool): Whether to run evaluations during training
        run_core_evals (bool): Whether to run CORE benchmark evaluations
        run_chatcore_evals (bool): Whether to run ChatCORE evaluations (runs after each epoch if enabled)
        checkpoint_path (str): Path to mid-training checkpoint to resume from
        depth (int, optional): Model depth for nanochat-style architecture
        aspect_ratio (int, optional): Aspect ratio for depth mode
        head_dim (int, optional): Target head dimension
        eval_interval (int, optional): Steps between validation evaluations
        core_eval_interval (int, optional): Steps between CORE evaluations
        use_muon (bool): Use hybrid AdamW+Muon optimizer (default: True)

    Returns:
        str: Path to the final checkpoint saved after SFT

    Raises:
        ValueError: If checkpoint_path is not provided
        FileNotFoundError: If the checkpoint file doesn't exist
    """
    # -----------------------------------------------------------------------
    # Status Message (Master Process Only)
    # -----------------------------------------------------------------------
    if master_process:
        print("\n" + "=" * 80)
        print("🎯 STARTING SFT PHASE")
        print("=" * 80 + "\n")

    # -----------------------------------------------------------------------
    # Checkpoint Validation
    # -----------------------------------------------------------------------
    # SFT REQUIRES a mid-training checkpoint to continue from
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required for SFT. Use --checkpoint flag.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # Initialize Trainer for SFT
    # -----------------------------------------------------------------------
    # sft_training=True tells the trainer to use:
    #   - SFT data (conversation data via multiplex dataloader)
    #   - SFT hyperparameters (LR schedule, warmup ratio, etc.)
    #   - SFT checkpoint interval
    #   - Loads model weights from the provided mid-training checkpoint
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/sft_checkpoints"
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
        sft_training=True,  # Use SFT configuration
        checkpoint_path=checkpoint_path,  # Rollover from mid-training checkpoint
        checkpoint_dir=checkpoint_dir,
        token_bytes_path="/mnt/localssd/NanoGPT/data/token_bytes.pt",
        depth=depth,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
        eval_interval=eval_interval,
        core_eval_interval=core_eval_interval,
        use_muon=use_muon,
    )
    trainer.train()

    # Construct path to the final checkpoint
    checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/sft_checkpoints"
    final_global_step = (trainer.max_steps * trainer.num_epochs) - 1
    final_checkpoint = os.path.join(
        checkpoint_dir, f"model_checkpoint_global{final_global_step}_sft.pt"
    )

    if master_process:
        print(f"\n✅ SFT complete! Final checkpoint: {final_checkpoint}\n")

    return final_checkpoint


# ===========================================================================
# Main Training Orchestration
# ===========================================================================


def run_trainer(args):
    """
    Main training orchestration function.

    This function handles the high-level training flow based on the selected mode.
    It initializes the distributed environment, validates arguments, and executes
    the appropriate training phase(s).

    Training Modes:
        - pretraining:  Train from scratch (or resume) on general data
        - mid-training: Continue from pretrained checkpoint on specialized data
        - sft:          Supervised fine-tuning from mid-training checkpoint
        - all:          Full pipeline (pretrain → mid-train → sft)

    Args:
        args: Parsed command-line arguments containing:
            - mode: Training mode (pretraining/mid-training/sft/all)
            - checkpoint: Path to checkpoint (for mid-training/sft/resume)
            - run_evals: Whether to run evaluations
            - run_core_evals: Whether to run CORE benchmark evaluations
            - run_chatcore_evals: Whether to run ChatCORE evaluations
            - depth, aspect_ratio, head_dim: Architecture parameters
            - target_flops: Target compute budget
            - use_muon: Whether to use hybrid AdamW+Muon optimizer
    """
    # -----------------------------------------------------------------------
    # Initialize Distributed Environment
    # -----------------------------------------------------------------------
    # Sets up DDP if running via torchrun, otherwise uses single-GPU/CPU mode
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = (
        setup_distributed()
    )

    try:
        # ===================================================================
        # Mode Execution: Dispatch to appropriate training phase(s)
        # ===================================================================

        # -------------------------------------------------------------------
        # Mode 1: Pretraining Only
        # -------------------------------------------------------------------
        # Train model from scratch on general-purpose data (FineWeb-Edu)
        # Optional: Resume from checkpoint if --checkpoint provided
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
                depth=args.depth,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                target_flops=args.target_flops,
                eval_interval=args.eval_interval,
                core_eval_interval=args.core_eval_interval,
                use_muon=args.use_muon,
            )

        # -------------------------------------------------------------------
        # Mode 2: Mid-Training Only
        # -------------------------------------------------------------------
        # Continue from pretrained checkpoint with specialized data
        # Requires: --checkpoint (path to pretrained model)
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
                depth=args.depth,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                eval_interval=args.eval_interval,
                core_eval_interval=args.core_eval_interval,
                use_muon=args.use_muon,
            )

        # -------------------------------------------------------------------
        # Mode 3: SFT (Supervised Fine-Tuning) Only
        # -------------------------------------------------------------------
        # Continue from mid-training checkpoint with conversation data
        # Requires: --checkpoint (path to mid-trained model)
        # Uses: Multiplex dataloader for conversation data
        elif args.mode == "sft":
            run_sft(
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
                depth=args.depth,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                eval_interval=args.eval_interval,
                core_eval_interval=args.core_eval_interval,
                use_muon=args.use_muon,
            )

        # -------------------------------------------------------------------
        # Mode 4: Full Pipeline (Pretrain → Mid-train → SFT)
        # -------------------------------------------------------------------
        # Execute complete training pipeline from scratch to chat model
        # Each phase outputs a checkpoint used by the next phase
        # This is useful for end-to-end experiments
        elif args.mode == "all":
            if master_process:
                print("\n" + "=" * 80)
                print("🎯 RUNNING FULL PIPELINE: PRETRAINING → MID-TRAINING → SFT")
                print("=" * 80 + "\n")

            # ---------------------------------------------------------------
            # Phase 1: Pretraining
            # ---------------------------------------------------------------
            # Train from scratch on general data (FineWeb-Edu)
            # Returns checkpoint path for next phase
            checkpoint_path = run_pretraining(
                ddp,
                ddp_rank,
                ddp_local_rank,
                ddp_world_size,
                master_process,
                device,
                args.run_evals,
                run_core_evals=args.run_core_evals,
                run_chatcore_evals=False,  # Don't run chatcore in pretraining
                depth=args.depth,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                target_flops=args.target_flops,
                eval_interval=args.eval_interval,
                core_eval_interval=args.core_eval_interval,
                use_muon=args.use_muon,
            )

            # ---------------------------------------------------------------
            # Transition: Pretrain → Mid-train
            # ---------------------------------------------------------------
            if master_process:
                print("\n" + "=" * 80)
                print("🔄 TRANSITIONING TO MID-TRAINING")
                print(f"📂 Using checkpoint: {checkpoint_path}")
                print("=" * 80 + "\n")

            # ---------------------------------------------------------------
            # Phase 2: Mid-Training
            # ---------------------------------------------------------------
            # Continue from pretrained checkpoint with specialized data
            # Returns checkpoint path for SFT phase
            checkpoint_path = run_midtraining(
                ddp,
                ddp_rank,
                ddp_local_rank,
                ddp_world_size,
                master_process,
                device,
                args.run_evals,
                run_core_evals=args.run_core_evals,
                run_chatcore_evals=False,  # Don't run chatcore in midtraining for full pipeline
                checkpoint_path=checkpoint_path,
                depth=args.depth,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                eval_interval=args.eval_interval,
                core_eval_interval=args.core_eval_interval,
                use_muon=args.use_muon,
            )

            # ---------------------------------------------------------------
            # Transition: Mid-train → SFT
            # ---------------------------------------------------------------
            if master_process:
                print("\n" + "=" * 80)
                print("🔄 TRANSITIONING TO SFT")
                print(f"📂 Using checkpoint: {checkpoint_path}")
                print("=" * 80 + "\n")

            # ---------------------------------------------------------------
            # Phase 3: SFT (Supervised Fine-Tuning)
            # ---------------------------------------------------------------
            # Continue from mid-trained checkpoint with conversation data
            # Final phase produces chat-capable model
            run_sft(
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
                depth=args.depth,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                eval_interval=args.eval_interval,
                core_eval_interval=args.core_eval_interval,
                use_muon=args.use_muon,
            )

            if master_process:
                print("\n" + "=" * 80)
                print("🎉 FULL PIPELINE COMPLETE!")
                print("=" * 80 + "\n")

        else:
            raise ValueError(
                f"Invalid mode: {args.mode}. Choose from: pretraining, mid-training, sft, all"
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
    # ========================================================================
    # Command-Line Argument Parser
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="GPT-2 Training with DDP - Supports pretraining, mid-training, SFT, or full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pretraining with 2 GPUs (Muon enabled by default)
  torchrun --nproc_per_node=2 ddp.py --mode pretraining --depth 12

  # Mid-training from checkpoint with AdamW-only
  torchrun --nproc_per_node=2 ddp.py --mode mid-training --checkpoint path/to/ckpt.pt --no-muon

  # Full pipeline with CORE evaluations
  torchrun --nproc_per_node=2 ddp.py --mode all --depth 12 --run-core-evals
        """,
    )

    # ========================================================================
    # Training Mode & Checkpoint
    # ========================================================================
    parser.add_argument(
        "--mode",
        type=str,
        default="pretraining",
        choices=["pretraining", "mid-training", "sft", "all"],
        help="Training mode: 'pretraining' (train from scratch), 'mid-training' (specialized data), "
        "'sft' (supervised fine-tuning), 'all' (full pipeline)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (required for mid-training/SFT, optional for pretraining)",
    )

    # ========================================================================
    # Model Architecture (Nanochat-style Depth Parameterization)
    # ========================================================================
    # Use depth-based parameterization for scaling law experiments
    # model_dim = depth × aspect_ratio, n_heads = model_dim // head_dim
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Model depth (n_layer); auto-calculates n_embed, n_head from depth × aspect_ratio",
    )

    parser.add_argument(
        "--aspect-ratio",
        type=int,
        default=64,
        help="Aspect ratio for depth mode: model_dim = depth × aspect_ratio (default: 64)",
    )

    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Attention head dimension (default: 128 for Flash Attention efficiency)",
    )

    # ========================================================================
    # Training Horizon (Compute Budget)
    # ========================================================================
    parser.add_argument(
        "--target-flops",
        type=float,
        default=None,
        help="Target total FLOPs for training (overrides config.target_flops); useful for scaling laws",
    )

    # ========================================================================
    # Evaluation Configuration
    # ========================================================================
    parser.add_argument(
        "--no-evals",
        action="store_true",
        help="Disable validation evaluations during training (faster, but no loss tracking)",
    )

    parser.add_argument(
        "--run-core-evals",
        action="store_true",
        help="Enable CORE benchmark evaluations (multiple-choice tasks: MMLU, HellaSwag, etc.)",
    )

    parser.add_argument(
        "--run-chatcore-evals",
        action="store_true",
        help="Enable ChatCORE evaluations (generative tasks: GSM8K, HumanEval, etc.)",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Steps between validation loss evaluations (default: adaptive, ~10 evals per run)",
    )

    parser.add_argument(
        "--core-eval-interval",
        type=int,
        default=None,
        help="Steps between CORE benchmark evaluations (default: adaptive, ~4 evals per run)",
    )

    # ========================================================================
    # Optimizer Configuration
    # ========================================================================
    # Muon optimizer is enabled by default (nanochat-style hybrid approach)
    # Use --no-muon to disable and use AdamW-only optimization
    # Note: Muon LR is configured in config.py (default: 0.02)
    parser.add_argument(
        "--no-muon",
        dest="use_muon",
        action="store_false",
        default=True,
        help="Disable Muon optimizer and use AdamW-only (default: Muon enabled)",
    )

    # Parse and process arguments
    args = parser.parse_args()
    args.run_evals = not args.no_evals  # Convert --no-evals flag to positive boolean

    # -----------------------------------------------------------------------
    # Argument Validation
    # Mid-training and SFT modes require a checkpoint
    # -----------------------------------------------------------------------
    if args.mode == "mid-training" and not args.checkpoint:
        parser.error("--checkpoint is required when using --mode mid-training")

    if args.mode == "sft" and not args.checkpoint:
        parser.error("--checkpoint is required when using --mode sft")

    # Start training
    run_trainer(args)
