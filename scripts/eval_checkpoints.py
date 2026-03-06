#!/usr/bin/env python3
"""
Simple script to load final checkpoints for each model depth and run core evaluations.
Supports distributed evaluation across multiple GPUs using torchrun.

Usage:
    # Single GPU
    python debug_tools/eval_checkpoints.py

    # Multi-GPU (4 GPUs)
    torchrun --standalone --nproc_per_node=4 debug_tools/eval_checkpoints.py
"""

import os
import sys

# Add gpt_2 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

import glob
import re

# Optional imports for plotting
try:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Warning: matplotlib/numpy not available. Plotting will be disabled.")
    print("   Install with: pip install matplotlib numpy")

import torch
import torch.distributed as dist
from torch.distributed import init_process_group

from eval_tasks import CoreEvaluator
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint

# Constants
CHECKPOINT_BASE_DIR = "<YOURPATH>/nanogpt/pretrain_checkpoints"
EVAL_BUNDLE_PATH = "/mnt/localssd/VibeNanoChat/resources/eval_bundle"
LOGS_DIR = "/mnt/localssd/VibeNanoChat/logs"


def setup_distributed():
    """
    Initialize the distributed training environment.

    This function detects whether we're running in a DDP context (launched via torchrun)
    or single-process mode and configures the environment accordingly.

    Returns:
        tuple: (ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device)
    """
    # Check if running in DDP mode by looking for RANK environment variable
    # torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE automatically
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        # DDP Mode: Initialize process group for multi-GPU evaluation
        assert torch.cuda.is_available(), "CUDA is not available"

        # Extract rank information from environment
        ddp_rank = int(os.environ["RANK"])  # Global rank across all nodes
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Rank within this node
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))  # Total processes

        # Only master process prints initialization message
        if ddp_rank == 0:
            print(f"Initializing DDP at rank: {ddp_rank}")

        # Assign this process to a specific GPU based on local rank
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

        # Initialize NCCL backend for GPU communication
        init_process_group(backend="nccl", device_id=torch.device(device))

        # Only rank 0 should handle logging and checkpointing
        master_process = ddp_rank == 0
    else:
        # Single Process Mode: No DDP, use best available device
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # Auto-detect best available device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(42)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def load_model_from_checkpoint(checkpoint_path, device, master_process):
    """Load model from checkpoint file."""
    if master_process:
        print(f"\n{'='*80}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n")

    # Load checkpoint to inspect config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", None)

    if config is None:
        raise ValueError(f"No config found in checkpoint: {checkpoint_path}")

    # Create model with checkpoint's config
    model = GPT(config, master_process=master_process)
    model.to(device)
    model.cast_embeddings_to_bfloat16()

    # Load checkpoint weights
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        optimizer=None,
        master_process=master_process,
        print_resume_info=False,
    )

    model.eval()

    if master_process:
        print("✅ Model loaded successfully")
        print(f"   Depth: {config.depth if hasattr(config, 'depth') else 'N/A'}")
        print(f"   n_layer: {config.n_layer}")
        print(f"   n_embed: {config.n_embed}")
        print(f"   n_head: {config.n_head}")

    return model, config


def find_final_checkpoint(depth_dir):
    """Find the final (highest step) checkpoint in a depth directory."""
    checkpoints = glob.glob(os.path.join(depth_dir, "step*_pretrain.pt"))

    if not checkpoints:
        print(f"⚠️  No checkpoints found in {depth_dir}")
        return None

    # Extract step numbers and find the highest
    def get_step(path):
        basename = os.path.basename(path)
        # Extract step number from filename like "step3431_d14_pretrain.pt"
        try:
            step_str = basename.split("_")[0].replace("step", "")
            return int(step_str)
        except (ValueError, IndexError):
            return 0

    final_checkpoint = max(checkpoints, key=get_step)
    return final_checkpoint


def extract_flops_from_log(depth):
    """Extract total training FLOPs from log file for a given depth."""
    log_file = os.path.join(LOGS_DIR, f"scaling_laws_N{depth}_R10.log")

    if not os.path.exists(log_file):
        print(f"⚠️  Log file not found: {log_file}")
        return None

    try:
        with open(log_file, "r") as f:
            content = f.read()
            # Search for "Total training FLOPs: X.XXXe+XX"
            match = re.search(r"Total training FLOPs:\s+([\d.]+e[+-]\d+)", content)
            if match:
                flops = float(match.group(1))
                return flops
            else:
                print(f"⚠️  Could not find FLOPs in {log_file}")
                return None
    except Exception as e:
        print(f"⚠️  Error reading {log_file}: {e}")
        return None


def run_core_evaluations(
    model, config, depth, device, ddp, ddp_rank, ddp_world_size, master_process
):
    """Run core evaluations on the model."""
    if master_process:
        print(f"\n{'='*80}")
        print(f"Running CORE evaluations for depth {depth}")
        print(f"{'='*80}\n")

    # Get tokenizer
    enc, _ = get_custom_tokenizer()

    # Create core evaluator with DDP support
    core_evaluator = CoreEvaluator(
        model=model,
        tokenizer=enc,
        device=device,
        master_process=master_process,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        eval_bundle_path=EVAL_BUNDLE_PATH,
        max_examples_per_task=3000,
    )

    # Run all CORE tasks (all processes participate, results are collated)
    results = core_evaluator.evaluate_all_tasks(
        step=0,
        global_step=0,
        flops_so_far=0,
    )

    if master_process:
        print(f"\n{'='*80}")
        print(f"CORE Evaluation Results - Depth {depth}")
        print(f"{'='*80}\n")

    return results


def main():
    """Main function to load checkpoints and run evaluations."""
    # Setup distributed environment
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = (
        setup_distributed()
    )

    if master_process:
        print(f"\n{'='*80}")
        print("Checkpoint Evaluation Script")
        print(f"{'='*80}\n")
        print(f"Device: {device}")
        print(f"DDP enabled: {ddp}")
        print(f"World size: {ddp_world_size}")
        print(f"Checkpoint directory: {CHECKPOINT_BASE_DIR}")
        print(f"Eval bundle: {EVAL_BUNDLE_PATH}\n")

    # Synchronize all processes before starting
    if ddp:
        dist.barrier()

    # Find all depth directories (all processes need to know this)
    depth_dirs = sorted(glob.glob(os.path.join(CHECKPOINT_BASE_DIR, "d*")))

    if not depth_dirs:
        if master_process:
            print(f"❌ No depth directories found in {CHECKPOINT_BASE_DIR}")
        return

    if master_process:
        print(f"Found {len(depth_dirs)} depth directories:")
        for depth_dir in depth_dirs:
            print(f"  - {os.path.basename(depth_dir)}")
        print()

    all_results = {}

    # Process each depth directory
    for depth_dir in depth_dirs:
        depth_name = os.path.basename(depth_dir)
        depth = int(depth_name.replace("d", ""))

        if master_process:
            print(f"\n{'='*80}")
            print(f"Processing {depth_name}")
            print(f"{'='*80}\n")

        # Find final checkpoint
        checkpoint_path = find_final_checkpoint(depth_dir)

        if checkpoint_path is None:
            if master_process:
                print(f"⚠️  Skipping {depth_name} - no checkpoints found\n")
            continue

        if master_process:
            print(f"Final checkpoint: {os.path.basename(checkpoint_path)}")

        # Synchronize before loading model
        if ddp:
            dist.barrier()

        try:
            # Load model (all processes load the same model)
            model, config = load_model_from_checkpoint(
                checkpoint_path, device, master_process
            )

            # Synchronize after loading model
            if ddp:
                dist.barrier()

            # Run evaluations (all processes participate, results are collated)
            # Returns: (raw_results, centered_core_score, centered_results)
            results, core_score, centered_results = run_core_evaluations(
                model,
                config,
                depth,
                device,
                ddp,
                ddp_rank,
                ddp_world_size,
                master_process,
            )

            # Store results (only master needs this for summary)
            if master_process:
                # Extract FLOPS from log file
                flops = extract_flops_from_log(depth)
                all_results[depth] = {
                    "results": results,
                    "core_score": core_score,  # Use the centered core score
                    "centered_results": centered_results,
                    "flops": flops,
                }

            # Clean up
            del model
            torch.cuda.empty_cache()

            # Synchronize after cleanup
            if ddp:
                dist.barrier()

        except Exception as e:
            if master_process:
                print(f"❌ Error processing {depth_name}: {e}")
                import traceback

                traceback.print_exc()
            continue

    # Print summary (only master)
    if master_process:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL RESULTS")
        print(f"{'='*80}\n")

        # Prepare data for table and plot
        table_data = []
        depths_list = []
        flops_list = []
        core_scores_list = []

        for depth in sorted(all_results.keys()):
            results_data = all_results[depth]
            # Use the centered core score (matches nanochat methodology)
            core_score = results_data.get("core_score", None)
            flops = results_data["flops"]

            table_data.append(
                {"depth": depth, "flops": flops, "core_score": core_score}
            )

            if flops is not None and core_score is not None:
                depths_list.append(depth)
                flops_list.append(flops)
                core_scores_list.append(core_score)

        # Print results table
        print("┌─────────┬────────────────────┬──────────────┐")
        print("│  Depth  │       FLOPs        │  CORE Score  │")
        print("├─────────┼────────────────────┼──────────────┤")

        for row in table_data:
            depth = row["depth"]
            flops = row["flops"]
            core_score = row["core_score"]

            flops_str = f"{flops:.3e}" if flops is not None else "N/A"
            score_str = f"{core_score:.4f}" if core_score is not None else "N/A"

            print(f"│   {depth:>2}    │  {flops_str:>16}  │   {score_str:>8}   │")

        print("└─────────┴────────────────────┴──────────────┘\n")

        # Create plot if we have valid data and plotting is available
        if PLOTTING_AVAILABLE and len(flops_list) > 0 and len(core_scores_list) > 0:
            print("📊 Generating plot: core_score_vs_flops.png\n")

            # Convert to numpy arrays
            flops_array = np.array(flops_list)
            core_scores_array = np.array(core_scores_list)

            # Fit linear regression: y = mx + b
            # x = log10(FLOPs), y = core_score
            log_flops = np.log10(flops_array)

            # Perform linear regression
            coeffs = np.polyfit(log_flops, core_scores_array, 1)
            slope, intercept = coeffs

            # Calculate R² (coefficient of determination)
            y_pred = slope * log_flops + intercept
            ss_res = np.sum(
                (core_scores_array - y_pred) ** 2
            )  # Residual sum of squares
            ss_tot = np.sum(
                (core_scores_array - np.mean(core_scores_array)) ** 2
            )  # Total sum of squares
            r_squared = 1 - (ss_res / ss_tot)

            # Print regression results
            print(f"{'='*80}")
            print("Linear Regression: CORE Score vs Log₁₀(FLOPs)")
            print(f"{'='*80}")
            print(f"Equation: y = {slope:.6f} * x + {intercept:.6f}")
            print("  where y = CORE Score (centered)")
            print("        x = Log₁₀(FLOPs)")
            print(f"\nR² (coefficient of determination): {r_squared:.6f}")
            print(f"{'='*80}\n")

            # Create plot
            plt.figure(figsize=(10, 6))

            # Plot actual data points
            plt.plot(
                log_flops,
                core_scores_array,
                "o",
                markersize=10,
                label="Actual",
                color="#2E86AB",
                zorder=3,
            )

            # Plot regression line
            x_line = np.linspace(log_flops.min(), log_flops.max(), 100)
            y_line = slope * x_line + intercept
            plt.plot(
                x_line,
                y_line,
                "-",
                linewidth=2,
                label="Linear Fit",
                color="#A23B72",
                alpha=0.7,
                zorder=2,
            )

            # Add labels for each point
            for i, depth in enumerate(depths_list):
                plt.annotate(
                    f"d{depth}",
                    (log_flops[i], core_scores_array[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )

            # Add regression equation and R² to plot
            textstr = f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r_squared:.4f}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            plt.text(
                0.05,
                0.95,
                textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

            plt.xlabel("Log₁₀(FLOPs)", fontsize=12)
            plt.ylabel("CORE Score (Centered)", fontsize=12)
            plt.title("CORE Score vs Training FLOPs", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower right", fontsize=10)

            # Save plot
            plot_path = os.path.join(
                os.path.dirname(__file__), "core_score_vs_flops.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✅ Plot saved to: {plot_path}\n")
        elif not PLOTTING_AVAILABLE:
            print("⚠️  Plotting disabled (matplotlib/numpy not installed)\n")
        else:
            print("⚠️  Insufficient data to create plot\n")

        print("✅ All evaluations complete!\n")

    # Cleanup distributed process group
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
