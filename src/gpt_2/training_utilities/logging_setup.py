"""Logging setup utilities for the trainer."""

import os
from datetime import datetime


def setup_logging(
    master_process: bool, depth: int = None, sft_training: bool = False
) -> str:
    """
    Setup generation log file for tracking model outputs.

    Args:
        master_process: Whether this is the master process
        depth: Model depth (for filename)
        sft_training: Whether doing SFT training (for filename)

    Returns:
        str: Path to the generation log file, or None if not master process
    """
    if master_process:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build filename with depth and training mode
        mode = "sft" if sft_training else "pretrain"
        if depth is not None:
            generation_log_file = os.path.join(
                log_dir, f"generations_{mode}_depth{depth}_{timestamp}.txt"
            )
        else:
            generation_log_file = os.path.join(
                log_dir, f"generations_{mode}_{timestamp}.txt"
            )

        print(f"📝 Saving generations to: {generation_log_file}")
        return generation_log_file
    else:
        return None
