"""Weights & Biases setup utilities for the trainer."""

import wandb


def setup_wandb(
    master_process: bool,
    sft_training: bool,
    config,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    num_epochs: int,
    run_evals: bool,
    run_core_evals: bool,
    run_chatcore_evals: bool,
    start_step: int,
    flops_per_token: float,
    total_batch_size: int,
) -> None:
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        master_process: Whether this is the master process
        sft_training: Whether doing SFT training
        config: GPTConfig instance
        max_learning_rate: Maximum learning rate
        min_learning_rate: Minimum learning rate
        warmup_steps: Number of warmup steps
        max_steps: Maximum steps per epoch
        num_epochs: Number of training epochs
        run_evals: Whether running evaluations
        run_core_evals: Whether running CORE evaluations
        run_chatcore_evals: Whether running ChatCORE evaluations
        start_step: Starting step for resumed training
        flops_per_token: FLOPs per token
        total_batch_size: Total batch size
    """
    if not master_process:
        return

    # Determine project name and training mode
    if sft_training:
        project_name = "gpt2-sft"
        training_mode = "SFT"
    else:
        project_name = "gpt2-pretraining-bos"
        training_mode = "pretraining"

    # Calculate total FLOPs budget for this run
    total_flops = flops_per_token * total_batch_size * max_steps

    # Format FLOPs in scientific notation
    flops_str = f"{total_flops:.1e}"

    # Create run name: L{layers}-{flops}
    run_name = f"model_L{config.n_layer}-C{flops_str}"

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_type": "GPT-2",
            "training_mode": training_mode,
            "batch_size": config.batch_size,
            "block_size": config.block_size,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "num_epochs": num_epochs,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "run_evals": run_evals,
            "run_core_evals": run_core_evals,
            "run_chatcore_evals": run_chatcore_evals,
            "start_step": start_step,
            "n_layers": config.n_layer,
            "total_flops": total_flops,
            # Nanochat-style learning rate parameters
            "embedding_lr": config.embedding_lr,
            "unembedding_lr": config.unembedding_lr,
            "matrix_lr": config.matrix_lr,
            "scalar_lr": config.scalar_lr,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "warmup_ratio": config.warmup_ratio,
            "warmdown_ratio": config.warmdown_ratio,
            "final_lr_frac": config.final_lr_frac,
        },
    )
