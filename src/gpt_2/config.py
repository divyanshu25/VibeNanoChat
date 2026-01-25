from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    This dataclass holds all the configuration parameters needed to define
    the architecture and training setup of the GPT model.
    """

    # ========================================================================
    # Model Architecture (NANOCHAT 560M CONFIG)
    # ========================================================================
    block_size: int = 2048  # Maximum sequence length (context window)
    vocab_size: int = 50266  # GPT-2 vocab (50257) + special tokens (9)
    # NOTE: Nanochat uses 65,536 with a custom tokenizer. We use GPT-2's tokenizer.
    n_layer: int = 10  # Number of transformer blocks in the model
    n_head: int = 10  # Number of attention heads per transformer block
    n_kv_head: int = 10  # Number of KV heads for GQA (MHA in this config)
    n_embed: int = (
        1280  # Embedding dimension (hidden size) = depth(20) * aspect_ratio(64)
    )

    # ========================================================================
    # Training Configuration (NANOCHAT SETTINGS)
    # ========================================================================
    num_epochs: int = (
        2  # Number of training epochs (nanochat uses iterations, not epochs)
    )
    batch_size: int = (
        32  # Batch size per GPU (32 sequences * 2048 tokens = 65,536 tokens/GPU)
    )
    total_batch_size: int = 524288  # Total tokens per gradient update (2^19)
    weight_decay: float = 0.10  # L2 regularization weight decay (nanochat default)
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping

    # Training horizon (nanochat-style calculation, priority order):
    # 1. num_iterations (if > 0): explicit number of optimization steps
    # 2. target_flops (if > 0): calculate iterations to reach target FLOPs
    # 3. target_param_data_ratio (if > 0): calculate iterations from data:param ratio (Chinchilla=20)
    num_iterations: int = (
        -1
    )  # Explicit number of optimization steps (-1 = calculate from ratio/flops)
    target_flops: float = -1.0  # Target total FLOPs (-1 = use param_data_ratio instead)
    target_param_data_ratio: int = 20  # Data:param ratio (Chinchilla optimal = 20)

    # ========================================================================
    # Learning Rate Schedule
    # ========================================================================
    # NOTE: Training steps are automatically calculated from target_param_data_ratio,
    # target_flops, or num_iterations (see "Training horizon" section above).
    # The trainer computes steps dynamically for all phases (pretrain/midtrain/sft).

    max_learning_rate: float = 6e-4  # Peak learning rate
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of peak (0.1 = 10% of peak)

    # Warmup steps for each training phase (as fraction of max_steps)
    lr_warmup_ratio_pretrain: float = 0.1  # Warmup as fraction of total steps (10%)
    lr_warmup_ratio_midtrain: float = 0.1  # Warmup as fraction of total steps (10%)
    lr_warmup_ratio_sft: float = 0.1  # Warmup as fraction of total steps (10%)

    # ========================================================================
    # Weight Tying
    # ========================================================================
    tie_embeddings: bool = True  # Tie input (wte) and output (lm_head) embeddings
    # Setting to False allows independent weights for embedding/unembedding

    # ========================================================================
    # Data Directories
    # ========================================================================
    data_dir_pretrain: str = "/sensei-fs/users/divgoyal/fineweb_edu"
    data_dir_midtrain: str = "/sensei-fs/users/divgoyal/nanochat_midtraining_data"
    sft_cache_dir: str = (
        "/sensei-fs/users/divgoyal/nanochat_midtraining_data"  # Cache dir for SFT datasets
    )

    # ========================================================================
    # Checkpointing
    # ========================================================================
    checkpoint_interval_pretrain: int = (
        5000  # Save checkpoint every N global steps (pretraining)
    )
    checkpoint_interval_midtrain: int = (
        400  # Save checkpoint every N global steps (mid-training)
    )
    checkpoint_interval_sft: int = 100  # Save checkpoint every N global steps (SFT)

    # ========================================================================
    # Evaluation Schedule
    # ========================================================================
    eval_interval: int = 100  # Run evaluations every N global steps
    val_loss_eval_batches: int = (
        37  # Number of batches for validation loss estimation (max safe value for 5M tokens with DDP)
    )

    # ========================================================================
    # Generation Sampling (during evaluation)
    # ========================================================================
    generation_num_samples: int = 4  # Number of sequences to generate per evaluation
    generation_max_length: int = 256  # Maximum tokens per generated sequence
    generation_seed: int = 42  # Random seed for reproducible generation
    use_kv_cache: bool = True  # Enable KV caching for faster generation (3-10x speedup)

    # ========================================================================
    # CORE Benchmark Evaluation (multiple choice tasks)
    # ========================================================================
    core_eval_max_examples: int = (
        500  # Max examples per task (for faster evals during training)
    )

    # ========================================================================
    # ChatCORE Evaluation (generative tasks like GSM8K)
    # ========================================================================
    chat_core_num_samples: int = 1  # Samples to generate per problem (for pass@k)
    chat_core_max_examples: int = (
        500  # Max examples per task (for faster evals during training)
    )
    chat_core_max_tokens: int = 512  # Maximum tokens per generation
    chat_core_temperature: float = 0.0  # Sampling temperature (0.0 = greedy decoding)
    chat_core_top_k: int = 50  # Top-k filtering for sampling
    chat_core_hf_cache_dir: str = (
        "/sensei-fs/users/divgoyal/nanochat_midtraining_data"  # HuggingFace cache directory for datasets
    )
