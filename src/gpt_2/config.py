from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    This dataclass holds all the configuration parameters needed to define
    the architecture and training setup of the GPT model.
    """

    # ========================================================================
    # Model Architecture
    # ========================================================================
    block_size: int = 1024  # Maximum sequence length (context window)
    # Vocab size: 50257 (GPT-2) + 9 special tokens
    # Chat tokens: <|bos|>, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>
    # Tool tokens: <|python|>, <|python_end|>, <|ouput_start|>, <|output_end|>
    vocab_size: int = 50266  # Extended vocabulary (50257 base + 9 special tokens)
    n_layer: int = 12  # Number of transformer blocks in the model
    n_head: int = 12  # Number of attention heads per transformer block
    n_kv_head: int = 12  # Number of KV heads for GQA (None = MHA, uses n_head)
    n_embed: int = 768  # Embedding dimension (hidden size)

    # ========================================================================
    # Training Configuration
    # ========================================================================
    num_epochs: int = 6  # Number of training epochs
    batch_size: int = 64  # Batch size per GPU
    total_batch_size: int = 524288  # Total tokens per gradient update (2^19)
    weight_decay: float = 0.10  # L2 regularization weight decay
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping

    # ========================================================================
    # Learning Rate Schedule - Pretraining
    # ========================================================================
    max_learning_rate: float = 5e-6  # Peak learning rate
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of peak (0.1 = 10% of peak)
    lr_warmup_steps_pretrain: int = 715  # Linear warmup steps (pretraining)
    steps_per_epoch_pretrain: int = 18977  # Training steps per epoch (pretraining)

    # ========================================================================
    # Learning Rate Schedule - Mid-training
    # ========================================================================
    lr_warmup_steps_midtrain: int = 80  # Linear warmup steps (mid-training)
    steps_per_epoch_midtrain: int = 878  # Training steps per epoch (mid-training)

    # ========================================================================
    # Learning Rate Schedule - SFT (Supervised Fine-Tuning)
    # ========================================================================
    lr_warmup_steps_sft: int = 85  # Linear warmup steps (SFT)
    steps_per_epoch_sft: int = (
        336  # Training steps per epoch (SFT) - approximate for multiplex dataset
    )

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
