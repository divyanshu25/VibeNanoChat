from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    This dataclass holds all the configuration parameters needed to define
    the architecture and training setup of the GPT model.

    Model architecture can be specified in two ways:
    1. nanochat-style: Set 'depth' and 'aspect_ratio', which auto-calculates n_layer, n_embed, n_head
    2. Traditional: Set n_layer, n_embed, n_head, n_kv_head directly
    """

    # ========================================================================
    # Model Architecture - nanochat-style (preferred for scaling laws)
    # ========================================================================
    # Set depth to auto-calculate architecture dimensions (depth × aspect_ratio design)
    # When depth > 0: n_layer = depth, n_embed = depth × aspect_ratio (rounded to head_dim multiple)
    depth: int = -1  # Model depth (-1 = use n_layer/n_embed instead)
    aspect_ratio: int = (
        64  # Multiplier for model_dim (model_dim = depth × aspect_ratio)
    )
    head_dim: int = (
        128  # Target dimension per attention head (for Flash Attention efficiency)
    )

    # ========================================================================
    # Model Architecture - Traditional (used when depth = -1)
    # ========================================================================
    block_size: int = 2048  # Maximum sequence length (context window)
    vocab_size: int = 50266  # GPT-2 vocab (50257) + special tokens (9)
    # NOTE: Nanochat uses 65,536 with a custom tokenizer. We use GPT-2's tokenizer.
    n_layer: int = 6  # Number of transformer blocks in the model
    n_head: int = 10  # Number of attention heads per transformer block
    n_kv_head: int = 10  # Number of KV heads for GQA (MHA in this config)
    n_embed: int = (
        1280  # Embedding dimension (hidden size) = depth(20) * aspect_ratio(64)
    )

    # ========================================================================
    # Training Configuration (NANOCHAT SETTINGS)
    # ========================================================================
    num_epochs: int = (
        1  # Number of training epochs (nanochat uses iterations, not epochs)
    )
    batch_size: int = (
        16  # Batch size per GPU (32 sequences * 2048 tokens = 65,536 tokens/GPU)
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
    target_flops: float = 1e18  # Target total FLOPs (-1 = use param_data_ratio instead)
    target_param_data_ratio: int = -1  # Data:param ratio (Chinchilla optimal = 20)

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
    lr_warmup_ratio_midtrain: float = 0.2  # Warmup as fraction of total steps (10%)
    lr_warmup_ratio_sft: float = 0.4  # Warmup as fraction of total steps (10%)

    # ========================================================================
    # Weight Tying
    # ========================================================================
    tie_embeddings: bool = False  # Tie input (wte) and output (lm_head) embeddings
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
    eval_interval: int = (
        500  # Run validation loss evaluations every N global steps (note: defaults to adaptive based on total_steps if not overridden)
    )
    core_eval_interval: int = (
        2000  # Run CORE benchmark evaluations every N global steps (note: defaults to adaptive based on total_steps if not overridden)
    )
    val_loss_eval_batches: int = (
        76  # Number of batches for validation loss estimation (max safe value for 5M tokens with DDP)
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

    def __post_init__(self):
        """
        Auto-calculate model architecture dimensions from depth if specified.

        This implements nanochat's depth × aspect_ratio parameterization:
        - n_layer = depth
        - base_dim = depth × aspect_ratio
        - n_embed = round base_dim up to nearest multiple of head_dim
        - n_head = n_embed // head_dim
        - n_kv_head = n_head (1:1 GQA ratio, i.e., MHA)
        """
        if self.depth > 0:
            # Calculate dimensions from depth
            self.n_layer = self.depth
            base_dim = self.depth * self.aspect_ratio

            # Round up to nearest multiple of head_dim for clean division
            # This ensures: n_embed % head_dim == 0
            self.n_embed = (
                (base_dim + self.head_dim - 1) // self.head_dim
            ) * self.head_dim

            # Calculate number of heads
            self.n_head = self.n_embed // self.head_dim
            self.n_kv_head = self.n_head  # Default: 1:1 GQA ratio (MHA)

            # Calculate the "nudge" for logging
            nudge = self.n_embed - base_dim

            # Store calculated values for later reference
            self._depth_mode = True
            self._base_dim = base_dim
            self._nudge = nudge
        else:
            self._depth_mode = False
            self._base_dim = None
            self._nudge = 0
