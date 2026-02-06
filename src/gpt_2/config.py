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
        64  # Target dimension per attention head (for Flash Attention efficiency)
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
    target_flops: float = -1  # Target total FLOPs (-1 = use param_data_ratio instead)
    target_param_data_ratio: int = 10  # Data:param ratio (Chinchilla optimal = 20)

    # ========================================================================
    # Learning Rate Schedule (Nanochat-style)
    # ========================================================================
    # NOTE: Training steps are automatically calculated from target_param_data_ratio,
    # target_flops, or num_iterations (see "Training horizon" section above).
    # The trainer computes steps dynamically for all phases (pretrain/sft).

    # ========================================================================
    # Optimizer Configuration (Nanochat-style per-parameter-group LRs)
    # ========================================================================
    # Separate learning rates for different parameter groups (nanochat-style)
    embedding_lr: float = 0.3  # Learning rate for embedding parameters (Adam)
    unembedding_lr: float = 0.004  # Learning rate for unembedding parameters (Adam)
    matrix_lr: float = 0.02  # Learning rate for matrix parameters (Muon)
    scalar_lr: float = 0.5  # Learning rate for scalars (if any)

    # Adam optimizer hyperparameters (for embeddings/unembedding)
    adam_beta1: float = 0.8  # Adam beta1 for embedding/unembedding
    adam_beta2: float = 0.95  # Adam beta2 for embedding/unembedding

    # Learning rate schedule (nanochat-style warmup/warmdown)
    warmup_ratio: float = 0.0  # Ratio of iterations for LR warmup
    warmdown_ratio: float = 0.4  # Ratio of iterations for LR warmdown
    final_lr_frac: float = 0.0  # Final LR as fraction of initial LR

    # ========================================================================
    # Data Directories
    # ========================================================================
    # Standard binary dataloaders (streaming, no BOS alignment):
    # Parquet dataloader (BOS-aligned, nanochat-style) - now the only option
    data_dir_pretrain_parquet: str = "/sensei-fs/users/divgoyal/fineweb_edu_parquet"

    sft_cache_dir: str = (
        "/sensei-fs/users/divgoyal/sft_cache"  # Cache dir for SFT datasets
    )

    # ========================================================================
    # Dataloader Configuration
    # ========================================================================
    bos_dataloader_buffer_size: int = (
        4096  # Document buffer size for BOS-aligned packing
    )
    # Note: num_workers and prefetch_factor are now auto-determined based on model depth (see fineweb_edu_parquet_bos_dataloader.py)
    dataloader_persistent_workers: bool = True  # Keep workers alive between epochs

    # ========================================================================
    # Checkpointing
    # ========================================================================
    checkpoint_interval_pretrain: int = (
        5000  # Save checkpoint every N global steps (pretraining)
    )
    checkpoint_interval_sft: int = 100  # Save checkpoint every N global steps (SFT)

    # ========================================================================
    # Evaluation Schedule
    # ========================================================================
    eval_interval: int = (
        2000  # Run validation loss evaluations every N global steps (note: defaults to adaptive based on total_steps if not overridden)
    )
    core_eval_interval: int = (
        2000  # Run CORE benchmark evaluations every N global steps (note: defaults to adaptive based on total_steps if not overridden)
    )
    val_loss_eval_tokens: int | None = (
        10485760  # Number of tokens for validation loss estimation (None = use nanochat default: 20 * 524288 = ~10.5M tokens)
    )

    # ========================================================================
    # Generation Sampling (during evaluation)
    # ========================================================================
    enable_sampling: bool = False  # Enable text generation sampling during training
    generation_num_samples: int = 4  # Number of sequences to generate per evaluation
    generation_max_length: int = 256  # Maximum tokens per generated sequence
    generation_seed: int = 42  # Random seed for reproducible generation
    use_kv_cache: bool = True  # Enable KV caching for faster generation (3-10x speedup)
    generation_verbose: bool = (
        False  # Print verbose progress during generation (every 10 tokens)
    )
    generation_temperature: float = (
        0.8  # Sampling temperature for generation (higher = more random)
    )
    generation_top_k: int = 50  # Top-k sampling parameter
    generation_repetition_penalty: float = (
        1.2  # Penalty for repeating tokens (>1.0 discourages repetition)
    )

    # ========================================================================
    # CORE Benchmark Evaluation (multiple choice tasks)
    # ========================================================================
    core_eval_max_examples: int | None = (
        500  # Max examples per task (None = all examples, int = limit for faster evals)
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
        "/sensei-fs/users/divgoyal/sft_cache"  # HuggingFace cache directory for datasets
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
