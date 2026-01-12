from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    This dataclass holds all the configuration parameters needed to define
    the architecture and training setup of the GPT model.
    """

    block_size: int = 1024  # Maximum sequence length (context window)
    # Vocab size: 50257 (GPT-2) + 5 special tokens for chat format
    # Special tokens: <|bos|>, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>
    vocab_size: int = 50262  # Extended vocabulary (50257 base + 5 special tokens)
    n_layer: int = 12  # Number of transformer blocks in the model
    n_head: int = 12  # Number of attention heads per transformer block
    n_kv_head: int = 12  # Number of KV heads for GQA (None = MHA, uses n_head)
    n_embed: int = 768  # Embedding dimension (hidden size)
    batch_size: int = 64  # Training batch size
    total_batch_size: int = 524288  # 2^19
    checkpoint_interval: int = 5000  # Save checkpoint every N steps
    eval_interval: int = 250  # Run evaluations every N steps
