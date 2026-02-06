"""
Configuration settings for the VibeNanoChat Server.
"""

import os


class ChatConfig:
    """Configuration class for the chat server."""

    # Directory containing model checkpoints
    CHECKPOINT_DIR = "<YOUR_PATH>/divgoyal/nanogpt/sft_checkpoints"

    # Session management
    SESSION_TIMEOUT_MINUTES = 60  # Sessions expire after 1 hour of inactivity

    # Default generation parameters
    DEFAULT_MAX_TOKENS = 256
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_K = 50

    # Server settings
    DEFAULT_PORT = int(os.environ.get("CHAT_SERVER_PORT", 8003))
    HOST = "0.0.0.0"

    # Generation limits
    MAX_TOKENS_LIMIT = 1024
    MIN_TOKENS_LIMIT = 1
    MAX_TEMPERATURE = 2.0
    MIN_TEMPERATURE = 0.0
    MAX_TOP_K = 1000
    MIN_TOP_K = 1

    # Special tokens
    CHAT_TOKENS = {
        "bos": "<|bos|>",
        "user_start": "<|user_start|>",
        "user_end": "<|user_end|>",
        "assistant_start": "<|assistant_start|>",
        "assistant_end": "<|assistant_end|>",
        "endoftext": "<|endoftext|>",
    }
