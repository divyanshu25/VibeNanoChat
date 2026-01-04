"""
Shared utilities for NanoGPT project.

This module provides common functionality used across the project,
including the custom tokenizer with special tokens for chat format.
"""

import tiktoken


def get_special_tokens():
    """
    Define special tokens for chat format.

    Token IDs start at 50257 (right after GPT-2's vocab which ends at 50256).

    Returns:
        dict: Mapping of special token strings to their token IDs
    """
    return {
        "<|bos|>": 50257,  # Beginning of sequence - marks start of conversation
        "<|user_start|>": 50258,  # Marks start of user message
        "<|user_end|>": 50259,  # Marks end of user message
        "<|assistant_start|>": 50260,  # Marks start of assistant response
        "<|assistant_end|>": 50261,  # Marks end of assistant response
    }


def get_custom_tokenizer():
    """
    Create a custom tiktoken encoder with our special tokens registered.

    This extends the GPT-2 tokenizer by adding our 5 chat-format special tokens.
    The custom encoder can then handle these tokens natively via encode/decode.

    Returns:
        tuple: (custom_encoder, special_tokens_dict)
    """
    # Get the base GPT-2 encoding
    base_enc = tiktoken.get_encoding("gpt2")

    # Define our custom special tokens
    special_tokens = get_special_tokens()

    # Create a new encoding that includes both:
    # 1. GPT-2's existing special tokens (like <|endoftext|>)
    # 2. Our new chat-format special tokens
    enc = tiktoken.Encoding(
        name="nano_chat",  # Custom name for our extended tokenizer
        pat_str=base_enc._pat_str,  # Use same regex pattern for tokenization
        mergeable_ranks=base_enc._mergeable_ranks,  # Use same BPE merges
        special_tokens={
            **base_enc._special_tokens,  # Keep GPT-2's <|endoftext|> (id=50256)
            **special_tokens,  # Add our 5 new special tokens
        },
    )

    return enc, special_tokens
