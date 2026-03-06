"""
KV Cache utilities for efficient text generation.

This module provides utilities for KV (Key-Value) caching during autoregressive
text generation, which dramatically speeds up inference:

Performance:
- Without KV cache: O(N²) - reprocess all tokens at each step
- With KV cache: O(N) - reuse cached attention keys/values
- Result: 5-10x faster generation with ~2x memory usage

The generation process uses two phases:
1. PREFILL: Process all prompt tokens at once, cache their K/V pairs
2. DECODE: Generate one token at a time, reusing cached K/V pairs
"""

import os
import sys
from typing import List, Optional, Tuple

import torch

# Import KVCache from the model package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from gpt_2.kv_cache import KVCache


def get_model_config(model) -> Tuple[int, int, int, int]:
    """
    Extract model configuration for KV cache dimensions.

    Args:
        model: The model to extract config from

    Returns:
        Tuple of (num_heads, head_dim, num_layers, max_seq_len)
    """
    if hasattr(model, "config"):
        m = model.config
        num_heads = m.n_kv_head if hasattr(m, "n_kv_head") else m.n_head
        head_dim = m.n_embed // m.n_head
        num_layers = m.n_layer
        max_seq_len = m.block_size if hasattr(m, "block_size") else 2048
    else:
        # Fallback: use default dimensions
        print("Warning: Could not find model.config, using fallback dimensions")
        num_heads = 12
        head_dim = 64
        num_layers = 12
        max_seq_len = 2048

    return num_heads, head_dim, num_layers, max_seq_len


def create_kv_cache(
    prompt_length: int,
    max_tokens: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    max_seq_len: int,
    use_kv_cache: bool = True,
) -> Optional[KVCache]:
    """
    Create KV cache for generation if enabled.

    Args:
        prompt_length: Length of the prompt in tokens
        max_tokens: Maximum tokens to generate
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length supported by model
        use_kv_cache: Whether to create the cache

    Returns:
        KVCache instance if use_kv_cache is True, None otherwise
    """
    if not use_kv_cache:
        return None

    estimated_length = prompt_length + max_tokens
    if estimated_length > max_seq_len:
        estimated_length = max_seq_len

    return KVCache(
        batch_size=1,
        num_heads=num_heads,
        seq_len=estimated_length,
        head_dim=head_dim,
        num_layers=num_layers,
    )


def prefill_prompt(
    model,
    prompt_tokens: List[int],
    device,
    kv_cache: Optional[KVCache] = None,
) -> torch.Tensor:
    """
    Process all prompt tokens at once (prefill phase).

    This is the first phase of KV-cached generation where we process
    the entire prompt in parallel and cache the key/value pairs.

    Args:
        model: The language model
        prompt_tokens: List of token IDs for the prompt
        device: Device to run on
        kv_cache: Optional KV cache to populate

    Returns:
        next_token_logits: Logits for the position after the prompt
    """
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.amp.autocast(
        device_type=(device.type if hasattr(device, "type") else str(device)),
        dtype=torch.bfloat16,
    ):
        logits, _ = model(prompt_tensor, kv_cache=kv_cache)

    return logits[0, -1, :]


def forward_pass(
    model,
    token_or_tokens,
    device,
    kv_cache: Optional[KVCache] = None,
) -> torch.Tensor:
    """
    Run forward pass through model with autocast.

    This handles both single token decode (with KV cache) and
    multi-token processing (without KV cache or prefill).

    Args:
        model: The language model
        token_or_tokens: Either a single token (int) or list of tokens
        device: Device to run on
        kv_cache: KV cache to use (or None)

    Returns:
        next_token_logits: Logits for the next token position
    """
    # Prepare input tensor
    if isinstance(token_or_tokens, int):
        # Single token - for decode phase with KV cache
        input_tensor = torch.tensor(
            [[token_or_tokens]], dtype=torch.long, device=device
        )
    else:
        # Multiple tokens - for prefill or no-cache mode
        input_tensor = torch.tensor([token_or_tokens], dtype=torch.long, device=device)

    with torch.amp.autocast(
        device_type=(device.type if hasattr(device, "type") else str(device)),
        dtype=torch.bfloat16,
    ):
        logits, _ = model(input_tensor, kv_cache=kv_cache)

    return logits[0, -1, :]


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 50,
) -> int:
    """
    Sample next token from logits using temperature and top-k.

    Args:
        logits: Logits tensor for next token prediction
        temperature: Sampling temperature (0.0 = greedy/deterministic)
        top_k: Top-k sampling parameter (0 = no filtering)

    Returns:
        next_token: The sampled token ID
    """
    if temperature > 0:
        # Apply temperature scaling
        logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, top_k_indices, top_k_logits)

        # Sample from probability distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
    else:
        # Greedy decoding (deterministic)
        next_token = logits.argmax().item()

    return next_token
