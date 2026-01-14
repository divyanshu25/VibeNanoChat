"""
KV Cache implementation for efficient transformer inference.

This module provides a Key-Value cache that dramatically speeds up autoregressive
text generation by avoiding redundant attention computations.

Performance impact:
- Speed: 5-10x faster generation
- Memory: ~2x more VRAM (worth the tradeoff!)

Inspired by the implementation in nanochat/engine.py
"""

import torch


class KVCache:
    """
    Key-Value cache for efficient transformer inference.
    
    ## Why KV Caching?
    
    Without KV caching, autoregressive generation recomputes attention for all
    previous tokens at each step, resulting in O(N²) computation complexity.
    
    Example without cache (inefficient):
    ```
    Step 1: Process "The" → generate "cat"
    Step 2: Process "The cat" → generate "sat" (recomputes "The" attention!)
    Step 3: Process "The cat sat" → generate "on" (recomputes "The" and "cat"!)
    ```
    
    With KV caching, we store the Key and Value tensors from previous tokens
    and reuse them, achieving O(N) computation instead.
    
    ## How It Works
    
    In transformer attention, for each token we compute three vectors:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What information do I provide?"
    
    The attention formula is:
    ```
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    ```
    
    **Key insight**: Keys and Values for previously processed tokens never change!
    
    So we:
    1. Compute K and V once for each token
    2. Store them in a cache
    3. Reuse them for all future tokens
    
    ## Cache Structure
    
    The cache has shape: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
                          └─────────┘  │  └─────────┘  └───────┘  └──────┘  └───────┘
                          Layer count  │  Batch size   Num heads  Sequence  Head dim
                                       │                          length
                                       └─ 2 for K and V
    
    Each layer of the transformer gets its own K and V cache.
    
    ## Usage Example
    
    ```python
    # Initialize cache for generation
    kv_cache = KVCache(
        batch_size=1,
        num_heads=12,
        seq_len=2048,
        head_dim=64,
        num_layers=12
    )
    
    # Prefill phase: Process prompt tokens (can be batched)
    prompt_tensor = torch.tensor([prompt_tokens], device=device)
    logits = model(prompt_tensor, kv_cache=kv_cache)
    # Now cache contains K,V for all prompt tokens
    
    # Decode phase: Generate one token at a time (super fast!)
    for _ in range(max_tokens):
        next_token = sample_from_logits(logits)
        
        # Only process ONE new token (huge speedup!)
        token_tensor = torch.tensor([[next_token]], device=device)
        logits = model(token_tensor, kv_cache=kv_cache)
        # Cache automatically stores K,V for this new token
    ```
    
    ## Performance
    
    For a model generating 100 tokens:
    - Without KV cache: ~10 seconds (O(N²) computation)
    - With KV cache: ~1 second (O(N) computation)
    
    Memory cost example (12 layers, 12 heads, head_dim=64, seq_len=2048, batch=1):
    ```
    12 layers × 2 (K+V) × 1 batch × 12 heads × 2048 seq × 64 dim × 2 bytes
    = ~38 MB per sequence
    ```
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        Initialize KV cache with specified dimensions.
        
        The cache is lazily initialized on first use (we need to know the
        dtype and device from actual tensors).
        
        Args:
            batch_size: Number of sequences to generate in parallel
            num_heads: Number of attention heads in the model
            seq_len: Maximum sequence length to support (can grow dynamically)
            head_dim: Dimension of each attention head
            num_layers: Number of transformer layers in the model
        """
        # Cache shape: one K and V per layer, per head, per position
        # Each K/V is of shape (batch_size, num_heads, seq_len, head_dim)
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        
        # Cache is lazily initialized (we don't know dtype/device yet)
        self.kv_cache = None
        
        # Current position: how many tokens have been processed
        self.pos = 0

    def reset(self):
        """
        Reset cache position to beginning.
        
        Use this when starting a new generation sequence. The cache memory
        is reused, but pos is reset to 0.
        """
        self.pos = 0

    def get_pos(self):
        """
        Get current position in cache (number of tokens cached).
        
        Returns:
            int: Number of tokens currently stored in cache
        """
        return self.pos

    def insert_kv(self, layer_idx, k, v):
        """
        Insert new Key and Value tensors into the cache at current position.
        
        This method is called during the model's forward pass for each transformer
        layer. It stores the K and V for the new token(s) and returns a view of
        all cached K and V up to the current position.
        
        The cache is lazily initialized on the first call (now we know dtype/device).
        If the cache runs out of space, it automatically grows.
        
        Args:
            layer_idx: Which transformer layer this is (0-indexed)
            k: Key tensor of shape (batch_size, num_heads, num_new_tokens, head_dim)
            v: Value tensor of shape (batch_size, num_heads, num_new_tokens, head_dim)
            
        Returns:
            tuple: (cached_keys, cached_values) containing all tokens processed so far
                   - cached_keys: shape (batch_size, num_heads, total_tokens, head_dim)
                   - cached_values: shape (batch_size, num_heads, total_tokens, head_dim)
        
        Example:
            >>> # First call (prefill with 5 prompt tokens)
            >>> k_cached, v_cached = cache.insert_kv(0, k_prompt, v_prompt)
            >>> k_cached.shape  # (1, 12, 5, 64) - all prompt tokens
            >>> 
            >>> # Second call (generate 1 new token)
            >>> k_cached, v_cached = cache.insert_kv(0, k_new, v_new)
            >>> k_cached.shape  # (1, 12, 6, 64) - prompt + new token
        """
        # =================================================================
        # LAZY INITIALIZATION
        # =================================================================
        # On first call, create the cache tensor with correct dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        
        # =================================================================
        # DETERMINE INSERTION RANGE
        # =================================================================
        # Extract shape info from incoming K tensor
        B, H, T_add, D = k.size()
        
        # Calculate where to insert: [t0, t1)
        t0 = self.pos  # Start at current position
        t1 = self.pos + T_add  # End after adding T_add new tokens
        
        # =================================================================
        # DYNAMIC CACHE GROWTH (if needed)
        # =================================================================
        # If we're about to exceed cache capacity, grow it
        # This handles cases where generation exceeds initial seq_len estimate
        if t1 > self.kv_cache.size(4):
            # Calculate how much more space we need
            t_needed = t1 + 1024  # Add current need + buffer of 1024 tokens
            t_needed = (t_needed + 1023) & ~1023  # Round up to multiple of 1024
            
            # Create additional cache space
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(
                additional_shape, dtype=k.dtype, device=k.device
            )
            
            # Concatenate with existing cache along sequence dimension (dim=4)
            self.kv_cache = torch.cat(
                [self.kv_cache, additional_cache], dim=4
            ).contiguous()
            
            # Update shape to reflect new size
            self.kv_shape = self.kv_cache.shape
        
        # =================================================================
        # INSERT K,V INTO CACHE
        # =================================================================
        # Store keys at: cache[layer_idx, 0 (for K), :, :, t0:t1, :]
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        
        # Store values at: cache[layer_idx, 1 (for V), :, :, t0:t1, :]
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        
        # =================================================================
        # RETURN VIEW OF ALL CACHED K,V
        # =================================================================
        # Return all keys and values up to current position (t1)
        # This is what the attention mechanism will use
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        
        # =================================================================
        # ADVANCE POSITION (after last layer)
        # =================================================================
        # Only increment position counter after the last layer processes
        # This ensures all layers process the same tokens before advancing
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        
        return key_view, value_view

    def prefill(self, other):
        """
        Prefill this cache by copying from another KV cache.
        
        This enables an efficient pattern:
        1. Process prompt once with batch_size=1 (fast single-batch prefill)
        2. Clone cache to batch_size=N (parallel generation)
        3. Generate N different completions in parallel
        
        All N completions share the same prompt computation, which is a
        huge efficiency gain!
        
        Args:
            other: Another KVCache instance to copy from
            
        Raises:
            AssertionError: If dimensions are incompatible
            
        Example:
            >>> # Process prompt once
            >>> cache_single = KVCache(batch_size=1, ...)
            >>> logits = model(prompt, kv_cache=cache_single)
            >>> 
            >>> # Generate 5 completions from same prompt
            >>> cache_batch = KVCache(batch_size=5, ...)
            >>> cache_batch.prefill(cache_single)  # Reuse prompt computation!
            >>> # Now generate 5 different completions in parallel
        """
        # =================================================================
        # VALIDATION
        # =================================================================
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill from an empty KV cache"
        
        # Extract dimensions explicitly for clear error messages
        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = (
            self.kv_shape
        )
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = (
            other.kv_shape
        )
        
        # Validate that all dimensions match (except batch and seq can differ)
        assert self_layers == other_layers, (
            f"Layer count mismatch: {self_layers} != {other_layers}"
        )
        assert self_kv == other_kv, f"K/V dimension mismatch: {self_kv} != {other_kv}"
        assert self_heads == other_heads, (
            f"Head count mismatch: {self_heads} != {other_heads}"
        )
        assert self_head_dim == other_head_dim, (
            f"Head dim mismatch: {self_head_dim} != {other_head_dim}"
        )
        
        # Batch size can be expanded (other can be 1, self can be larger)
        assert self_batch == other_batch or other_batch == 1, (
            f"Batch size mismatch: {self_batch} vs {other_batch} "
            "(other must be 1 or equal)"
        )
        
        # Self must have enough space for other's cached tokens
        assert self_seq >= other_seq, (
            f"Sequence length mismatch: {self_seq} < {other_seq}"
        )
        
        # =================================================================
        # INITIALIZATION
        # =================================================================
        # Create cache with same dtype/device as source
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        
        # =================================================================
        # COPY DATA
        # =================================================================
        # Copy all cached K,V from other cache (up to other.pos)
        # If other has batch=1 and self has batch>1, broadcasting happens
        self.kv_cache[:, :, :, :, : other.pos, :] = other.kv_cache[
            :, :, :, :, : other.pos, :
        ]
        
        # =================================================================
        # UPDATE POSITION
        # =================================================================
        # Start from same position as source cache
        self.pos = other.pos
