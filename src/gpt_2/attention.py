# Add gpt_2 to python path
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from gpt_2.rope import apply_rotary_emb


# =============================================================================
# Flash Attention 3 Detection and Loading
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 pre-compiled kernels (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Print which backend is being used
if HAS_FA3:
    print("✅ Flash Attention 3 is available and will be used (pre-compiled kernels)")
else:
    print("⚠️  Flash Attention 3 not available, falling back to PyTorch SDPA")


def attention_forward(q, k, v, is_causal=True, enable_gqa=False):
    """
    Unified attention interface that uses Flash Attention 3 if available,
    otherwise falls back to PyTorch SDPA (which uses FA2).

    Args:
        q: Query tensor (B, n_head, T, head_dim)
        k: Key tensor (B, n_kv_head, T, head_dim)
        v: Value tensor (B, n_kv_head, T, head_dim)
        is_causal: Whether to apply causal masking
        enable_gqa: Whether grouped-query attention is being used

    Returns:
        Output tensor (B, n_head, T, head_dim)
    """
    # Use FA3 only if available AND tensors are on CUDA (FA3 doesn't support CPU)
    # Use FA3 only if:
    # 1. FA3 is available
    # 2. Tensors are on CUDA (FA3 doesn't support CPU)
    # 3. Dtype is supported (fp16, bf16, or fp8_e4m3)
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    use_fa3 = HAS_FA3 and q.is_cuda and q.dtype in supported_dtypes

    if use_fa3:
        # Flash Attention 3 expects (B, T, n_head, head_dim)
        # Transpose: (B, n_head, T, head_dim) → (B, T, n_head, head_dim)
        B, n_head, T, head_dim = q.shape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Use Flash Attention 3 pre-compiled kernels
        output = _fa3.flash_attn_func(q, k, v, causal=is_causal)
        # Transpose back: (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        output = output.transpose(1, 2)
    else:
        # PyTorch SDPA expects (B, n_head, T, head_dim) - no transpose needed!
        # Handle GQA: if k/v have fewer heads than q, repeat them
        if enable_gqa and k.size(1) < q.size(1):
            # Repeat each k/v head to match q heads
            # E.g., if n_head=8 and n_kv_head=4, each kv head is used by 2 q heads
            n_head = q.size(1)
            n_kv_head = k.size(1)
            assert (
                n_head % n_kv_head == 0
            ), f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
            repeat_factor = n_head // n_kv_head

            # Store original dimensions before modification
            B, _, T, head_dim = k.shape

            # Shape: (B, n_kv_head, T, head_dim) → (B, n_kv_head, 1, T, head_dim)
            #     → (B, n_kv_head, repeat_factor, T, head_dim)
            #     → (B, n_head, T, head_dim)
            k = (
                k.unsqueeze(2)
                .repeat(1, 1, repeat_factor, 1, 1)
                .view(B, n_head, T, head_dim)
            )
            v = (
                v.unsqueeze(2)
                .repeat(1, 1, repeat_factor, 1, 1)
                .view(B, n_head, T, head_dim)
            )

        output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    return output


class CausalSelfAttention(nn.Module):
    # Class variable to track if we've printed attention info (shared across all instances)
    _printed_attention_info = False

    def __init__(self, config, layer_idx=None):
        """
        Initialize Causal Self-Attention layer.

        Args:
            config: Model configuration with attention parameters
            layer_idx: Index of this layer in the transformer (needed for KV caching)
        """
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # Store layer index for KV cache coordination
        self.layer_idx = layer_idx

        # Setup for GQA vs MHA
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head

        # If n_kv_head is None or equal to n_head, use MHA; otherwise use GQA
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )
        assert (
            config.n_head % self.n_kv_head == 0
        ), "n_head must be divisible by n_kv_head"

        # For GQA: Q has n_head, but K and V have n_kv_head
        kv_dim = self.head_dim * self.n_kv_head

        # key, query, value projections (nanochat-style: no bias)
        # Q: n_embed, K: kv_dim, V: kv_dim
        self.c_attn = nn.Linear(config.n_embed, config.n_embed + 2 * kv_dim, bias=False)

        # output projection (nanochat-style: no bias, zero-initialized)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x, cos_sin=None, kv_cache=None):
        """
        Forward pass with RoPE and optional KV caching for efficient generation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            cos_sin: Tuple of (cos, sin) tensors for RoPE, shape (1, seq_len, 1, head_dim//2)
            kv_cache: Optional KVCache object for efficient autoregressive generation

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimension

        # =====================================================================
        # STEP 1: Project input to Query, Key, Value
        # =====================================================================
        # Calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        # nh is number of heads, hs is head size and C is embedding dimension = nh * hs
        # e.g in GPT-2 (124M) n_head = 12 and hs = 64, so nh*hs = 768 channels
        qkv = self.c_attn(x)

        # Split into Q, K, V with potentially different sizes for GQA
        kv_dim = self.head_dim * self.n_kv_head
        q, k, v = qkv.split([self.n_embed, kv_dim, kv_dim], dim=2)

        # Reshape to (B, T, n_head, head_dim) for RoPE application
        q = q.view(B, T, self.n_head, self.head_dim)  # (B, T, n_head, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)  # (B, T, n_kv_head, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)  # (B, T, n_kv_head, head_dim)

        # =====================================================================
        # STEP 1.5: Apply RoPE to Q and K (nanochat-style)
        # =====================================================================
        if cos_sin is not None:
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)  # (B, T, n_head, head_dim)
            k = apply_rotary_emb(k, cos, sin)  # (B, T, n_kv_head, head_dim)

        # =====================================================================
        # STEP 1.6: QK Normalization (nanochat-style, critical for stability!)
        # =====================================================================
        # Normalize Q and K to prevent attention logits from exploding
        # This is especially important for deeper models (depth 8+)
        # RMSNorm is applied over the head_dim dimension
        q = F.rms_norm(q, (q.size(-1),))  # (B, T, n_head, head_dim)
        k = F.rms_norm(k, (k.size(-1),))  # (B, T, n_kv_head, head_dim)

        # Transpose to (B, n_head, T, head_dim) for attention computation
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # =====================================================================
        # STEP 2: Apply KV cache if provided
        # =====================================================================
        # If KV cache is active, insert new K,V and retrieve full cached history
        # This is the KEY optimization: we only compute K,V for new tokens,
        # then reuse all previously computed K,V from the cache
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # Now k, v contain all tokens (cached + current)
        Tq = q.size(2)  # Number of query positions (current forward pass)
        Tk = k.size(2)  # Number of key/value positions (cache + current)

        # =====================================================================
        # STEP 3: Compute attention with appropriate masking
        # =====================================================================
        # We need different attention strategies depending on the situation:
        # 1. Training (no cache): Standard causal attention
        # 2. Prefill (Tq == Tk): Processing prompt, use causal attention
        # 3. Single token generation (Tq == 1): Attend to all cached tokens
        # 4. Multi-token generation (Tq > 1, Tq < Tk): Mixed attention

        if kv_cache is None or Tq == Tk:
            # CASE 1: No KV cache (training) OR prefill phase (Tq == Tk)
            # Use standard causal attention: each position can only attend to
            # itself and previous positions
            y = attention_forward(
                q, k, v, is_causal=True, enable_gqa=(self.n_kv_head < self.n_head)
            )
        elif Tq == 1:
            # CASE 2: Single token generation (most common during inference)
            # The single query token can attend to ALL cached tokens (no masking needed)
            # This is super efficient: only 1 token forward pass!
            y = attention_forward(
                q, k, v, is_causal=False, enable_gqa=(self.n_kv_head < self.n_head)
            )
        else:
            # CASE 3: Multi-token generation with cache
            # We have multiple queries but also cached keys/values
            # Need custom attention mask:
            # - Each query can attend to ALL cached tokens (prefix)
            # - Each query can only attend causally within the new tokens
            #
            # NOTE: We use F.scaled_dot_product_attention here instead of flash_attn_func
            # because arbitrary attention masks are easier to handle with PyTorch SDPA.
            # This case is rare (only during multi-token generation with cache).

            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq

            # Allow attention to all prefix (cached) tokens
            attn_mask[:, :prefix_len] = True

            # Causal attention within the current chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )

            # PyTorch 2.5+ supports enable_gqa parameter for efficient GQA head broadcasting
            # This is more memory efficient than manually repeating K/V heads
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=(self.n_kv_head < self.n_head)
            )

        # =====================================================================
        # STEP 4: Re-assemble heads and project output
        # =====================================================================
        # Flash attention with native GQA support (PyTorch 2.5+)
        # enable_gqa=True lets PyTorch handle the head broadcasting internally
        # This is more memory efficient than manually repeating K/V heads
        # Shape: (B, n_head, Tq, head_dim) - note: Tq might differ from T when using KV cache

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)

        # Final output projection
        y = self.c_proj(y)
        return y
