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
from gpt_2.utils import has_value_embedding


# =============================================================================
# Flash Attention Detection and Loading
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


def _load_flash_attention_2():
    """Try to load Flash Attention 2 pre-compiled kernels."""
    if not torch.cuda.is_available():
        return None
    try:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        return get_kernel("kernels-community/flash-attn2")
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
_fa2 = _load_flash_attention_2()
HAS_FA3 = _fa3 is not None
HAS_FA2 = _fa2 is not None

# Print which backend is being used (Flash Attention is required for actual use)
if HAS_FA3:
    print("Flash Attention 3 loaded (pre-compiled kernels)")
elif HAS_FA2:
    print("Flash Attention 2 loaded (pre-compiled kernels)")
# Note: We don't raise an error here to allow module import for testing
# The error will be raised at runtime in attention_forward() if FA is actually used


def attention_forward(q, k, v, is_causal=True, enable_gqa=False, window_size=()):
    """
    Unified attention interface using Flash Attention 3 or 2 pre-compiled kernels,
    with fallback to PyTorch SDPA.

    Args:
        q: Query tensor (B, n_head, T, head_dim)
        k: Key tensor (B, n_kv_head, T, head_dim)
        v: Value tensor (B, n_kv_head, T, head_dim)
        is_causal: Whether to apply causal masking
        enable_gqa: Whether grouped-query attention is being used (handled natively by FA)
        window_size: Sliding window size tuple (left, right) for local attention.
                     left: how many tokens before to attend (-1 = unlimited)
                     right: how many tokens after to attend (0 for causal)
                     Empty tuple or large left value = full attention

    Returns:
        Output tensor (B, n_head, T, head_dim)
    """
    # Determine which Flash Attention kernel to use
    supported_dtypes_fa3 = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    supported_dtypes_fa2 = (torch.float16, torch.bfloat16)

    use_fa3 = HAS_FA3 and q.is_cuda and q.dtype in supported_dtypes_fa3
    use_fa2 = HAS_FA2 and q.is_cuda and q.dtype in supported_dtypes_fa2

    if use_fa3 or use_fa2:
        # Flash Attention expects (B, T, n_head, head_dim)
        # Transpose: (B, n_head, T, head_dim) → (B, T, n_head, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use the appropriate Flash Attention kernel
        # Note: FA expects window_size=(-1, -1) for full attention
        # Convert large window sizes to unlimited (-1) for Flash Attention
        T = q.size(1)
        if len(window_size) == 0 or window_size[0] >= T:
            # Full attention: empty tuple or window >= sequence length
            fa_window_size = (-1, -1)
        else:
            # Limited window: use as-is
            fa_window_size = window_size

        if use_fa3:
            output = _fa3.flash_attn_func(
                q, k, v, causal=is_causal, window_size=fa_window_size
            )
        else:
            output = _fa2.flash_attn_func(
                q, k, v, causal=is_causal, window_size=fa_window_size
            )

        # Transpose back: (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        return output.transpose(1, 2)
    else:
        # Fallback to PyTorch SDPA
        # Handle GQA by repeating K,V heads to match Q heads
        if enable_gqa:
            B, n_kv_head, T, head_dim = k.shape
            n_head = q.shape[1]
            repeat_factor = n_head // n_kv_head

            # Repeat each KV head repeat_factor times
            # (B, n_kv_head, T, head_dim) -> (B, n_head, T, head_dim)
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Create attention mask for sliding window if needed (nanochat-style)
        attn_mask = None
        T = q.shape[2]

        # Check if we need a sliding window mask
        # Skip mask creation if window_size is empty or left >= T (full attention)
        if len(window_size) > 0 and window_size[0] < T:
            left, right = (
                window_size  # (left, right) where left = window size, right = 0 for causal
            )

            # Build causal mask using torch.tril (lower triangular)
            # mask[i, j] = True means position i CAN attend to position j
            if is_causal:
                attn_mask = torch.tril(
                    torch.ones(T, T, device=q.device, dtype=torch.bool)
                )
            else:
                attn_mask = torch.ones(T, T, device=q.device, dtype=torch.bool)

            # Apply sliding window constraint (limit how far back we can look)
            if left >= 0:
                # Create position indices for distance calculation
                row_idx = torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
                col_idx = torch.arange(T, device=q.device).unsqueeze(0)  # (1, T)
                # Only keep positions within window distance
                attn_mask = attn_mask & ((row_idx - col_idx) <= left)

            # Convert to additive mask for SDPA (False -> -inf, True -> 0)
            attn_mask = torch.where(attn_mask, 0.0, float("-inf"))

        # Use PyTorch's scaled_dot_product_attention
        # Already in correct format: (B, n_head, T, head_dim)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(is_causal and attn_mask is None),
        )

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

        # Value embedding gate (nanochat-style): input-dependent mixing
        # Only create gate for layers that actually have value embeddings
        self.value_embed_gate_channels = 32
        self.value_embed_gate = (
            nn.Linear(self.value_embed_gate_channels, self.n_kv_head, bias=False)
            if has_value_embedding(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, value_embed=None, cos_sin=None, kv_cache=None, window_size=()):
        """
        Forward pass with RoPE, optional KV caching, and sliding window attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            value_embed: Optional value embedding tensor (batch_size, seq_len, kv_dim)
            cos_sin: Tuple of (cos, sin) tensors for RoPE, shape (1, seq_len, 1, head_dim//2)
            kv_cache: Optional KVCache object for efficient autoregressive generation
            window_size: Tuple (left, right) for sliding window attention.
                        left: tokens before to attend (-1 = unlimited)
                        right: tokens after to attend (0 for causal)
                        Empty tuple or large left = full attention

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
        # STEP 1.4: Mix in Value Embedding (nanochat-style)
        # =====================================================================
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if value_embed is not None and self.value_embed_gate is not None:
            value_embed = value_embed.view(B, T, self.n_kv_head, self.head_dim)
            # Gate is computed from first 32 channels of input, range (0, 2)
            gate = 2 * torch.sigmoid(
                self.value_embed_gate(x[..., : self.value_embed_gate_channels])
            )  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * value_embed  # Add gated value embedding

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

        if kv_cache is None or Tq == Tk:
            # CASE 1: No KV cache (training) OR prefill phase (Tq == Tk)
            # Use standard causal attention with optional sliding window
            y = attention_forward(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=(self.n_kv_head < self.n_head),
                window_size=window_size,
            )
        elif Tq == 1:
            # CASE 2: Single token generation (most common during inference)
            # The single query token can attend to ALL cached tokens (no masking needed)
            # Note: During inference, we ignore window_size since we need full cache access
            y = attention_forward(
                q, k, v, is_causal=False, enable_gqa=(self.n_kv_head < self.n_head)
            )
        else:
            raise RuntimeError(
                f"Unsupported attention mode: Tq={Tq}, Tk={Tk}. "
                "Only single-token generation (Tq=1) is supported with KV cache."
            )

        # =====================================================================
        # STEP 4: Re-assemble heads and project output
        # =====================================================================
        # Flash Attention handles GQA natively (no manual head broadcasting needed)
        # Output shape: (B, n_head, Tq, head_dim) - note: Tq might differ from T when using KV cache

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)

        # Final output projection
        y = self.c_proj(y)
        return y
