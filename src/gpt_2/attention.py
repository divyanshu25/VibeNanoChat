import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
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

        # key, query, value projections
        # Q: n_embed, K: kv_dim, V: kv_dim
        self.c_attn = nn.Linear(config.n_embed, config.n_embed + 2 * kv_dim)

        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x, kv_cache=None):
        """
        Forward pass with optional KV caching for efficient generation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
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

        # Reshape Q with n_head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(
            1, 2
        )  # (B, n_head, T, head_dim)

        # Reshape K and V with n_kv_head
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(
            1, 2
        )  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(
            1, 2
        )  # (B, n_kv_head, T, head_dim)

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
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=(self.n_kv_head < self.n_head)
            )
        elif Tq == 1:
            # CASE 2: Single token generation (most common during inference)
            # The single query token can attend to ALL cached tokens (no masking needed)
            # This is super efficient: only 1 token forward pass!
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=(self.n_kv_head < self.n_head)
            )
        else:
            # CASE 3: Multi-token generation with cache
            # We have multiple queries but also cached keys/values
            # Need custom attention mask:
            # - Each query can attend to ALL cached tokens (prefix)
            # - Each query can only attend causally within the new tokens
            import torch

            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq

            # Allow attention to all prefix (cached) tokens
            attn_mask[:, :prefix_len] = True

            # Causal attention within the current chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )

            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=(self.n_kv_head < self.n_head)
            )

        # =====================================================================
        # STEP 4: Re-assemble heads and project output
        # =====================================================================
        # Flash attention with native GQA support (PyTorch 2.5+)
        # enable_gqa=True lets PyTorch handle the head broadcasting internally
        # This is more memory efficient than manually repeating K/V heads
        # Shape: (B, n_head, T, head_dim)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        y = self.c_proj(y)
        return y
