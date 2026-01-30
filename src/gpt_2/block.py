# Add gpt_2 to python path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch.nn as nn
import torch.nn.functional as F

from gpt_2.attention import CausalSelfAttention
from gpt_2.mlp import MLP


class Block(nn.Module):
    """
    A single transformer block with optional KV caching support.

    This contains the self-attention and the feed-forward network.
    They are both preceded by RMSNorm (nanochat-style).
    The output of the self attention is fed into the feed-forward network.
    """

    def __init__(self, config, layer_idx=None):
        """
        Initialize a transformer block.

        Args:
            config: Model configuration
            layer_idx: Index of this block in the transformer (needed for KV caching)
        """
        super().__init__()
        # Multi-head causal self-attention mechanism with KV cache support
        # This allows the model to attend to different positions in the sequence
        # layer_idx is passed to enable KV caching during inference
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)

        # Multi-layer perceptron (feed-forward network)
        # This processes the attended representations
        self.mlp = MLP(config)

        # Note: We use functional RMSNorm (no learnable params) instead of LayerNorm

    def forward(self, x, cos_sin=None, kv_cache=None):
        """
        Forward pass through transformer block with RoPE and optional KV caching.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            cos_sin: Tuple of (cos, sin) tensors for RoPE
            kv_cache: Optional KVCache object for efficient generation

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        # First residual connection: x + attention(rms_norm(x))
        # Apply RMSNorm, then self-attention with RoPE (with optional caching),
        # then add residual connection
        # The residual connection helps with gradient flow and training stability
        x = x + self.attn(
            F.rms_norm(x, (x.size(-1),)), cos_sin=cos_sin, kv_cache=kv_cache
        )  # shape: (B, T, n_embed)

        # Second residual connection: x + mlp(rms_norm(x))
        # Apply RMSNorm, then MLP, then add residual connection
        # This creates a two-stage processing: attention -> feed-forward
        # Note: MLP doesn't use KV cache (only attention needs it)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))  # shape: (B, T, n_embed)

        # Return the processed representation
        return x  # shape: (B, T, n_embed)
