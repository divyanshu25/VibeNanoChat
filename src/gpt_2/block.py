# Add gpt_2 to python path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch.nn as nn

from gpt_2.attention import CausalSelfAttention
from gpt_2.mlp import MLP


class Block(nn.Module):
    """
    A single transformer block with optional KV caching support.

    This contains the self-attention and the feed-forward network.
    They are both preceded by a layer normalization.
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
        # Layer normalization before self-attention (pre-norm architecture)
        # This normalizes the input to have zero mean and unit variance
        self.ln_1 = nn.LayerNorm(config.n_embed)

        # Multi-head causal self-attention mechanism with KV cache support
        # This allows the model to attend to different positions in the sequence
        # layer_idx is passed to enable KV caching during inference
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)

        # Layer normalization before MLP (pre-norm architecture)
        # Second normalization layer for the feed-forward network
        self.ln_2 = nn.LayerNorm(config.n_embed)

        # Multi-layer perceptron (feed-forward network)
        # This processes the attended representations
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        """
        Forward pass through transformer block with optional KV caching.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            kv_cache: Optional KVCache object for efficient generation

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        # First residual connection: x + attention(norm(x))
        # Apply layer norm, then self-attention (with optional caching),
        # then add residual connection
        # The residual connection helps with gradient flow and training stability
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)  # shape: (B, T, n_embed)

        # Second residual connection: x + mlp(norm(x))
        # Apply layer norm, then MLP, then add residual connection
        # This creates a two-stage processing: attention -> feed-forward
        # Note: MLP doesn't use KV cache (only attention needs it)
        x = x + self.mlp(self.ln_2(x))  # shape: (B, T, n_embed)

        # Return the processed representation
        return x  # shape: (B, T, n_embed)
