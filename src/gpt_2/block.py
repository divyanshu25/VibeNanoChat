# Add gpt_2 to python path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
from gpt_2.mlp import MLP
from gpt_2.attention import CausalSelfAttention


class Block(nn.Module):
    """
    A single transformer block.
    This contains the self-attention and the feed-forward network.
    They are both preceeded by a layer normalization.
    The output of the self attention is fed into the feed-forward network.
    """

    def __init__(self, config):
        super().__init__()
        # Layer normalization before self-attention (pre-norm architecture)
        # This normalizes the input to have zero mean and unit variance
        self.ln_1 = nn.LayerNorm(config.n_embed)

        # Multi-head causal self-attention mechanism
        # This allows the model to attend to different positions in the sequence
        self.attn = CausalSelfAttention(config)

        # Layer normalization before MLP (pre-norm architecture)
        # Second normalization layer for the feed-forward network
        self.ln_2 = nn.LayerNorm(config.n_embed)

        # Multi-layer perceptron (feed-forward network)
        # This processes the attended representations
        self.mlp = MLP(config)

    def forward(self, x):
        # First residual connection: x + attention(norm(x))
        # Apply layer norm, then self-attention, then add residual connection
        # The residual connection helps with gradient flow and training stability
        x = x + self.attn(self.ln_1(x))  # shape: (B, T, n_embed)

        # Second residual connection: x + mlp(norm(x))
        # Apply layer norm, then MLP, then add residual connection
        # This creates a two-stage processing: attention -> feed-forward
        x = x + self.mlp(self.ln_2(x))  # shape: (B, T, n_embed)

        # Return the processed representation
        return x  # shape: (B, T, n_embed)
