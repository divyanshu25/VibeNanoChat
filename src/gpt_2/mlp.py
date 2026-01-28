import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in transformer blocks.

    This implements the feed-forward network that follows the multi-head attention
    in each transformer block. It consists of two linear transformations with a
    Squared ReLU activation in between (nanochat-style).

    Squared ReLU: relu(x)^2
    - More expressive than standard ReLU
    - Smoother gradients than ReLU
    - Used in modern architectures (PaLM, nanochat)
    """

    def __init__(self, config):
        """
        Initialize the MLP layers.

        Args:
            config: Configuration object containing model hyperparameters
                   - n_embed: The embedding dimension / model width
        """
        super().__init__()

        # First linear layer: expand from n_embed to 4 * n_embed
        # This expansion is standard in transformer architectures
        # No bias (nanochat-style)
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)

        # Second linear layer: project back from 4 * n_embed to n_embed
        # This creates a bottleneck that helps with feature learning
        # No bias (nanochat-style)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)

        # Custom attribute for initialization scaling
        # This flag is used by the model's weight initialization routine
        # to apply zero-initialization to this layer's weights (residual projection)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        Forward pass through the MLP with Squared ReLU activation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        # Expand: n_embed -> 4 * n_embed
        x = self.c_fc(x)

        # Apply Squared ReLU activation: relu(x)^2 (nanochat-style)
        # This provides smoother gradients and more expressivity than standard ReLU
        x = F.relu(x).square()

        # Project back: 4 * n_embed -> n_embed
        x = self.c_proj(x)

        return x
