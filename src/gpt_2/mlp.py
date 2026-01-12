import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in GPT-2 transformer blocks.

    This implements the feed-forward network that follows the multi-head attention
    in each transformer block. It consists of two linear transformations with a
    GELU activation in between, following the standard transformer architecture.
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
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)

        # GELU activation function with tanh approximation
        # GELU (Gaussian Error Linear Unit) is preferred over ReLU in transformers
        # as it provides smoother gradients and better performance
        self.gelu = nn.GELU(
            approximate="tanh"
        )  # GELU is a non-linear activation function that is used in the feed-forward network.

        # Second linear layer: project back from 4 * n_embed to n_embed
        # This creates a bottleneck that helps with feature learning
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

        # Custom attribute for initialization scaling
        # This flag is used by the model's weight initialization routine
        # to apply special scaling to this layer's weights
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        # Expand: n_embed -> 4 * n_embed
        x = self.c_fc(x)

        # Apply non-linear activation
        x = self.gelu(x)

        # Project back: 4 * n_embed -> n_embed
        x = self.c_proj(x)

        return x
