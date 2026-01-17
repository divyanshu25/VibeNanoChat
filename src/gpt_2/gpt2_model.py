# Add gpt_2 to python path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt_2.block import Block
from gpt_2.config import GPTConfig
from gpt_2.utils import get_custom_tokenizer


class GPT(nn.Module):
    """
    The GPT (Generative Pre-trained Transformer) language model.

    This class implements the core GPT architecture with:
    - Token and position embeddings
    - Stack of transformer blocks
    - Layer normalization
    - Language modeling head for next token prediction
    """

    def __init__(self, config):
        """
        Initialize the GPT model with the given configuration.

        Args:
            config (GPTConfig): Configuration object containing model hyperparameters
        """
        super().__init__()
        self.config = config
        self.max_seq_len = config.block_size  # Maximum sequence length for evaluation

        # Define the main transformer components
        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: convert token indices to dense vectors
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                # Position embeddings: add positional information to tokens
                wpe=nn.Embedding(config.block_size, config.n_embed),
                # Stack of transformer blocks (the core of the model)
                # Each block gets a layer_idx for KV cache coordination
                h=nn.ModuleList(
                    [Block(config, layer_idx=i) for i in range(config.n_layer)]
                ),
                # Final layer normalization before output
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

        # Language modeling head: projects hidden states to vocabulary logits
        # Note: We'll tie the weights with the embedding layer
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight sharing scheme: tie input and output embeddings
        # This is a common technique in language models to reduce parameters
        # and improve performance
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights using custom initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights using appropriate strategies for different layer types.

        Args:
            module: The PyTorch module to initialize
        """
        # Base standard deviation for weight initialization
        std = 0.02

        # Scale down initialization for residual layers (helps with training stability)
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5

        # Initialize linear layers with normal distribution
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Initialize embedding layers with normal distribution
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, learning_rate, weight_decay, device):
        """
        Configure the optimizer for training with weight decay regularization.

        This method separates parameters into two groups:
        - Parameters with weight decay (typically weights of linear layers)
        - Parameters without weight decay (typically biases and embeddings)

        Args:
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay coefficient for regularization
            device (str): Device type ('cuda', 'mps', or 'cpu')

        Returns:
            torch.optim.AdamW: Configured optimizer
        """
        # Get all trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate parameters based on dimensionality
        # 2D+ parameters (weights) get weight decay, 1D parameters (biases) don't
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        # Create optimizer parameter groups
        optim_groups = [
            {"params": [p for p in decay_params], "weight_decay": weight_decay},
            {"params": [p for p in nodecay_params], "weight_decay": 0.0},
        ]

        # Print parameter statistics for debugging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Num decay parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
        )
        print(
            f"Num nodecay parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
        )

        # Use fused AdamW if available on CUDA (faster training)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.startswith("cuda")
        print(f"Using fused AdamW: {use_fused}")

        # Create and return the optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        """
        Forward pass through the GPT model with optional KV caching.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target tokens for loss computation
            kv_cache: Optional KVCache object for efficient autoregressive generation
            loss_reduction (str): Loss reduction mode - 'mean', 'sum', or 'none'

        Returns:
            tuple: (logits, loss) where:
                - logits: Output logits of shape (batch_size, sequence_length, vocab_size)
                - loss: Cross-entropy loss if targets provided, None otherwise
        """
        # Get batch size and sequence length
        B, T = idx.size()

        # Ensure sequence length doesn't exceed model's maximum context length
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # =====================================================================
        # POSITION EMBEDDINGS with KV Cache Support
        # =====================================================================
        # When using KV cache, we need to offset positions by the cache position
        # Example: If cache has 10 tokens, new token gets position 10 (not 0)
        if kv_cache is not None:
            # Get current cache position (how many tokens already processed)
            cache_pos = kv_cache.get_pos()
            # Create position indices offset by cache position
            pos = torch.arange(
                cache_pos, cache_pos + T, dtype=torch.long, device=idx.device
            )
        else:
            # No cache: normal position indices starting from 0
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Get token embeddings: convert token indices to dense vectors
        tok_emb = self.transformer.wte(idx)  # Shape: (B, T, n_embed) # type: ignore

        # Get position embeddings: add positional information
        pos_emb = self.transformer.wpe(pos)  # Shape: (T, n_embed) # type: ignore

        # Combine token and position embeddings
        x = tok_emb + pos_emb  # Shape: (B, T, n_embed)

        # =====================================================================
        # TRANSFORMER BLOCKS with KV Cache
        # =====================================================================
        # Pass through all transformer blocks, propagating the KV cache
        # Each block will insert its K,V into the cache and retrieve full history
        for block in self.transformer.h:
            x = block(x, kv_cache=kv_cache)

        # Apply final layer normalization
        x = self.transformer.ln_f(x)

        # Project to vocabulary size to get logits for next token prediction
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss computation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction=loss_reduction,
            )  # shape depends on reduction: (1,) for mean/sum, (B*T,) for none

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load a pretrained GPT-2 model from Hugging Face transformers.

        This method creates a GPT model with the architecture matching the
        specified pretrained model and loads the pretrained weights.

        Args:
            model_type (str): Type of GPT-2 model ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')

        Returns:
            GPT: Model instance with pretrained weights loaded
        """
        # Only GPT-2 models are supported for pretraining
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from {model_type}")

        # Define architecture parameters for different GPT-2 model sizes
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=24, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=32, n_embed=1600),  # 1558M params
        }[model_type]

        # Set vocabulary size and context length (same for all GPT-2 models)
        config_args["vocab_size"] = 50257  # Standard GPT-2 vocabulary size
        config_args["block_size"] = 1024  # Standard GPT-2 context length

        # Create a new model instance with the appropriate configuration
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Get the state dict of our model
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Remove attention bias keys (not used in our implementation)
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Load the pretrained Hugging Face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Prepare Hugging Face state dict keys
        sd_keys_hf = sd_hf.keys()
        # Remove keys that don't exist in our model
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        # These weights need to be transposed because HuggingFace uses Conv1D instead of Linear
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # Ensure both models have the same number of parameters
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # Copy weights from HuggingFace model to our model
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose weights for Conv1D -> Linear conversion
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Copy weights directly
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print("All parameters initialized and match in size")
        return model


def generate(
    num_sequences,
    max_length,
    model,
    context,
    device,
    random_number_generator,
    use_kv_cache=True,
):
    """
    Generate text sequences using the trained GPT model with optional KV caching.

    This function performs autoregressive text generation using top-k sampling
    to produce diverse and coherent text continuations.

    KV Caching: When enabled (default), uses a two-phase approach:
    1. PREFILL: Process all context tokens at once, cache their K/V
    2. DECODE: Generate one token at a time, reusing cached K/V
    This reduces computation from O(N²) to O(N), making generation 3-10x faster.

    Args:
        num_sequences (int): Number of sequences to generate
        max_length (int): Maximum length of each generated sequence
        model (GPT): The trained GPT model
        context (str): Initial text context to start generation from
        device (str): Device to run generation on ('cuda', 'mps', or 'cpu')
        random_number_generator: PyTorch random generator for sampling
        use_kv_cache (bool): Whether to use KV caching (default: True)

    Returns:
        list: List of decoded text sequences
    """
    from gpt_2.kv_cache import KVCache

    # Initialize the custom tokenizer with special tokens for chat format
    enc, _ = get_custom_tokenizer()

    # Encode the context string to token indices (allow special tokens in context)
    tokens = enc.encode(context, allowed_special="all")
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Create multiple copies for batch generation
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens.to(device)

    # Set random seed for reproducible generation
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # =========================================================================
    # SETUP: Create KV cache if enabled
    # =========================================================================
    kv_cache = None
    if use_kv_cache:
        # Extract model dimensions for KV cache
        config = model.config
        num_heads = config.n_kv_head if hasattr(config, "n_kv_head") else config.n_head
        head_dim = config.n_embed // config.n_head
        num_layers = config.n_layer

        # =====================================================================
        # OPTIMIZATION: Use prefill pattern for batch generation
        # =====================================================================
        # For batch generation from the same prompt, we can:
        # 1. Process prompt ONCE with batch_size=1 (efficient single forward pass)
        # 2. Copy cached K/V to batch_size=num_sequences (cheap memory copy)
        # 3. Generate multiple sequences in parallel
        # This avoids recomputing the same prompt num_sequences times!

        if num_sequences > 1:
            # Step 1: Create single-batch cache and process prompt once
            single_cache = KVCache(
                batch_size=1,
                num_heads=num_heads,
                seq_len=max_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )

            # Process prompt once with batch_size=1
            prompt_tensor = x[0:1, :]  # Take first sequence (they're all identical)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits_single, _ = model(prompt_tensor, kv_cache=single_cache)

            # Step 2: Create batch cache and prefill from single cache
            kv_cache = KVCache(
                batch_size=num_sequences,
                num_heads=num_heads,
                seq_len=max_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )
            kv_cache.prefill(single_cache)  # Copy cached K/V to all batch positions

            # Replicate logits for all sequences
            logits = logits_single.repeat(num_sequences, 1, 1)  # (B, T, vocab_size)
        else:
            # Single sequence: no need for prefill optimization
            kv_cache = KVCache(
                batch_size=num_sequences,
                num_heads=num_heads,
                seq_len=max_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )

            # Process prompt tokens
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(x, kv_cache=kv_cache)  # Shape: (B, T, vocab_size)
    else:
        # =====================================================================
        # NO KV CACHE: Process entire sequence each time (O(N²))
        # =====================================================================
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(x, kv_cache=None)  # Shape: (B, T, vocab_size)

    # Only use the last token's predictions for next token
    logits = logits[:, -1, :]  # Shape: (B, vocab_size)

    # =========================================================================
    # PHASE 2: DECODE - Generate tokens one at a time
    # =========================================================================
    # Generate tokens autoregressively until max_length is reached
    while x.size(1) < max_length:
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # Shape: (B, vocab_size)

        # Apply top-k sampling (k=50) for diverse generation
        # This helps avoid repetitive text by sampling from top-k most likely tokens
        topk_probs, topk_indices = torch.topk(
            probs, 50, dim=-1
        )  # topk_probs: (B, 50), topk_indices: (B, 50)

        # Sample from the top-k distribution
        ix = torch.multinomial(
            topk_probs, num_samples=1, generator=random_number_generator
        )  # ix: (B, 1)

        # Get the actual token indices from the top-k indices
        xcol = torch.gather(topk_indices, -1, ix)  # Shape: (B, 1)

        # Append the new token to the sequence
        x = torch.cat((x, xcol), dim=1)

        # Get next token's logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if use_kv_cache:
                    # WITH CACHE: Only pass the new token (O(1) per step)
                    logits, _ = model(xcol, kv_cache=kv_cache)
                else:
                    # WITHOUT CACHE: Pass entire sequence (O(N) per step)
                    logits, _ = model(x, kv_cache=None)

        # Extract logits for next prediction
        logits = logits[:, -1, :]  # Shape: (B, vocab_size)

    # Decode and return all generated sequences
    all_decoded = []
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        all_decoded.append(decoded)
    return all_decoded


# class DataLoaderLite:
#     def __init__(self, B, T):

#         with open("data/input.txt", "r") as f:
#             text = f.read()

#         import tiktoken

#         self.enc = tiktoken.get_encoding("gpt2")
#         self.tokens = self.enc.encode(text)
#         self.tokens = torch.tensor(self.tokens, dtype=torch.long)
#         self.current_position = 0
#         self.B = B
#         self.T = T
#         print(f"Loaded {len(self.tokens)} tokens")
#         print(f"1 epoch has {len(self.tokens) // (self.B * self.T)} batches")

#     def next_batch(self):
#         buf = self.tokens[
#             self.current_position : self.current_position + self.B * self.T + 1
#         ]
#         x = buf[:-1].view(self.B, self.T)
#         y = buf[1:].view(self.B, self.T)
#         self.current_position += self.B * self.T
#         if self.current_position + (self.B * self.T + 1) > len(self.tokens):
#             self.current_position = 0
#         return x, y


# Test the model when script is run directly
if __name__ == "__main__":
    # Determine the best available device for computation
    device = "cpu"  # Default fallback
    if torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    else:
        device = "cpu"  # CPU fallback

    print(f"Using device: {device}")
