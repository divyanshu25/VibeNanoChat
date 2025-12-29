# Add gpt_2 to python path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt_2.block import Block
import tiktoken
import inspect
import time


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    This dataclass holds all the configuration parameters needed to define
    the architecture and training setup of the GPT model.
    """

    block_size: int = 1024  # Maximum sequence length (context window)
    vocab_size: int = 50257  # Size of the vocabulary (number of unique tokens)
    n_layer: int = 12  # Number of transformer blocks in the model
    n_head: int = 12  # Number of attention heads per transformer block
    n_embed: int = 768  # Embedding dimension (hidden size)
    batch_size: int = 16  # Training batch size
    total_batch_size: int = 524288  # 2^19

    # Intent discovery parameters
    n_intents: int = 16  # Number of discrete intents to learn (codebook size)
    use_intent: bool = True  # Whether to use intent embeddings
    intent_temperature: float = 1.0  # Temperature for intent sampling during training


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

        # Define the main transformer components
        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: convert token indices to dense vectors (WHAT to say)
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                # Position embeddings: add positional information to tokens (WHEN to say it)
                wpe=nn.Embedding(config.block_size, config.n_embed),
                # Stack of transformer blocks (the core of the model)
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final layer normalization before output
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

        # Intent embeddings: learnable intents that capture WHY to say something
        # This is a codebook of discrete intents that the model learns to use
        if config.use_intent:
            self.intent_codebook = nn.Embedding(config.n_intents, config.n_embed)
            # Intent predictor: learns to predict which intent to use based on context
            # This allows unsupervised discovery of intents from the data
            self.intent_predictor = nn.Sequential(
                nn.Linear(config.n_embed, config.n_embed // 2),
                nn.GELU(),
                nn.Linear(config.n_embed // 2, config.n_intents),
            )

        # Language modeling head: projects hidden states to vocabulary logits
        # Note: We'll tie the weights with the embedding layer
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight Tying scheme: tie input and output embeddings
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

    def forward(self, idx, targets=None, intent_idx=None):
        """
        Forward pass through the GPT model.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target tokens for loss computation
            intent_idx (torch.Tensor, optional): Intent indices of shape (batch_size,)
                If None and use_intent=True, intents are predicted from the input

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

        # Create position indices for the sequence
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape (T,)

        # Get token embeddings: convert token indices to dense vectors (WHAT to say)
        tok_emb = self.transformer.wte(idx)  # Shape: (B, T, n_embed) # type: ignore

        # Get position embeddings: add positional information (WHEN to say it)
        pos_emb = self.transformer.wpe(pos)  # Shape: (T, n_embed) # type: ignore

        # Discover and apply intent embeddings (WHY to say it)
        if self.config.use_intent:
            if intent_idx is None:
                # Unsupervised intent discovery: predict intent from initial embeddings
                # Use the mean of token embeddings as a sequence representation
                seq_repr = tok_emb.mean(dim=1)  # Shape: (B, n_embed)

                # Predict intent logits
                intent_logits = self.intent_predictor(seq_repr)  # Shape: (B, n_intents)

                if self.training:
                    # During training: sample from the distribution (Gumbel-Softmax for differentiability)
                    intent_probs = F.gumbel_softmax(
                        intent_logits, tau=self.config.intent_temperature, hard=True
                    )  # Shape: (B, n_intents)
                    # Get intent embedding using soft assignment
                    intent_emb = torch.matmul(
                        intent_probs, self.intent_codebook.weight
                    )  # Shape: (B, n_embed)
                else:
                    # During inference: use argmax (deterministic)
                    intent_idx = torch.argmax(intent_logits, dim=-1)  # Shape: (B,)
                    intent_emb = self.intent_codebook(intent_idx)  # Shape: (B, n_embed)
            else:
                # Use provided intent indices
                intent_emb = self.intent_codebook(intent_idx)  # Shape: (B, n_embed)

            # Broadcast intent across all positions in the sequence
            intent_emb = intent_emb.unsqueeze(1)  # Shape: (B, 1, n_embed)

            # Combine all three dimensions: WHAT + WHEN + WHY
            x = tok_emb + pos_emb + intent_emb  # Shape: (B, T, n_embed)
        else:
            # Original behavior: only WHAT + WHEN
            x = tok_emb + pos_emb  # Shape: (B, T, n_embed)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Apply final layer normalization
        x = self.transformer.ln_f(x)

        # Project to vocabulary size to get logits for next token prediction
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss computation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

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

        print(f"All parameters initialized and match in size")
        return model

    def get_intent_distribution(self, text_samples, device="cpu"):
        """
        Analyze which intents the model assigns to given text samples.
        This helps understand what the model has learned about "why" to say things.

        Args:
            text_samples (list of str): Text samples to analyze
            device (str): Device to run on

        Returns:
            dict: Intent statistics including distribution and per-sample assignments
        """
        if not self.config.use_intent:
            return {"error": "Intent embeddings not enabled"}

        enc = tiktoken.get_encoding("gpt2")
        self.eval()

        intent_assignments = []
        intent_logits_list = []

        with torch.no_grad():
            for text in text_samples:
                tokens = enc.encode(text)
                tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

                # Get token embeddings and predict intent
                tok_emb = self.transformer.wte(tokens)
                seq_repr = tok_emb.mean(dim=1)
                intent_logits = self.intent_predictor(seq_repr)
                intent_idx = torch.argmax(intent_logits, dim=-1).item()

                intent_assignments.append(intent_idx)
                intent_logits_list.append(intent_logits.cpu().numpy())

        # Calculate distribution
        from collections import Counter

        intent_counts = Counter(intent_assignments)

        return {
            "assignments": intent_assignments,
            "distribution": dict(intent_counts),
            "logits": intent_logits_list,
            "samples": text_samples,
        }


def generate(
    num_sequences,
    max_length,
    model,
    context,
    device,
    random_number_generator=None,
    intent_idx=None,
):
    """
    Generate text sequences using the trained GPT model.

    This function performs autoregressive text generation using top-k sampling
    to produce diverse and coherent text continuations.

    Args:
        num_sequences (int): Number of sequences to generate
        max_length (int): Maximum length of each generated sequence
        model (GPT): The trained GPT model
        context (str): Initial text context to start generation from
        device (str): Device to run generation on ('cuda', 'mps', or 'cpu')
        intent_idx (int or torch.Tensor, optional): Intent index(es) to use for generation.
            If None, intents are predicted automatically.
            If int, same intent used for all sequences.
            If tensor of shape (num_sequences,), different intent per sequence.
    """
    # Initialize the tokenizer (GPT-2 uses byte-pair encoding)
    enc = tiktoken.get_encoding("gpt2")

    # Encode the context string to token indices
    tokens = enc.encode(context)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Create multiple copies for batch generation
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens.to(device)

    # Prepare intent indices if provided
    if intent_idx is not None:
        if isinstance(intent_idx, int):
            # Single intent for all sequences
            intent_indices = torch.full(
                (num_sequences,), intent_idx, dtype=torch.long, device=device
            )
        else:
            # Different intent per sequence
            intent_indices = intent_idx.to(device)
    else:
        intent_indices = None

    # Set random seed for reproducible generation
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # Generate tokens autoregressively until max_length is reached
    while x.size(1) < max_length:
        with torch.no_grad():
            # Get model predictions for next token
            logits, _ = model(x, intent_idx=intent_indices)  # Shape: (B, T, vocab_size)

            # Only use the last token's predictions for next token
            logits = logits[:, -1, :]  # Shape: (B, vocab_size)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # Shape: (B, vocab_size)

            # Apply top-k sampling (k=50) for diverse generation
            # This helps avoid repetitive text by sampling from top-k most likely tokens
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            # Sample from the top-k distribution
            ix = torch.multinomial(
                topk_probs,
                num_samples=1,
                # generator=random_number_generator
            )  # Shape: (B, 1)

            # Get the actual token indices from the top-k indices
            xcol = torch.gather(topk_indices, -1, ix)  # Shape: (B, 1)

            # Append the new token to the sequence
            x = torch.cat((x, xcol), dim=1)

    # Decode and print all generated sequences
    all_decoded = []
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        all_decoded.append(decoded)
    return all_decoded


class DataLoaderLite:
    def __init__(self, B, T):
        with open("data/input.txt", "r") as f:
            text = f.read()

        self.B = B
        self.T = T
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (self.B*self.T)} batches")
        self.current_position = 0

    def next_batch(self):
        buf = self.tokens[
            self.current_position : self.current_position + self.B * self.T + 1
        ]
        x = buf[:-1].view(self.B, self.T).to(device)
        y = buf[1:].view(self.B, self.T).to(device)
        self.current_position += self.B * self.T
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


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

    # Create and initialize the model

    # model = GPT.from_pretrained("gpt2")
    model = GPT(GPTConfig())
    print("Model loaded")
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    model.to(device)

    dataloader = DataLoaderLite(B=4, T=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(2640):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Step {i} | Loss: {loss.item()}")

    # context = "Hello, I'm a language model,"
    # start_time = time.time()
    # decoded = generate(
    #     num_sequences=5,  # Generate 3 different sequences
    #     max_length=50,  # Each sequence up to 30 tokens
    #     model=model,
    #     context=context,
    #     device=device,
    #     # random_number_generator=torch.Generator(device=device),
    # )
    # for decoded_seq in decoded:
    #     print(">", decoded_seq)
    #     print("-"*100)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
