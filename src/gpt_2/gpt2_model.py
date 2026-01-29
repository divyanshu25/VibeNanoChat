# ===================================================================
# GPT-2 Model Implementation with Modern Enhancements
# ===================================================================
# This module implements a GPT-2 style transformer with nanochat-inspired
# improvements including RoPE, functional RMSNorm, and KV caching.

import os
import sys

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt_2.block import Block
from gpt_2.rope import precompute_rotary_embeddings


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) with nanochat-inspired enhancements.

    Architecture Components:
    ------------------------
    - Token embeddings (optional weight tying with output head)
    - RoPE (Rotary Position Embeddings) for positional encoding
    - Stack of transformer blocks (attention + feedforward)
    - Functional RMSNorm (no learnable parameters)
    - Language modeling head for next-token prediction
    - KV cache support for efficient autoregressive generation
    - Dual optimizer support (AdamW-only or AdamW+Muon hybrid)

    Key Design Choices:
    -------------------
    - RoPE instead of learned position embeddings
    - Functional RMSNorm instead of LayerNorm (no params)
    - Zero-initialized residual projections (pure skip connections at init)
    - Embedding weight tying enabled by default
    """

    def __init__(self, config, master_process=True):
        """
        Initialize GPT model with specified configuration.

        Args:
            config (GPTConfig): Model hyperparameters
                - vocab_size: Vocabulary size
                - n_embed: Embedding dimension (model width)
                - n_layer: Number of transformer blocks (depth)
                - n_head: Number of attention heads per block
                - block_size: Maximum sequence length (context window)
                - tie_embeddings: Share input/output embedding weights (default: True)
            master_process (bool): Whether this is the master process (for printing)
        """
        super().__init__()
        self.config = config
        self.max_seq_len = config.block_size

        # ===== Transformer Core Components =====
        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: vocab indices -> dense vectors
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                # NOTE: No learned position embeddings (using RoPE instead)
                # Transformer blocks: self-attention + feedforward
                # Each block has a layer_idx for proper KV cache coordination
                h=nn.ModuleList(
                    [Block(config, layer_idx=i) for i in range(config.n_layer)]
                ),
                # NOTE: Final norm is functional RMSNorm (no learnable params)
            )
        )

        # ===== Rotary Position Embeddings (RoPE) Setup =====
        # Precompute cos/sin buffers for positional encoding
        # Over-allocate by 10x to support longer sequences during inference
        self.rotary_seq_len = config.block_size * 10
        head_dim = config.n_embed // config.n_head
        cos, sin = precompute_rotary_embeddings(
            self.rotary_seq_len,
            head_dim,
            base=10000,
            device=None,  # Device assignment happens in _init_weights
            dtype=torch.bfloat16,
        )  # Shape: (1, seq_len, 1, head_dim//2)

        # Register as non-persistent buffers (recomputed on load, not saved)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # ===== Language Modeling Head =====
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # ===== Weight Tying (Optional) =====
        # Share weights between input embeddings and output projection
        # Reduces parameters and often improves performance
        if getattr(config, "tie_embeddings", True):
            self.transformer.wte.weight = self.lm_head.weight
            if master_process:
                print("Weight tying: ENABLED (wte <-> lm_head share weights)")
        else:
            if master_process:
                print(
                    "Weight tying: DISABLED (wte and lm_head have independent weights)"
                )

        # ===== Weight Initialization =====
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights (nanochat-style).

        Initialization Strategy:
        ------------------------
        - Input projections: Uniform init, bound = √3/√n_embd (avoids outliers)
        - Residual projections: Zero init (pure skip connections at start)
        - Token embeddings: Normal init, std = 1.0 (rich initial representations)
        - Output head: Normal init, std = 0.001 (stable near-uniform logits)
        - RoPE buffers: Recomputed on correct device after module init

        Args:
            module: PyTorch module to initialize
                    When module is self, RoPE buffers are moved to correct device
        """
        # ===== Linear Layer Initialization =====
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # Zero-init residual projections (c_proj in attention/MLP blocks)
                # Makes residual stream start as pure skip connections for stable training
                torch.nn.init.zeros_(module.weight)

            elif module is self.lm_head:
                # Small-scale init for language model head
                # Produces stable, near-uniform logits at initialization
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)

            else:
                # Width-aware uniform init for all other linear layers
                # bound = √3/√n_embd maintains unit variance while avoiding outliers
                bound = (3.0**0.5) * (self.config.n_embed**-0.5)
                torch.nn.init.uniform_(module.weight, -bound, bound)

            # Always zero-initialize biases
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # ===== Embedding Layer Initialization =====
        elif isinstance(module, nn.Embedding):
            # Use std=1.0 for richer initial token representations (nanochat-style)
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

        # ===== RoPE Buffer Recomputation =====
        # After all modules are initialized, move RoPE buffers to correct device
        if hasattr(self, "cos") and module is self:
            head_dim = self.config.n_embed // self.config.n_head
            device = next(self.parameters()).device
            cos, sin = precompute_rotary_embeddings(
                self.rotary_seq_len,
                head_dim,
                base=10000,
                device=device,
                dtype=torch.bfloat16,
            )
            self.cos.copy_(cos)
            self.sin.copy_(sin)

    def estimate_flops(self):
        """
        Estimate FLOPs per token for training (forward + backward pass).

        Used for scaling law experiments and compute-optimal training calculations.
        Follows PaLM/nanochat methodology.

        FLOP Accounting:
        ----------------
        1. Matrix Multiplication: 6 FLOPs per parameter
           - Forward: 2 FLOPs (multiply + accumulate)
           - Backward: 4 FLOPs (2x forward for gradients)

        2. Attention Operations: 12 * n_head * head_dim * seq_len * n_layer
           - QK^T matmul: 2 * head_dim * seq_len (per head)
           - softmax(QK^T) @ V: 2 * head_dim * seq_len (per head)
           - Both forward & backward: 2x multiplier
           - Total: 12 * h * d * T per layer

        Exclusions:
        -----------
        - Embedding lookups (table lookups, not matmuls)
        - RoPE (precomputed, registered as buffers)
        - RMSNorm (functional, no learnable params)

        References:
        - https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        - https://arxiv.org/abs/2204.02311 (PaLM paper)

        Returns:
            int: Estimated FLOPs per token for training
        """
        nparams = self.num_scaling_params()

        # Exclude embedding parameters if weight tying is enabled
        # (embeddings are lookups, not matrix multiplications)
        nparams_exclude = 0
        if getattr(self.config, "tie_embeddings", False):
            nparams_exclude = self.transformer.wte.weight.numel()

        # Attention FLOPs calculation
        h = self.config.n_head  # Number of attention heads
        d = self.config.n_embed // h  # Dimension per head
        T = self.config.block_size  # Sequence length
        attn_flops = 12 * h * d * T * self.config.n_layer

        # Total: matmul FLOPs + attention FLOPs
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Count total trainable parameters for scaling law experiments.

        Used in Chinchilla-style scaling law studies to determine compute-optimal
        model sizes and training configurations. Counts all parameters including
        embeddings, transformer blocks, and output heads.

        Reference:
        - https://arxiv.org/abs/2203.15556 (Chinchilla: Training Compute-Optimal LLMs)

        Returns:
            int: Total trainable parameter count
        """
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(
        self,
        learning_rate,
        weight_decay,
        device,
        config=None,
        use_muon=False,
        muon_lr=0.02,
        ddp=False,
        master_process=True,
    ):
        """
        Configure optimizer(s) with proper parameter grouping and weight decay.

        Two optimization strategies:
        1. AdamW-only (use_muon=False): Standard approach with weight decay grouping
        2. AdamW + Muon hybrid (use_muon=True): Nanochat-style dual optimizer
           - AdamW: embeddings, output head, biases
           - Muon: transformer block weights (2D parameters)

        Args:
            learning_rate (float): Learning rate for AdamW
            weight_decay (float): Weight decay coefficient
            device (str): Device type ('cuda', 'mps', 'cpu')
            config (GPTConfig, optional): Unused, kept for API compatibility
            use_muon (bool): Enable hybrid AdamW+Muon optimization
            muon_lr (float): Learning rate for Muon optimizer (if enabled)
            ddp (bool): Distributed training flag

        Returns:
            Single optimizer (AdamW) or list of optimizers [AdamW, Muon]
        """
        # Get all trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # ===== Strategy 1: AdamW-Only (Standard) =====
        if not use_muon:
            # Group parameters by dimensionality
            # 2D+ parameters (weight matrices) → weight decay
            # 1D parameters (biases, norms) → no weight decay
            decay_params = [p for p in param_dict.values() if p.dim() >= 2]
            nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]

            # Log parameter counts
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            if master_process:
                print(
                    f"Num decay parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
                )
                print(
                    f"Num nodecay parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
                )

            # Use fused kernel if available (faster on CUDA)
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device.startswith("cuda")
            if master_process:
                print(f"Using fused AdamW: {use_fused}")

            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused,
            )

            return optimizer

        # ===== Strategy 2: AdamW + Muon Hybrid (Nanochat-style) =====
        else:
            from gpt_2.muon import DistMuon, Muon

            # Partition parameters by location and type
            matrix_params = []  # Transformer block 2D weights → Muon
            embedding_params = []  # Token/position embeddings → AdamW
            lm_head_params = []  # Output projection → AdamW
            other_params = []  # Biases and other 1D params → AdamW

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue

                if "transformer.h." in name and param.dim() == 2:
                    # Transformer block weight matrices → Muon
                    matrix_params.append(param)
                elif "wte" in name or "wpe" in name:
                    # Embedding layers → AdamW
                    embedding_params.append(param)
                elif "lm_head" in name:
                    # Language model head → AdamW
                    lm_head_params.append(param)
                else:
                    # Everything else (biases, etc.) → AdamW
                    other_params.append(param)

            # Log parameter distribution
            if master_process:
                print(
                    f"Muon: {len(matrix_params)} matrix params, {sum(p.numel() for p in matrix_params):,} parameters"
                )
                print(
                    f"AdamW: {len(embedding_params)} embedding params, {sum(p.numel() for p in embedding_params):,} parameters"
                )
                print(
                    f"AdamW: {len(lm_head_params)} lm_head params, {sum(p.numel() for p in lm_head_params):,} parameters"
                )
                print(
                    f"AdamW: {len(other_params)} other params, {sum(p.numel() for p in other_params):,} parameters"
                )

            # Configure AdamW for embeddings, output head, and 1D params
            # Note: weight decay handled by Muon for transformer blocks
            adam_groups = [
                {"params": lm_head_params, "lr": learning_rate},
                {"params": embedding_params, "lr": learning_rate},
                {"params": other_params, "lr": learning_rate, "weight_decay": 0.0},
            ]

            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device.startswith("cuda") and not ddp
            if master_process:
                print(f"Using fused AdamW: {use_fused}")

            adamw_optimizer = torch.optim.AdamW(
                adam_groups,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused,
            )

            # Configure Muon for transformer block weight matrices
            MuonFactory = DistMuon if ddp else Muon
            muon_optimizer = MuonFactory(
                matrix_params,
                lr=muon_lr,
                momentum=0.95,
                weight_decay=weight_decay,  # Muon handles weight decay
            )

            # Store initial learning rates for scheduler compatibility
            for opt in [adamw_optimizer, muon_optimizer]:
                for group in opt.param_groups:
                    group["initial_lr"] = group["lr"]

            return [adamw_optimizer, muon_optimizer]

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        """
        Forward pass through GPT model with RoPE and optional KV caching.

        Pipeline:
        ---------
        1. Token embedding lookup
        2. RoPE setup (position-dependent rotations for Q,K)
        3. Transformer blocks (attention + feedforward)
        4. Final RMSNorm (functional, no learnable params)
        5. Language modeling head → logits
        6. Loss computation (if targets provided)

        Args:
            idx (torch.Tensor): Input token indices
                Shape: (batch_size, sequence_length)

            targets (torch.Tensor, optional): Target tokens for supervised learning
                Shape: (batch_size, sequence_length)
                Use -1 to ignore specific positions (e.g., padding)

            kv_cache (KVCache, optional): Cache for autoregressive generation
                Enables efficient incremental decoding

            loss_reduction (str): Cross-entropy loss reduction mode
                - 'mean': Average over non-ignored tokens (default)
                - 'sum': Sum over non-ignored tokens
                - 'none': Per-token losses (B*T,)

        Returns:
            tuple: (logits, loss)
                - logits: (B, T, vocab_size) - Next token predictions
                - loss: Scalar (mean/sum) or (B*T,) (none), None if no targets
        """
        B, T = idx.size()

        # ===== RoPE Setup with KV Cache =====
        # Determine starting position (0 for fresh, cache_pos for continuation)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()

        # Validate we don't exceed precomputed RoPE buffer
        # This is the true limit: rotary_seq_len = block_size * 10
        assert (
            T0 + T <= self.rotary_seq_len
        ), f"Position overflow: cache_pos ({T0}) + seq_len ({T}) = {T0 + T} exceeds rotary_seq_len ({self.rotary_seq_len})"

        # Slice cos/sin for current position range
        # RoPE applies these rotations to Q,K in each attention layer
        cos_sin = (
            self.cos[:, T0 : T0 + T, :, :],  # (1, T, 1, head_dim//2)
            self.sin[:, T0 : T0 + T, :, :],  # (1, T, 1, head_dim//2)
        )

        # ===== Token Embeddings =====
        x = self.transformer.wte(idx)  # (B, T, n_embed) # type: ignore
        # NOTE: No learned position embeddings (RoPE handles position)

        # ===== Transformer Blocks =====
        # Each block applies: attention (with RoPE) + feedforward
        # KV cache updated incrementally if provided
        for block in self.transformer.h:
            x = block(x, cos_sin=cos_sin, kv_cache=kv_cache)

        # ===== Final Normalization =====
        x = F.rms_norm(x, (x.size(-1),))  # Functional RMSNorm (nanochat-style)

        # ===== Language Modeling Head =====
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # ===== Loss Computation =====
        loss = None
        if targets is not None:
            # Cross-entropy loss with ignore_index=-1 for masked positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),  # (B*T,)
                ignore_index=-1,
                reduction=loss_reduction,
            )

        return logits, loss
