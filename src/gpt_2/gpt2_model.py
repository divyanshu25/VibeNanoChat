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

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt_2.block import Block
from gpt_2.rope import precompute_rotary_embeddings
from gpt_2.utils import has_value_embedding


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) with nanochat-inspired enhancements.

    Architecture Components:
    ------------------------
    - Token embeddings
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
    """

    def __init__(self, config, master_process=True, pad_vocab_size_to=64):
        """
        Initialize GPT model with specified configuration.

        Args:
            config (GPTConfig): Model hyperparameters
                - vocab_size: Vocabulary size
                - n_embed: Embedding dimension (model width)
                - n_layer: Number of transformer blocks (depth)
                - n_head: Number of attention heads per block
                - block_size: Maximum sequence length (context window)
            master_process (bool): Whether this is the master process (for printing)
            pad_vocab_size_to (int): Pad vocab size to multiple of this (for DDP efficiency)
        """
        super().__init__()
        self.config = config
        self.max_seq_len = config.block_size

        # ===== Vocab Padding (Nanochat-style) =====
        # Pad vocab for efficiency (DDP, tensor cores). Outputs are cropped in forward().
        # This ensures embedding dimensions are divisible by common world_sizes
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        # This is a classical padding trick to make vocab_size scale to next number that is divisible by pad_vocab_size_to.
        self.padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if master_process and self.padded_vocab_size != config.vocab_size:
            print(
                f"Padding vocab_size from {config.vocab_size} to {self.padded_vocab_size} for efficiency (DDP/distributed)"
            )

        # ===== Transformer Core Components =====
        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: vocab indices -> dense vectors (use padded vocab size)
                wte=nn.Embedding(self.padded_vocab_size, config.n_embed),
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
            device="cpu",  # Will be moved to GPU via .to(device) in trainer
            dtype=torch.bfloat16,
        )  # Shape: (1, seq_len, 1, head_dim//2)

        # Register as non-persistent buffers (recomputed on load, not saved)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # ===== Sliding Window Attention Setup (nanochat-style) =====
        # Compute per-layer window sizes from pattern string
        self.window_sizes = self._compute_window_sizes(config)
        if master_process:
            print(f"Sliding window pattern: '{config.window_pattern}'")
            for i, (left, right) in enumerate(self.window_sizes):
                window_type = f"Window=({left}, {right})"
                print(f"  Layer {i}: {window_type}")

        # ===== Value Embeddings (nanochat-style) =====
        # Value embeddings: alternating layers, last layer always included
        # These are lookup tables that add extra capacity to attention values
        head_dim = config.n_embed // config.n_head
        kv_dim = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        ) * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(self.padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_value_embedding(i, config.n_layer)
            }
        )
        if master_process:
            value_embed_layers = [
                i
                for i in range(config.n_layer)
                if has_value_embedding(i, config.n_layer)
            ]
            print(f"Value embeddings enabled at layers: {value_embed_layers}")

        # ===== Language Modeling Head =====
        # Use padded vocab size (will crop to actual vocab_size in forward)
        self.lm_head = nn.Linear(config.n_embed, self.padded_vocab_size, bias=False)

        # ===== Per-Layer Scalars (nanochat-style) =====
        # Learnable scalars that control information flow through residual stream
        # - resid_lambdas: Scale the residual stream entering each layer
        # - x0_lambdas: Blend in the original normalized embedding at each layer
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # ===== Weight Initialization =====
        self.apply(self._init_weights)

        # ===== Post-init: Initialize per-layer scalars =====
        # resid_lambdas: Start at 1.0 (neutral, no scaling)
        # x0_lambdas: Start at 0.1 (small contribution from original embedding)
        self.resid_lambdas.data.fill_(1.0)
        self.x0_lambdas.data.fill_(0.1)

        # ===== Post-init: Zero-initialize value embedding gates =====
        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if (
                hasattr(block.attn, "value_embed_gate")
                and block.attn.value_embed_gate is not None
            ):
                torch.nn.init.zeros_(block.attn.value_embed_gate.weight)

    def cast_embeddings_to_bfloat16(self):
        """
        Cast token and value embeddings to bfloat16 (nanochat-style).

        This saves memory and the optimizer can tolerate it. MUST be called
        after the model is moved to the target device (CUDA).

        Note: This is separate from _init_weights because it needs to run
        after .to(device) is called in the training script.
        """
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

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
            # But for value embeddings, use uniform init like c_v in nanochat
            if module in self.value_embeds.values():
                # Value embeddings: uniform init with same bound as other projections
                bound = (3.0**0.5) * (self.config.n_embed**-0.5)
                torch.nn.init.uniform_(module.weight, -bound, bound)
            else:
                # Token embeddings: normal init
                torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention (nanochat-style).

        Returns list of window_size values for attention_forward:
        - (left, right) tuple: Sliding window specification
          - left: how many tokens before current position to attend to
          - right: how many tokens after current position to attend to (0 for causal)
          - Use (block_size, 0) for full context

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)

        Examples:
            "L" → all layers use full context (all (block_size, 0))
            "SL" → alternating short/long windows
            "SSSL" → 3 short windows, 1 long window (nanochat default)
        """
        # Handle None or empty pattern - use "L" (full context for all layers)
        pattern = config.window_pattern
        if pattern is None or pattern == "":
            pattern = "L"

        pattern = pattern.upper()
        assert all(
            c in "SL" for c in pattern
        ), f"Invalid window_pattern: {pattern}. Use only S (short) and L (long)."

        # Map characters to window sizes
        # Use (left, 0) tuples consistently - left is lookback, 0 means causal
        short_window = config.block_size // 2  # Half context
        char_to_window = {
            "L": (config.block_size, 0),  # Full context (full lookback, causal)
            "S": (short_window, 0),  # Half context, causal (0 right)
        }

        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])

        # Final layer always gets full context (critical for loss computation)
        window_sizes[-1] = (config.block_size, 0)

        return window_sizes

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
        - Per-layer scalars (element-wise multiplications, not matmuls)
        - RoPE (precomputed, registered as buffers)
        - RMSNorm (functional, no learnable params)

        References:
        - https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        - https://arxiv.org/abs/2204.02311 (PaLM paper)

        Returns:
            int: Estimated FLOPs per token for training
        """
        param_counts = self.num_scaling_params()

        # Exclude non-matmul parameters: embeddings (table lookups) and scalars (element-wise ops)
        # Only count transformer matrices and lm_head (the actual matmul operations)
        nparams_exclude = (
            param_counts["wte"] + param_counts["value_embeds"] + param_counts["scalars"]
        )

        # Attention FLOPs calculation
        h = self.config.n_head  # Number of attention heads
        d = self.config.n_embed // h  # Dimension per head
        T = self.config.block_size  # Sequence length
        attn_flops = 12 * h * d * T * self.config.n_layer

        # Total: matmul FLOPs + attention FLOPs
        num_flops_per_token = 6 * (param_counts["total"] - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis (nanochat-style).

        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters

        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. scaling laws)

        Following nanochat's approach: transformer_matrices + lm_head gives the
        cleanest scaling laws (see nanochat dev/LOG.md Jan 27, 2026).

        Returns:
            dict: Parameter counts for each group:
                - wte: Token embeddings
                - value_embeds: Value embeddings (if used)
                - lm_head: Language model output head
                - transformer_matrices: All transformer block weight matrices
                - scalars: Per-layer scalars (resid_lambdas, x0_lambdas)
                - total: Sum of all parameters
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        # Note: Using nanochat's simple counting approach
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(
            p.numel() for p in self.parameters()
        ), "Parameter count mismatch"

        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def configure_optimizers(
        self,
        weight_decay,
        device,
        config=None,
        ddp=False,
        master_process=True,
        embedding_lr=None,
        unembedding_lr=None,
        matrix_lr=None,
        scalar_lr=None,
        adam_beta1=0.8,
        adam_beta2=0.95,
    ):
        """
        Configure optimizers with AdamW + Muon hybrid (nanochat-style).

        - AdamW: embeddings, output head, biases
        - Muon: transformer block weights (2D parameters)

        Args:
            weight_decay (float): Weight decay coefficient
            device (str): Device type ('cuda', 'mps', 'cpu')
            config (GPTConfig, optional): Unused, kept for API compatibility
            ddp (bool): Distributed training flag
            embedding_lr (float): Learning rate for embeddings (nanochat-style)
            unembedding_lr (float): Learning rate for output head (nanochat-style)
            matrix_lr (float): Learning rate for Muon matrix params (nanochat-style)
            scalar_lr (float): Learning rate for scalars (nanochat-style)
            adam_beta1 (float): Adam beta1 parameter
            adam_beta2 (float): Adam beta2 parameter

        Returns:
            List of optimizers [AdamW, Muon]
        """
        # ===== AdamW + Muon Hybrid (Nanochat-style) =====
        # This hybrid approach uses a SINGLE COMBINED optimizer with different
        # optimization strategies for different parameter types:
        # - Muon: For transformer weight matrices (where Newton-Schulz preconditioning shines)
        # - AdamW: For embeddings, output head, and 1D params (where adaptive LR is better)
        from gpt_2.muon import DistMuonAdamW, MuonAdamW

        # ========== STEP 1: Partition Parameters by Type ==========
        # Different parameter types have different optimization characteristics:
        #
        # MUON CANDIDATES (2D weight matrices in transformer blocks):
        # - Attention projections (c_attn fused QKV, c_proj output)
        # - MLP weight matrices (c_fc, c_proj)
        # - Value embedding gates (value_embed_gate): only some layers have these
        #   (alternating pattern based on has_value_embedding)
        # - These benefit from Newton-Schulz orthogonalization
        # - Typically the bulk of trainable parameters (~90%+)
        #
        # ADAMW CANDIDATES (everything else):
        # - Embeddings: High-variance sparse updates (only touched tokens get gradients)
        # - Output head: Similar to embeddings, needs adaptive per-parameter learning
        # - 1D params (biases): Too small for Newton-Schulz, use adaptive LR
        #   (Note: this model uses functional RMSNorm with no learnable params)

        matrix_params = []  # Transformer block 2D weights → Muon
        embedding_params = []  # Token/position embeddings → AdamW
        value_embeds_params = []  # Value embeddings → AdamW (nanochat-style)
        lm_head_params = []  # Output projection → AdamW
        resid_lambdas_params = []  # Per-layer resid scalars → AdamW
        x0_lambdas_params = []  # Per-layer x0 scalars → AdamW
        other_params = []  # Biases and other 1D params → AdamW

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "resid_lambdas" in name:
                # Per-layer residual scalars: conservative LR (nanochat-style)
                resid_lambdas_params.append(param)
            elif "x0_lambdas" in name:
                # Per-layer x0 scalars: full scalar LR with high momentum (nanochat-style)
                x0_lambdas_params.append(param)
            elif "value_embeds" in name:
                # Value embeddings: lookup table, use same LR as embeddings (nanochat-style)
                value_embeds_params.append(param)
            elif "transformer.h." in name and param.dim() == 2:
                # Transformer block weight matrices (e.g., attn.c_attn.weight, mlp.c_fc.weight)
                # Note: This includes value_embed_gate which goes to Muon (nanochat-style)
                # Only some layers have ve_gate (conditional creation based on alternating pattern)
                matrix_params.append(param)
            elif "wte" in name:
                # Word token embeddings (wte) — position embeddings are not used
                # (model uses RoPE inside attention).
                # Sparse updates (only active tokens) → needs adaptive per-embedding LR
                embedding_params.append(param)
            elif "lm_head" in name:
                # Language model head (final projection to vocabulary)
                # Similar gradient pattern to embeddings → AdamW works better
                lm_head_params.append(param)
            else:
                # Everything else: biases, LayerNorm weights/biases, etc.
                # 1D parameters → no benefit from Newton-Schulz, use AdamW
                other_params.append(param)

        # ========== STEP 2: Log Parameter Distribution ==========
        # Verify that parameters are correctly partitioned (useful for debugging)
        if master_process:
            print(
                f"Muon: {len(matrix_params)} matrix params, {sum(p.numel() for p in matrix_params):,} parameters"
            )
            print(
                f"AdamW: {len(embedding_params)} embedding params, {sum(p.numel() for p in embedding_params):,} parameters"
            )
            print(
                f"AdamW: {len(value_embeds_params)} value_embeds params, {sum(p.numel() for p in value_embeds_params):,} parameters"
            )
            print(
                f"AdamW: {len(lm_head_params)} lm_head params, {sum(p.numel() for p in lm_head_params):,} parameters"
            )
            print(
                f"AdamW: {len(resid_lambdas_params)} resid_lambdas params, {sum(p.numel() for p in resid_lambdas_params):,} parameters"
            )
            print(
                f"AdamW: {len(x0_lambdas_params)} x0_lambdas params, {sum(p.numel() for p in x0_lambdas_params):,} parameters"
            )
            print(
                f"AdamW: {len(other_params)} other params, {sum(p.numel() for p in other_params):,} parameters"
            )

        # ========== STEP 3: Configure Learning Rates ==========
        # Nanochat-style: Different LRs for different parameter types
        # This allows fine-grained control over optimization dynamics
        #
        # Nanochat defaults (aggressive, well-tuned):
        # - matrix_lr (Muon): 0.02 - Muon's preconditioning allows stable high LR
        # - embedding_lr: 0.3 - High LR for fast adaptation (sparse updates)
        # - unembedding_lr: 0.004 - Lower for output stability
        # - scalar_lr: 0.5 - Very high for biases/LayerNorm (quick adjustments)
        #
        # These are much higher than typical AdamW rates because:
        # - AdamW's adaptive per-parameter scaling provides built-in safety
        # - Muon's orthogonalization prevents instability at high LRs

        if unembedding_lr is None:
            raise ValueError("unembedding_lr must be specified")
        if embedding_lr is None:
            raise ValueError("embedding_lr must be specified")
        if scalar_lr is None:
            raise ValueError("scalar_lr must be specified")
        if matrix_lr is None:
            raise ValueError("matrix_lr must be specified")

        _unembedding_lr = unembedding_lr
        _embedding_lr = embedding_lr
        _matrix_lr = matrix_lr
        _scalar_lr = scalar_lr

        # ========== STEP 4: Build Combined Parameter Groups (Nanochat-style) ==========
        # Create unified parameter groups for the combined optimizer.
        # Each group specifies its 'kind' (adamw or muon) and gets appropriate
        # hyperparameters.
        #
        # Why unified param_groups?
        # - Single optimizer.step() call handles all parameters
        # - Simplified LR scheduling (single scheduler for all groups)
        # - Better integration with distributed training (no DDP wrapper needed)

        param_groups = [
            # ===== Per-Layer Scalars (nanochat-style) =====
            # resid_lambdas: Conservative LR (0.01x scalar_lr) for stable gradient flow
            {
                "params": resid_lambdas_params,
                "lr": _scalar_lr * 0.01,
                "initial_lr": _scalar_lr * 0.01,
                "betas": (adam_beta1, adam_beta2),  # Standard momentum
                "eps": 1e-10,
                "weight_decay": 0.0,
                "kind": "adamw",
            },
            # x0_lambdas: Full scalar LR with higher momentum for stability
            {
                "params": x0_lambdas_params,
                "lr": _scalar_lr,
                "initial_lr": _scalar_lr,
                "betas": (0.96, adam_beta2),  # Higher beta1 for stability
                "eps": 1e-10,
                "weight_decay": 0.0,
                "kind": "adamw",
            },
            # AdamW groups (embeddings, lm_head, value_embeds, scalars)
            {
                "params": lm_head_params,
                "lr": _unembedding_lr,
                "initial_lr": _unembedding_lr,
                "betas": (adam_beta1, adam_beta2),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "kind": "adamw",
            },
            {
                "params": embedding_params,
                "lr": _embedding_lr,
                "initial_lr": _embedding_lr,
                "betas": (adam_beta1, adam_beta2),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "kind": "adamw",
            },
            {
                "params": value_embeds_params,
                "lr": _embedding_lr,  # Same LR as embeddings (nanochat-style)
                "initial_lr": _embedding_lr,
                "betas": (adam_beta1, adam_beta2),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "kind": "adamw",
            },
            {
                "params": other_params,
                "lr": _scalar_lr,
                "initial_lr": _scalar_lr,
                "betas": (adam_beta1, adam_beta2),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "kind": "adamw",
            },
        ]

        # ========== STEP 5: Add Muon Groups (by shape for stacking efficiency) ==========
        # Group matrix parameters by shape so they can be stacked for efficient
        # batched processing in the Muon optimizer.
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "params": group_params,
                    "lr": _matrix_lr,
                    "initial_lr": _matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                    "kind": "muon",
                }
            )

        if master_process:
            print(
                f"Learning rates: unembedding={_unembedding_lr:.4f}, embedding={_embedding_lr:.4f}, matrix={_matrix_lr:.4f}, scalar={_scalar_lr:.4f}"
            )
            print(
                f"Using combined {'DistMuonAdamW' if ddp else 'MuonAdamW'} optimizer (nanochat-style)"
            )

        # ========== STEP 6: Create Combined Optimizer ==========
        # Use DistMuonAdamW for distributed training (handles gradient sync internally)
        # Use MuonAdamW for single GPU training
        #
        # Key difference from standard approach:
        # - NO DDP wrapper on model when using DistMuonAdamW
        # - Gradient synchronization happens inside optimizer.step()
        # - Enables better overlap of communication with computation
        OptimizerFactory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = OptimizerFactory(param_groups)

        return optimizer

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

        # ===== Embedding Normalization (nanochat-style) =====
        # Normalize embedding scale for stability (minor improvement)
        x = F.rms_norm(x, (x.size(-1),))

        # ===== Save x0 for Per-Layer Scalars =====
        # Store normalized embedding for skip connections at each layer
        x0 = x  # (B, T, n_embed)

        # ===== Transformer Blocks =====
        # Each block applies: attention (with RoPE) + feedforward
        # KV cache updated incrementally if provided
        # Pass layer-specific window size for sliding window attention
        for i, block in enumerate(self.transformer.h):
            # ===== Per-Layer Scalars: Pre-layer mixing =====
            # Apply learnable scalars BEFORE the block processes input
            # - resid_lambdas[i]: scale the current residual stream
            # - x0_lambdas[i]: blend in the original normalized embedding
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Look up value embedding if this layer has one
            value_embed = (
                self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            )
            # Block processes with its own internal residual connections
            # Internally: x = x + attn(norm(x)) and x = x + mlp(norm(x))
            x = block(
                x,
                value_embed=value_embed,
                cos_sin=cos_sin,
                kv_cache=kv_cache,
                window_size=self.window_sizes[i],
            )

        # ===== Final Normalization =====
        x = F.rms_norm(x, (x.size(-1),))  # Functional RMSNorm (nanochat-style)

        # ===== Language Modeling Head =====
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        # Crop to actual vocab_size (remove padding)
        logits = logits[..., : self.config.vocab_size]  # (B, T, vocab_size)

        # ===== Logit Soft Capping (nanochat-style) =====
        # Smoothly cap logits to range [-softcap, softcap] using tanh
        # This prevents extreme logit values that can cause training instability
        logits = logits.float()  # Switch to fp32 for numerical stability
        softcap = self.config.logit_softcap
        logits = softcap * torch.tanh(logits / softcap)  # Apply soft cap

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
