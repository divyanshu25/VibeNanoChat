---
title: Model
tags: [model, architecture, gpt, transformer]
---

# Model

> [!ABSTRACT]
> GPT-2 style decoder-only transformer with nanochat-inspired enhancements. Source: `src/gpt_2/`. For the *why* behind design choices, see [[architecture#Key design decisions]].

---

## Contents

- [[#Overview]]
- [[#Configuration]]
- [[#Forward pass]]
- [[#Transformer block]]
- [[#Sliding window attention]]
- [[#RoPE]]
- [[#Value embeddings]]
- [[#Per-layer scalars]]
- [[#Weight initialization]]
- [[#Parameter groups]]
- [[#FLOPs estimation]]

---

## Overview

| Feature | Original GPT-2 | VibeNanoChat |
|---------|---------------|-------------|
| Positional encoding | Learned absolute | RoPE (no learnable params) |
| Normalization | LayerNorm (learnable) | Functional RMSNorm (no params) |
| Attention pattern | Full context, all layers | Sliding window, configurable per-layer |
| Residual stream | Standard | Per-layer scalars (`resid_lambdas`, `x0_lambdas`) |
| Value representation | Standard V projection | Value embeddings (lookup table supplement) |
| Logit stability | None | Soft capping via `tanh` |

---

## Configuration

All hyperparameters live in `GPTConfig` (`src/gpt_2/config.py`).

### Depth parameterization (preferred)

Set `depth` and the rest is derived automatically:

```python
n_layer = depth
n_embed = round_up(depth × aspect_ratio, head_dim)
n_head  = n_embed // head_dim
n_kv_head = n_head  # MHA by default
```

| `depth` | Layers | Embed dim | Heads | Approx params |
|---------|--------|----------|-------|--------------|
| 6 | 6 | 384 | 6 | ~44M |
| 8 | 8 | 512 | 8 | ~77M |
| 12 | 12 | 768 | 12 | ~154M |
| 16 | 16 | 1024 | 16 | ~270M |
| 20 | 20 | 1280 | 20 | ~522M |

Defaults: `aspect_ratio = 64`, `head_dim = 64`

### Key config defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `block_size` | 2048 | Context window (max sequence length) |
| `vocab_size` | 50,266 | GPT-2 tokenizer + 9 special tokens |
| `window_pattern` | `"SSSL"` | 3 short-window layers, 1 full-context (tiled) |
| `logit_softcap` | 15.0 | Tanh cap range: `[−15, 15]` |

> [!TIP] Traditional config
> You can bypass depth parameterization by setting `depth = -1` and specifying `n_layer`, `n_embed`, `n_head` directly.

---

## Forward pass

```
Input tokens  (B, T)
      │
      ▼
Token embedding  wte  →  (B, T, n_embed)
      │
      ▼
RMSNorm  (functional, no params)
      │
      ▼
Save x0  ← used for per-layer skip connections
      │
      ▼
┌──────────────────────────────────────────┐
│  For each layer i = 0 … n_layer−1:       │
│                                          │
│  x = resid_lambdas[i] * x                │
│    + x0_lambdas[i] * x0                  │
│                                          │
│  value_embed = value_embeds[i](tokens)   │  (if layer has value embed)
│                                          │
│  x = TransformerBlock(x, value_embed,    │
│        cos_sin, kv_cache, window_size)   │
└──────────────────────────────────────────┘
      │
      ▼
RMSNorm  (functional)
      │
      ▼
lm_head  Linear(n_embed → padded_vocab_size)
      │
      ▼
Crop to vocab_size  →  (B, T, vocab_size)
      │
      ▼
Logit soft cap:  softcap * tanh(logits / softcap)
      │
      ▼
Cross-entropy loss  (if targets provided)
```

---

## Transformer block

Each block (`src/gpt_2/block.py`) uses the standard pre-norm residual pattern:

```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

### Attention (`src/gpt_2/attention.py`)

- Multi-head attention with optional GQA via `n_kv_head`
- RoPE applied to Q and K before the dot product
- Sliding window mask applied per-layer based on `window_pattern`
- Flash Attention 3 kernel used when available (H100)
- KV cache support for autoregressive generation

### MLP (`src/gpt_2/mlp.py`)

Standard two-layer feed-forward with GELU:

```
x → Linear(n_embed → 4×n_embed) → GELU → Linear(4×n_embed → n_embed)
```

---

## Sliding window attention

Controlled by `window_pattern` in `GPTConfig`. The pattern string is tiled across all layers. The **final layer always gets full context** regardless of pattern.

| Character | Window size | Use |
|-----------|------------|-----|
| `L` | `block_size` (full) | Long-range dependencies |
| `S` | `block_size // 2` | Local context, memory-efficient |

Default `"SSSL"` on a 12-layer model:

```
Layer  0: S    Layer  4: S    Layer  8: S
Layer  1: S    Layer  5: S    Layer  9: S
Layer  2: S    Layer  6: S    Layer 10: S
Layer  3: L    Layer  7: L    Layer 11: L  ← always full context
```

> [!NOTE]
> The final layer is forced to full context because it computes the logits used for loss. Restricting it would prevent the model from attending to relevant tokens at the end of a sequence.

---

## RoPE

Source: `src/gpt_2/rope.py`

RoPE encodes position by rotating Q and K vectors. Key properties:
- No learnable parameters
- Relative position is preserved in the dot product
- Generalizes to longer sequences than seen during training
- Buffers precomputed at `block_size × 10` to support extended inference

The cos/sin buffers are **non-persistent** — not saved in checkpoints, recomputed on load.

---

## Value embeddings

Certain layers receive an additional **value embedding**: a learned lookup table indexed by input tokens, added to the standard V projection.

- Enabled on alternating layers (every other layer, plus the final layer)
- Implemented as `nn.Embedding(vocab_size, kv_dim)` per enabled layer
- Stored in `self.value_embeds` as a `ModuleDict` keyed by layer index
- Optimized with AdamW at the same LR as token embeddings

This adds capacity to attention values without increasing attention computation cost.

---

## Per-layer scalars

Two sets of learnable scalars control information flow through the residual stream:

| Parameter | Shape | Init | Role |
|-----------|-------|------|------|
| `resid_lambdas` | `(n_layer,)` | 1.0 | Scales the residual stream entering each layer |
| `x0_lambdas` | `(n_layer,)` | 0.1 | Blends in the original normalized embedding |

Applied before each block:
```python
x = resid_lambdas[i] * x + x0_lambdas[i] * x0
```

`resid_lambdas` uses a very conservative LR (`scalar_lr × 0.01`). `x0_lambdas` uses the full `scalar_lr` with higher momentum (`beta1 = 0.96`).

---

## Weight initialization

| Parameter type | Strategy | Rationale |
|---------------|---------|-----------|
| Input projections | Uniform `±√3/√n_embed` | Maintains unit variance, avoids outliers |
| Residual projections (`c_proj`) | Zero init | Pure skip connections at start — stable training |
| Token embeddings | Normal `std=1.0` | Rich initial representations |
| Value embeddings | Uniform `±√3/√n_embed` | Consistent with other projections |
| LM head | Normal `std=0.001` | Near-uniform logits at initialization |
| Value embedding gates | Zero init | Gates start at `sigmoid(0) × 2 = 1.0` (neutral) |

---

## Parameter groups

| Group | Optimizer | Default LR | Parameters |
|-------|----------|-----------|-----------|
| `matrix_params` | Muon | 0.02 | All 2D weight matrices in transformer blocks |
| `embedding_params` | AdamW | 0.3 | Token embeddings (`wte`) |
| `value_embeds_params` | AdamW | 0.3 | Value embedding tables |
| `lm_head_params` | AdamW | 0.004 | Language model head |
| `resid_lambdas_params` | AdamW | `scalar_lr × 0.01` | Per-layer residual scalars |
| `x0_lambdas_params` | AdamW | 0.5 | Per-layer x0 scalars |
| `other_params` | AdamW | 0.5 | Biases, 1D params |

---

## FLOPs estimation

`model.estimate_flops()` follows the `C = 6ND` convention:

- **6 FLOPs per parameter** (2 forward + 4 backward)
- **Attention FLOPs:** `12 × n_head × head_dim × block_size × n_layer`
- **Excluded:** embedding lookups, per-layer scalars, RoPE buffers, RMSNorm

See [[scaling-laws]] for how this feeds into compute-optimal training.

---

## Related pages

- [[training]] — optimizer configuration and training loop
- [[scaling-laws]] — depth parameterization and compute budgets
- [[architecture]] — system overview
