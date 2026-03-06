# Muon Optimizer

**MomentUm Orthogonalized by Newton-schulz**

## What is Muon?

Muon is a specialized optimizer for 2D matrix parameters (weight matrices in neural networks). It combines SGD with momentum and adds an orthogonalization step that projects gradient updates onto the nearest orthogonal matrix.

**The key insight:** Most parameters in a transformer are 2D weight matrices (attention Q/K/V/O, MLP projections), and these benefit from having their updates constrained to preserve geometric properties - specifically, to behave like rotations rather than arbitrary transformations.

## Quick Start: Why Use Muon?

In nanochat (and this codebase), we use a **hybrid optimizer strategy**:

```python
# AdamW for embeddings and scalar parameters
adamw_params = [embeddings, unembedding, layernorm_params, ...]

# Muon for all 2D linear layer weights
muon_params = [attention_qkvo, mlp_up_down, ...]

optimizers = [
    AdamW(adamw_params, lr=0.2),
    Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.1)
]
```

**Why this division?**
- **Embeddings are lookup tables** - orthogonalization doesn't apply to discrete lookups
- **Linear layers are geometric transformations** - this is where Muon shines
- **Result:** Faster training with comparable or better performance

## Understanding Orthogonal Matrices

Before diving into how Muon works, let's build intuition for what "orthogonal" means.

### The Visual Intuition

An orthogonal matrix is like a **pure rotation or reflection** - it transforms data without stretching or squashing:

```
ORTHOGONAL TRANSFORMATIONS (preserve shape):
┌─────┐         ┌─────┐
│     │   →     │     │    [rotation: shape preserved]
└─────┘         └─────┘

┌─────┐         ┌─────┐
│     │   →     │     │    [reflection: shape preserved]
└─────┘         └─────┘

NON-ORTHOGONAL TRANSFORMATIONS (distort):
┌─────┐         ┌───────────┐
│     │   →     │           │    [scaling: stretched]
└─────┘         └───────────┘

┌─────┐         ╱───────╲
│     │   →     │        │    [shear: angles changed]
└─────┘         ╲───────╱
```

### Concrete Examples

**Orthogonal Matrices** (rotations and reflections):

```python
# 90° rotation counterclockwise
R = [[0, -1],
     [1,  0]]
# [1, 0] → [0, 1]  (rotated 90°, length preserved)

# Reflection across y-axis
F = [[-1, 0],
     [ 0, 1]]
# [3, 2] → [-3, 2]  (mirrored, length preserved)

# Identity (do nothing)
I = [[1, 0],
     [0, 1]]
```

**Non-Orthogonal Matrices** (for contrast):

```python
# Scaling (stretches)
S = [[2, 0],
     [0, 2]]
# [1, 0] → [2, 0]  (length changed: 1 → 2 ❌)

# Shearing (distorts)
H = [[1, 1],
     [0, 1]]
# Turns squares into parallelograms (angles changed ❌)
```

### The Mathematical Test

A matrix M is orthogonal if:
1. **Preserves lengths:** `|Mv| = |v|` for all vectors v
2. **Preserves angles:** Perpendicular vectors stay perpendicular
3. **Easy to invert:** `M^T @ M = I` (transpose equals inverse)

### Why This Matters for Neural Networks

In deep networks, signals pass through many transformations:

```python
output = W_12 @ W_11 @ W_10 @ ... @ W_2 @ W_1 @ input
```

**Without orthogonality:**
- Small eigenvalues → signal vanishes (gradients die, can't learn)
- Large eigenvalues → signal explodes (training becomes unstable)

**With orthogonal matrices:**
- All eigenvalues have magnitude = 1
- Signal "energy" is preserved through every layer
- Information flows stably through 50+ layers

### Physical Analogy

Think of rigid transformations in the real world:

- ✅ **Rotating a book** → orthogonal (shape unchanged)
- ✅ **Flipping a coin** → orthogonal (dimensions preserved)
- ❌ **Stretching a rubber band** → not orthogonal (length changed)
- ❌ **Squishing a sponge** → not orthogonal (volume changed)

### The Key Insight

When Muon orthogonalizes gradient updates, it's saying:

> "You can only **rotate** the weight space, not stretch or squash it. This keeps training stable."

Surprisingly, this restriction is enough to learn effectively. The constraint **helps** by preventing pathological behavior.

## How Muon Works

Muon performs four steps each iteration:

### Step 1: Nesterov Momentum

Standard SGD with momentum and Nesterov's lookahead:

```python
momentum_buffer = β * momentum_buffer + (1 - β) * grad
update = grad + β * momentum_buffer  # Nesterov lookahead
```

Default: β = 0.95

### Step 2: Orthogonalization via Polar Express

**This is the core innovation.** Instead of using the raw gradient update, Muon projects it onto the nearest orthogonal matrix using the [Polar Express Sign Method](https://arxiv.org/pdf/2505.16932):

```python
# Normalize
X = update / (||update|| * 1.02 + ε)

# Quintic iteration (5 steps)
for (a, b, c) in polar_express_coeffs:
    A = X @ X.T
    B = b*A + c*(A @ A)
    X = a*X + B @ X

orthogonalized_update = X
```

This efficiently computes an orthogonal approximation in just 5 iterations. The clever part: **it runs stably in bfloat16**, making it fast on modern GPUs.

**Historical note:** Original Muon used Newton-Schulz iteration. nanochat upgraded to Polar Express, which has better convergence properties.

### Step 3: Variance Reduction

Apply adaptive per-row or per-column learning rates (similar to Adam's second moment):

```python
v_mean = update².mean(dim=row_or_col)  # Per-row or per-column variance
second_moment = β₂ * second_moment + (1 - β₂) * v_mean
step_size = 1 / √(second_moment + ε)
scaled_update = update * step_size
```

Default: β₂ = 0.95

This gives each row/column its own adaptive learning rate, helping with conditioning.

### Step 4: Cautious Weight Decay

Apply weight decay **only where the update and weights agree in sign**:

```python
mask = (update * weight) >= 0
weight -= lr * update + lr * wd * weight * mask
```

This avoids "fighting" between gradient descent and regularization. The weight only decays when both the gradient and current weight point in the same direction.

## Implementation in nanochat

From `nanochat/nanochat/gpt.py`:

```python
def setup_optimizers(self, matrix_lr=0.02, weight_decay=0.1, ...):
    # Separate parameters by type
    matrix_params = []       # All 2D linear layer weights
    embedding_params = []    # Token/position embeddings
    lm_head_params = []      # Unembedding layer
    scalar_params = []       # LayerNorm, biases, etc.
    
    # AdamW for embeddings and non-matrix parameters
    adamw_optimizer = AdamW([
        dict(params=lm_head_params, lr=0.004),
        dict(params=embedding_params, lr=0.2),
        dict(params=scalar_params, lr=0.5),
    ], betas=(0.8, 0.95), weight_decay=0.0)
    
    # Muon for linear layer weights
    muon_optimizer = Muon(
        matrix_params, 
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=5,
        beta2=0.95
    )
    
    return [adamw_optimizer, muon_optimizer]
```

Both optimizers step together during training. Note that **weight decay is only applied to Muon**, not AdamW.

## Key Implementation Details

### Efficient Batched Updates

Muon groups parameters by shape and stacks them into a single tensor:

```python
# Naive approach: 12 separate kernel launches
for param in [W1, W2, ..., W12]:  # 12 layers, each (768, 3072)
    orthogonalize(param)
    update(param)

# Muon approach: 1 kernel launch
stacked = torch.stack([W1, W2, ..., W12])  # (12, 768, 3072)
muon_step_fused(stacked, ...)              # Single compiled kernel
torch._foreach_copy_([W1, W2, ...], stacked.unbind())
```

This dramatically reduces Python overhead and improves GPU utilization.

### Automatic Learning Rate Scaling

Muon applies shape-based LR scaling:

```python
effective_lr = lr * sqrt(max(1.0, n_rows / n_cols))
```

**Intuition:** Wide matrices (more columns than rows) have more degrees of freedom, so they can handle a higher learning rate without becoming unstable.

### Distributed Training (DistMuon)

`DistMuon` shards parameter updates across GPUs for memory efficiency:

1. **Reduce-scatter:** Each GPU gets gradients for its parameter chunk
2. **Local update:** Each GPU updates its chunk with `muon_step_fused`
3. **All-gather:** Broadcast updated parameters back to all GPUs

This reduces optimizer memory by `1/world_size` compared to replicating full state.

## Constraints

⚠️ **Muon only works with 2D parameters.** Use AdamW (or similar) for:

- Embedding layers (token embeddings, position embeddings)
- LayerNorm parameters (scale γ, bias β)
- Biases in linear layers
- Any 1D, 0D, or 3D+ tensors

This is why the hybrid approach is necessary.

## Hyperparameters

Defaults from extensive experiments (documented in `nanochat/dev/LOG.md`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.02 | Learning rate for matrix parameters |
| `momentum` | 0.95 | Nesterov momentum coefficient |
| `weight_decay` | 0.1 | Cautious weight decay coefficient |
| `ns_steps` | 5 | Number of Polar Express iterations |
| `beta2` | 0.95 | Second moment decay for variance reduction |

### Scale-Dependent Tuning

**Key finding:** Hyperparameters are **scale-dependent**. What works at small models (depth=12) doesn't necessarily transfer to larger models (depth=20).

**Empirical results:**
- At **depth=12**: Tuning `matrix_lr`, `weight_decay`, and `embedding_lr` gave ~0.002 bpb improvement
- At **depth=20**: The baseline is already well-tuned; aggressive tuning helps less (~0.0007 bpb improvement)
- **Simple wins:** Just setting `x0_beta1=0.96` (for scalar parameters in AdamW) captured most of the gains at scale

**Takeaway:** Start with defaults. Only tune if you have compute budget for sweeps at your target scale.

## Why It Works for Transformers

Transformers are mostly **linear projections**:
- Attention: Q, K, V, O matrices
- MLP: up-projection, down-projection

These operations are fundamentally about **comparing and combining** vectors in a geometric space. Orthogonal constraints enforce:

1. **Stable gradient flow** through deep networks
2. **Information preservation** (no accidental signal loss)
3. **Implicit regularization** toward well-conditioned transformations

The result: better training dynamics, especially as models get deeper.

## Why It's Called "Muon"

The name is a clever pun:

**Mu** (momentum) + **On** (orthogonalized) = **Muon**

Like the subatomic particle:
- **Heavy** (uses momentum, unlike massless AdamW)
- **Fundamental** (works on basic matrix operations)
- **Penetrating** (affects all layers uniformly)
- Heavier than an electron but still fast enough to be practical

## References

- **Original Muon blog post:** https://kellerjordan.github.io/posts/muon/
- **modded-nanogpt (Keller Jordan):** https://github.com/KellerJordan/modded-nanogpt
- **Polar Express paper:** Amsel et al., https://arxiv.org/pdf/2505.16932
- **nanochat repository:** https://github.com/karpathy/nanochat

---

*This implementation is adapted from modded-nanogpt and enhanced for nanochat with Polar Express orthogonalization, efficient batching, and distributed training support.*
