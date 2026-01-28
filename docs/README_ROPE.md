# RoPE: Rotary Position Embeddings

## Introduction

Position information is critical for language models. Without it, "The cat chased the mouse" would be indistinguishable from "The mouse chased the cat". But *how* we encode position has evolved significantly.

This document explains **RoPE (Rotary Position Embeddings)**, the modern approach to position encoding used in LLaMA, GPT-NeoX, PaLM, and this NanoGPT implementation. We'll start from the original GPT-2 approach, understand its limitations, and see why RoPE is a significant improvement.

## The Old Way: Learned Positional Embeddings (GPT-2 Style)

### How It Worked

In the original GPT-2 and early transformers, position was encoded through **learned positional embeddings**:

```python
# Original GPT-2 approach
self.wte = nn.Embedding(vocab_size, n_embed)      # Token embeddings
self.wpe = nn.Embedding(block_size, n_embed)      # Position embeddings

# During forward pass
token_embeds = self.wte(idx)                      # (B, T, C)
pos = torch.arange(0, T, device=device)           # [0, 1, 2, ..., T-1]
pos_embeds = self.wpe(pos)                        # (T, C)
x = token_embeds + pos_embeds                     # Add position info
```

### Key Properties

1. **Absolute positioning**: Each position (0, 1, 2, ...) gets its own learned embedding vector
2. **Added to tokens**: Position embeddings are summed with token embeddings
3. **Fixed context length**: Can't handle sequences longer than training (block_size)
4. **Position-agnostic attention**: The attention mechanism itself doesn't "see" positions directly

### The Problem

Consider attention between tokens at positions `i` and `j`. The model learns position through:
- The additive embeddings `pos_embeds[i]` and `pos_embeds[j]`
- Whatever position-dependent patterns emerge in the learned Q, K, V projections

But there's no explicit encoding of **relative position** (distance `j - i`). The model must implicitly learn that:
- Position 5 attending to position 3 (distance = -2)
- Position 100 attending to position 98 (distance = -2)

...should probably behave similarly! This is wasteful and makes generalization harder.

### More Issues

**Context length problem**: If you train on sequences of length 1024 but then want to run inference on length 2048, you're in trouble. The model never learned embeddings for positions 1024-2047. You can try:
- Extrapolation (usually poor)
- Interpolation (rescale positions, breaks learned patterns)
- Fine-tuning on longer sequences (expensive)

**Parameter cost**: For block_size=2048 and n_embed=768, you need `2048 × 768 = 1,572,864` parameters just for position embeddings. That's roughly 1.5M parameters doing nothing but saying "this is position 0", "this is position 1", etc.

## The New Way: Rotary Position Embeddings (RoPE)

RoPE takes a fundamentally different approach: **encode position through rotation in the complex plane**.

### Core Insight

Instead of *adding* position information, RoPE *rotates* the query and key vectors by an angle proportional to their position.

Think about a clock: if the hour hand rotates 30° per hour, you can tell time by looking at the angle. Similarly, if we rotate each token's representation by an angle proportional to its position, attention scores naturally encode relative position.

### The Math (Building Intuition)

Let's build this up step by step.

**Step 1: Complex Numbers**

Represent each 2D pair in the embedding as a complex number:
- Instead of: `[x₁, x₂]` as separate real numbers
- Think of: `z = x₁ + i·x₂` as a complex number

**Step 2: Rotation in Complex Plane**

Rotating a complex number `z` by angle `θ`:
```
z' = z · e^(iθ) = (x₁ + i·x₂) · (cos θ + i·sin θ)
```

In Cartesian form:
```
x₁' = x₁·cos(θ) - x₂·sin(θ)
x₂' = x₁·sin(θ) + x₂·cos(θ)
```

This is exactly a 2D rotation matrix!

**Step 3: Position-Dependent Rotation**

For a token at position `m`, rotate by angle `θ_m = m · θ`:
```
q_m' = q_m · e^(i·m·θ)
k_n' = k_n · e^(i·n·θ)
```

**Step 4: The Magic - Relative Position Emerges**

When computing attention score between query at position `m` and key at position `n`:

```
score = q_m' · k_n'ᵀ
      = (q_m · e^(i·m·θ)) · (k_n · e^(i·n·θ))ᵀ
      = q_m · k_nᵀ · e^(i·(m-n)·θ)
```

Notice: The attention score is modulated by `e^(i·(m-n)·θ)`, which depends only on **relative position** `(m-n)`!

This is beautiful: without any learned parameters, we've encoded relative position into the attention mechanism itself.

### Different Rotation Speeds for Different Dimensions

Here's where it gets even better. We don't use the same rotation speed for all dimensions. Instead:

```python
# Compute inverse frequencies - higher dimensions rotate slower
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
```

Why different speeds? Think of it like a clock with multiple hands:
- Fast rotation (high frequency) → captures short-range dependencies
- Slow rotation (low frequency) → captures long-range dependencies

This is similar to sinusoidal position embeddings (Attention Is All You Need), but applied through rotation rather than addition.

### Implementation Details

Here's the actual implementation:

```python
def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    """Precompute cos and sin for all positions"""
    # Different rotation speeds for each dimension pair
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    
    # Position indices [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len)
    
    # Outer product: position × frequency = rotation angle
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim//2)
    
    # Precompute cos and sin for efficiency
    cos, sin = freqs.cos(), freqs.sin()
    
    return cos, sin  # (seq_len, head_dim//2)

def apply_rotary_emb(x, cos, sin):
    """Apply rotation to query or key"""
    # Split into pairs
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    
    # Rotate each pair
    y1 = x1 * cos - x2 * sin  # Real part
    y2 = x1 * sin + x2 * cos  # Imaginary part
    
    return torch.cat([y1, y2], dim=-1)
```

Usage in attention:

```python
# In attention layer
q = self.q_proj(x)  # (B, T, n_head, head_dim)
k = self.k_proj(x)  # (B, T, n_head, head_dim)

# Apply RoPE to Q and K (NOT V!)
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)

# Now compute attention normally
attn = (q @ k.T) * scale  # Attention scores now encode relative position!
```

**Critical detail**: We only rotate Q and K, not V! Why?
- Q and K determine *where* to attend (position-dependent)
- V contains *what* to retrieve (position can be less important)

## Key Differences: Old vs New

| Aspect | Learned Positional Embeddings | RoPE |
|--------|------------------------------|------|
| **Method** | Add learned vectors to tokens | Rotate Q/K by position-dependent angles |
| **Position Type** | Absolute (position 0, 1, 2, ...) | Relative (distance between tokens) |
| **Parameters** | `block_size × n_embed` learned params | Zero learned parameters! |
| **Where Applied** | Before transformer blocks | Inside attention, to Q and K |
| **Context Length** | Fixed at training time | Can extrapolate beyond training |
| **KV Cache Friendly** | Requires re-adding embeddings | Natural support, just offset angles |

## Why RoPE is Better

### 1. **Relative Position Encoding**

RoPE naturally captures relative distances. The model learns patterns like "attend to the previous token" rather than "when at position 5, attend to position 4".

### 2. **Length Extrapolation**

Since position is encoded through rotation angles, we can handle longer sequences than trained on:
- Train on sequences up to 2048 tokens
- Generate with 4096+ tokens at inference
- Just need to compute `cos(m·θ)` and `sin(m·θ)` for larger `m`

This is increasingly important as we push toward longer contexts.

### 3. **Zero Parameters**

RoPE uses no learned parameters. It's a pure geometric transformation. This means:
- Fewer parameters to learn (faster training, less memory)
- No overfitting of position representations
- More sample-efficient training

### 4. **KV Cache Friendly**

During autoregressive generation, we cache key/value vectors. With RoPE:
- Precomputed RoPE cos/sin values are cheap to compute and store
- Just offset the position index when generating new tokens
- No need to recompute or adjust cached K/V pairs

With learned position embeddings, you'd need to store position info with cached states or handle it carefully.

### 5. **Better Inductive Bias**

The rotation-based approach provides a strong geometric inductive bias:
- Smooth interpolation between positions
- Natural decay of positional signal (through different rotation speeds)
- Respects the sequential nature of language

## Implementation in This Codebase

### Precomputation (in GPT model)

```python
# In GPT.__init__() - precompute RoPE buffers
self.rotary_seq_len = config.block_size * 10  # Over-allocate for inference
head_dim = config.n_embed // config.n_head
cos, sin = precompute_rotary_embeddings(
    self.rotary_seq_len, 
    head_dim, 
    base=10000,
    dtype=torch.bfloat16
)
# Register as non-persistent buffers (recomputed on load)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)
```

Note: We precompute for `block_size × 10` to support longer sequences during inference!

### During Forward Pass

```python
# In GPT.forward() - slice for current sequence
T0 = 0 if kv_cache is None else kv_cache.get_pos()  # Offset for KV cache
cos_sin = (
    self.cos[:, T0:T0+T, :, :],  # Current sequence slice
    self.sin[:, T0:T0+T, :, :]
)

# Pass to each transformer block
for block in self.transformer.h:
    x = block(x, cos_sin=cos_sin, kv_cache=kv_cache)
```

### In Attention Layer

```python
# In CausalSelfAttention.forward()
q, k, v = self.c_attn(x).split([...], dim=2)
q = q.view(B, T, n_head, head_dim)
k = k.view(B, T, n_kv_head, head_dim)
v = v.view(B, T, n_kv_head, head_dim)

# Apply RoPE to queries and keys
if cos_sin is not None:
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    # Note: V is NOT rotated!

# Continue with attention computation...
```

## Practical Considerations

### Base Frequency

The `base` parameter (default 10000) controls the rotation speed. Higher base = slower rotation = better for longer contexts.

Recent work (e.g., LLaMA 2) uses:
- `base=10000` for sequences up to ~2k tokens
- `base=500000` for very long contexts (32k+ tokens)

This is because slower rotation allows better extrapolation to longer sequences.

### Memory Efficiency

RoPE cos/sin buffers are registered as **non-persistent** (`persistent=False`):
- Not saved in model checkpoints (saves disk space)
- Recomputed from scratch when loading model
- Cheap to recompute (just `cos(m·θ)` and `sin(m·θ)`)

### Precision

We use `bfloat16` for RoPE buffers:
- Good enough precision for rotation operations
- Saves memory (half the size of float32)
- Compatible with mixed-precision training

## Historical Context

**2017**: Attention Is All You Need (original Transformer)
- Sinusoidal position embeddings: `PE(pos, 2i) = sin(pos/10000^(2i/d))`
- Fixed (not learned), but added to embeddings
- Still absolute positioning

**2019**: GPT-2
- Switched to learned positional embeddings
- More flexible, but loses the nice properties of sinusoidal
- Becomes standard in many models

**2021**: RoFormer paper introduces RoPE
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Combines benefits of both approaches
- Geometric interpretation through rotation

**2023**: Adopted in LLaMA, GPT-NeoX, PaLM, and most modern LLMs
- Becomes the de facto standard for position encoding
- Key enabler for long context models (100k+ tokens)

## Conclusion

RoPE is one of those ideas that seems obvious in hindsight but represents a genuine improvement:
- More efficient (zero parameters)
- More general (relative positioning)
- More flexible (length extrapolation)
- More practical (KV cache friendly)

By encoding position through geometric rotation rather than learned or added embeddings, RoPE provides a better inductive bias for sequence modeling. This is why you'll find it in nearly every modern LLM.

The key insight: **position is not something you add to tokens; it's something you encode in how tokens attend to each other**.

## Further Reading

- **Original RoFormer paper**: [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- **LLaMA paper**: Uses RoPE with various base frequencies
- **nanochat codebase**: Reference implementation this code is inspired by
- **Positional encoding comparison**: Analysis of different position encoding schemes

## Files in This Codebase

- `src/gpt_2/rope.py` - RoPE implementation (precompute + apply)
- `src/gpt_2/gpt2_model.py` - RoPE buffer setup and slicing
- `src/gpt_2/attention.py` - RoPE application in attention layer
- `docs/README_ROPE.md` - This document!

---

*"The best ideas are the ones that make you wonder why nobody thought of them sooner."* - Unknown

RoPE is one of those ideas. 🌀
