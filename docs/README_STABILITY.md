# How We Made Training Actually Stable (Nanochat-style)

So you want to train deeper transformers without everything exploding? Let me tell you what actually matters.

## The Problem

Training a depth-8 transformer with "normal" initialization? Good luck. Here's what we saw:

```
Step 13430 | Loss: 4.3477 | grad_norm: 1842.3  # 💥 RIP
Step 13431 | Loss: NaN     | grad_norm: inf     # it's so over
```

Gradient norms going from ~1.0 to >1800 in a single step. Classic exploding gradients. The model is cooked.

## The Fix: 6 Things That Actually Matter

### 1. QK Normalization (THE BIG ONE ⭐)

This is the thing that actually fixed depth-8 training. Here's what we do:

```python
# After RoPE, BEFORE attention computation:
q = rms_norm(q)  # normalize queries
k = rms_norm(k)  # normalize keys
# now compute attention: softmax(qk^T / sqrt(d))
```

**Why does this work?** In deep models, Q and K can have large magnitudes. When you multiply them (`qk^T`), you get MASSIVE logits that explode the softmax. Normalizing Q and K keeps the attention logits well-behaved.

**Impact**: Gradient norms dropped from 100-1800 to 0.09-5.8. Training became boring (in a good way).

### 2. Zero-Init Residual Projections

Every transformer block ends with `x = x + attn(x)` and `x = x + mlp(x)`. At initialization, we want these to be pure identity functions:

```python
# In _init_weights:
if hasattr(module, "NANOGPT_SCALE_INIT"):  # marks c_proj layers
    torch.nn.init.zeros_(module.weight)    # start at zero!
```

Now at step 0, `attn(x) = 0` and `mlp(x) = 0`, so gradients flow cleanly through the residual stream. No explosions.

### 3. Width-Aware Initialization

When your embedding dimension grows (say 768 → 1536), you need to adjust your initialization:

```python
# For input projections (Q, K, V, MLP input):
bound = (3.0 ** 0.5) * (n_embed ** -0.5)  # sqrt(3) / sqrt(d)
torch.nn.init.uniform_(module.weight, -bound, bound)
```

This keeps activation magnitudes consistent as you scale the model. Without this, wider models would have exploding activations at init.

### 4. RMSNorm (The Textbook Answer is Missing the Point)

LayerNorm has learnable scale/bias parameters and normalizes mean+variance. RMSNorm just normalizes variance:

```python
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))  # that's it, no parameters
```

**The textbook explanation**: "LayerNorm fixes Internal Covariate Shift by stabilizing the mean and variance of layer inputs."

Sure. That's *procedurally* correct. But here's what's actually happening geometrically, and why everyone is switching to RMSNorm:

**The Geometric Reality:**

LayerNorm is projecting your hidden vector onto a hyperplane orthogonal to the uniform vector `[1, 1, ..., 1]`. When you subtract the mean, you're removing any component pointing in the direction where all dimensions are equal. Then you scale.

But here's the catch: **In high-dimensional spaces, learned hidden vectors are naturally orthogonal to this uniform vector anyway.** The "centering" step (calculating and subtracting the mean) is redundant. The model doesn't need it.

**Enter RMSNorm:**

If the vectors are already centered enough, why pay the computational tax to enforce it? RMSNorm drops the mean-centering operation entirely and handles only the scaling (re-scaling invariance).

**The Wins:**
1. **Simplification**: We stop forcing orthogonality to a vector we were already orthogonal to.
2. **Performance**: Removing the mean calculation/subtraction reduces computational overhead.
3. **Speed**: ~10-20% faster than LayerNorm for normalization operations in practice.

When you're training models with billions of parameters, that 10% could be *days* of compute saved.

**Bottom line**: Fewer parameters, simpler math, same (or better) results. LLaMA 2, Mistral, Gemma - they all use RMSNorm now. The geometric intuition tells you why it works; the benchmarks tell you it's worth it.

### 5. RoPE (Rotary Position Embeddings)

Instead of learned position embeddings that get added to tokens, we rotate Q and K in attention:

```python
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
```

This encodes relative positions directly into the attention mechanism. Better extrapolation to longer sequences, and one less set of parameters to worry about.

### 6. Muon Optimizer

For 2D weight matrices (everything except embeddings), we use Muon instead of AdamW:

```python
# Muon = Momentum Orthogonalized by Newton-schulz
# Better conditioning for matrix parameters
# See README_MUON.md for details
```

Think of it as Adam that's been beaten into better behavior for neural net weights.

### 7. Squared ReLU Activation

In the MLP, we use Squared ReLU instead of GELU:

```python
# In MLP forward pass:
x = self.c_fc(x)
x = F.relu(x).square()  # squared ReLU
x = self.c_proj(x)
```

**Why?** Squared ReLU (`relu(x)²`) gives you:
- Smoother gradients than standard ReLU (differentiable at 0)
- More expressivity than ReLU (quadratic instead of linear)
- Used in PaLM, nanochat, and other modern architectures
- Possibly faster than GELU (no exp/tanh approximations)

## What We Didn't Implement (Yet)

These are in nanochat but probably less critical:
- **Flash Attention 3**: Fancy fused CUDA kernels. We use PyTorch's SDPA which is "good enough" for now.
- **Value embeddings**: Extra learned embeddings added to values in alternating layers. Interesting but not stability-critical.
- **Learnable residual scales**: Per-layer scalars `lambda` for residual connections. Adds flexibility.
- **Sliding window attention**: Only attend to last N tokens. Good for long context.
- **Logit softcapping**: `tanh(logits / cap) * cap` to prevent extreme logits. Probably helps but we haven't seen issues yet.

## The Results

**Before** (depth-8, "standard" init):
```
grad_norm: 147.2 → 523.8 → 1842.3 → NaN  # 💀
Loss: 3.2 → 3.8 → 4.3 → NaN
```

**After** (depth-8, nanochat-style init):
```
grad_norm: 0.87 → 1.23 → 0.94 → 1.15  # 😴 (boring is good)
Loss: 4.2 → 3.8 → 3.6 → 3.4  # smooth sailing
```

## The Takeaway

If you're training deep transformers:

1. **Add QK normalization** - this is the big one
2. **Zero-init your residual projections** - clean gradients at init
3. **Scale your initialization by 1/√d** - consistent magnitudes
4. **Use RMSNorm** - simpler is better
5. **Consider RoPE** - better position encoding
6. **Maybe try Muon** - can't hurt

Do these 6 things and your depth-8+ models will actually train. The math checks out, and more importantly, it works in practice.

---

*"The best initialization is the one where your model doesn't explode."* - ancient ML wisdom
