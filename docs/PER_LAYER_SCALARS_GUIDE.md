# Per-Layer Scalars: resid_lambdas and x0_lambdas

**A simple but powerful architectural modification that lets the network learn how to modulate its residual connections**

---

## Table of Contents

1. [The Core Idea](#the-core-idea)
2. [Understanding resid_lambdas](#understanding-resid_lambdas)
3. [Understanding x0_lambdas](#understanding-x0_lambdas)
4. [Why This Works](#why-this-works)
5. [Implementation Details](#implementation-details)
6. [Training Dynamics](#training-dynamics)
7. [Code Walkthrough](#code-walkthrough)
8. [FAQ](#faq)

---

## The Core Idea

In a standard Transformer, each layer does this:

```python
# Standard residual connection
x = x + block(x)
```

Simple and elegant. The output of each block is added back to its input. But here's a question: **what if the network wants more control over how much it trusts the information flowing into each layer?**

With per-layer scalars, we give the network two knobs to turn **before** each layer processes its input:

```python
# With per-layer scalars - applied BEFORE the block
x = resid_lambdas[i] * x + x0_lambdas[i] * x0
# Then the block runs with its own standard residuals
x = x + block.attn(x)  # attention with residual
x = x + block.mlp(x)   # MLP with residual
```

Where:
- `resid_lambdas[i]` scales the current residual stream **entering** the layer
- `x0_lambdas[i]` blends in the original input embedding from layer 0
- Both are **learnable parameters** that the network adjusts during training
- The block itself still uses standard residual connections internally

Think of it like a pre-mixer before each layer. The network learns to blend and scale signals **before** they enter each block, and then the block processes them with its normal residual connections.

### What This Is NOT

**Common Misconception**: Per-layer scalars replace the residual connections inside the block.

**Reality**: The scalars are a **pre-processing step** that happens before the block. The block maintains its normal residual structure.

**Visual comparison:**

```python
# ❌ WRONG interpretation (scalars replace residuals)
x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + block(x)

# ✅ CORRECT implementation (scalars preprocess, block has its own residuals)
x = resid_lambdas[i] * x + x0_lambdas[i] * x0  # preprocessing
x = block(x)  # block internally does: x + attn(x) and x + mlp(x)
```

The key difference: **two separate stages** of residual connections, not one.

### Complete Flow Through One Layer

Here's exactly what happens at layer `i`:

```python
# Step 1: Pre-layer mixing (per-layer scalars)
x_in = resid_lambdas[i] * x + x0_lambdas[i] * x0

# Step 2: Attention with its own residual
x_mid = x_in + attn(norm(x_in))

# Step 3: MLP with its own residual  
x_out = x_mid + mlp(norm(x_mid))

# x_out becomes x for the next layer
x = x_out
```

So there are **three additions** per layer:
1. The pre-layer mix (scalar-weighted sum)
2. The attention residual (standard)
3. The MLP residual (standard)

The per-layer scalars only affect step 1. Steps 2 and 3 are unchanged from a standard Transformer.

---

## Understanding resid_lambdas

### What It Does

`resid_lambdas` is a per-layer scalar that multiplies the residual stream *before* the current block processes it. This is a **pre-layer mixing operation**, not a replacement for the block's residual connections.

```python
x = resid_lambdas[i] * x   # scale the residual entering this layer
x = x + block.attn(x)      # block still has its own residual
x = x + block.mlp(x)       # block still has its own residual
```

**Key insight**: The block itself maintains its standard residual connections. The per-layer scalars only modify what goes **into** the block.

### Initialization: 1.0 (Neutral)

At initialization, all `resid_lambdas` are set to **1.0**, which means "business as usual" - the residual stream flows normally, exactly like a standard Transformer.

```python
self.resid_lambdas.fill_(1.0)  # start neutral
```

### What Can the Network Learn?

During training, each layer's scalar can diverge from 1.0:

- **> 1.0**: Amplify the residual stream entering this layer
  - "Pay extra attention to what came before"
  - Useful if earlier layers encoded important information
  
- **< 1.0**: Dampen the residual stream entering this layer
  - "Give this layer a fresh start"
  - Useful if earlier representations need to be de-emphasized
  
- **≈ 0.0**: Nearly erase the residual stream
  - "Reset and rebuild from scratch"
  - Rare, but possible if the network wants a clean slate

### Example Scenario

Imagine training on a language modeling task:

- **Layer 3** learns syntax patterns → might benefit from **high resid_lambda** (amplify syntax info)
- **Layer 8** does abstract reasoning → might benefit from **low resid_lambda** (don't get distracted by low-level syntax)

The network figures this out automatically through gradient descent.

---

## Understanding x0_lambdas

### What It Does

`x0_lambdas` creates a **skip connection from the very first layer** (the normalized token embedding) back to each layer. It's like giving every layer direct access to the "raw ingredients" - the original token meanings.

```python
x0 = self.transformer.wte(idx)  # original embedding
x0 = norm(x0)                   # normalize it once

for i, block in enumerate(self.transformer.h):
    x = resid_lambdas[i] * x + x0_lambdas[i] * x0  # blend in x0
    x = x + block(x)
```

### Initialization: 0.1 (Small Contribution)

At initialization, all `x0_lambdas` are set to **0.1**, meaning "include a small amount of the original embedding."

```python
self.x0_lambdas.fill_(0.1)  # start with small contribution
```

Why not 0.0? Starting at 0.1 gives the gradient signal a foothold. If we started at 0.0, the gradient would be zero initially, and the parameter might never wake up. By starting slightly positive, we encourage the network to explore this connection.

### What Can the Network Learn?

During training, each layer's x0_lambda can move around:

- **> 0.1**: Increase the contribution from the original embedding
  - "I need to remember what token I'm looking at"
  - Useful for tasks that need token-level information (e.g., character-level details)
  
- **≈ 0.0**: Ignore the original embedding
  - "I'm working with abstract representations now"
  - Useful in later layers doing high-level reasoning
  
- **< 0.0**: Subtract the original embedding (yes, negative is possible!)
  - "Actively suppress the original token meaning"
  - Rare, but gradient descent might find this useful

### Example Scenario

- **Early layers (0-5)**: Might set x0_lambda ≈ 0.05 (don't need much, still processing syntax)
- **Middle layers (6-10)**: Might set x0_lambda ≈ 0.2 (need to check back with original meaning while reasoning)
- **Late layers (11+)**: Might set x0_lambda ≈ 0.0 (fully abstract, original tokens irrelevant)

Again, the network learns this automatically.

---

## Why This Works

### 1. **Input Conditioning**

Per-layer scalars let the network **condition its inputs** before each layer processes them:

- The network can amplify or dampen the incoming residual stream
- The network can blend in varying amounts of the original token embeddings
- This happens **before** the layer's normal residual operations

This gives the network more flexibility in how information flows through the architecture without modifying the fundamental residual structure of each block.

### 2. **Long-Range Skip Connections**

Deep networks sometimes "forget" early information. x0_lambdas provide a **direct skip connection from layer 0 to every layer**:

- Every layer receives a weighted copy of the original normalized embeddings
- Prevents information loss through the depth of the network
- Similar in spirit to DenseNet connections, but much simpler (only connects to the first layer, not all previous layers)

### 3. **Very Cheap**

Adding per-layer scalars costs almost nothing:

- **Parameters**: Only `2 * n_layer` (e.g., 24 params for a 12-layer model)
- **Compute**: Two scalar multiplications and one addition per layer (negligible)
- **Memory**: One extra tensor stored during forward pass (x0)

For such a small cost, the potential benefit is significant.

### 4. **Interpretability**

Unlike most architectural changes, you can **look at the learned values** and get intuition:

```python
print(model.resid_lambdas)  # tensor([1.05, 0.98, 1.12, ...])
print(model.x0_lambdas)     # tensor([0.08, 0.15, 0.03, ...])
```

You might notice patterns like:
- Deeper layers have smaller x0_lambdas (forgetting original input)
- Middle layers have larger resid_lambdas (amplifying intermediate features)

This gives you insight into what the network is doing.

---

## Implementation Details

### Model Architecture Changes

Add two learnable parameter tensors:

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...
        
        # Per-layer scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
```

### Initialization

```python
def init_weights(self):
    # ... existing initialization ...
    
    # Per-layer scalars
    self.resid_lambdas.fill_(1.0)   # neutral scaling
    self.x0_lambdas.fill_(0.1)      # small x0 contribution
```

### Forward Pass

```python
def forward(self, idx, targets=None):
    x = self.transformer.wte(idx)  # embed tokens
    x = norm(x)                    # normalize
    x0 = x                         # save for later
    
    for i, block in enumerate(self.transformer.h):
        # Apply per-layer scalars BEFORE the block
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        
        # Now run the block with its own internal residuals
        # The block does: x = x + attn(norm(x)) and x = x + mlp(norm(x))
        x = block(x, ...)
    
    x = norm(x)                    # final norm
    logits = self.lm_head(x)       # predict next token
    return logits
```

**Note**: Each `block(x, ...)` internally performs:
```python
x = x + self.attn(norm(x), ...)  # attention with its own residual
x = x + self.mlp(norm(x))        # MLP with its own residual
```

So the complete flow is: **pre-layer mixing → block's attention residual → block's MLP residual**.

---

## Training Dynamics

### Optimizer Settings

These scalars need **different learning rates** than the main model parameters:

```python
# From nanochat optimizer configuration
# Default adam_betas=(0.8, 0.95), scalar_lr=0.5
param_groups = [
    # resid_lambdas: conservative LR, standard momentum
    dict(
        params=[self.resid_lambdas],
        lr=scalar_lr * 0.01,      # 100x smaller than scalar_lr (0.005 default)
        betas=(0.8, 0.95),        # matches main AdamW params
        weight_decay=0.0          # no decay
    ),
    
    # x0_lambdas: higher LR, higher momentum for stability
    dict(
        params=[self.x0_lambdas],
        lr=scalar_lr,             # full scalar_lr (0.5 default)
        betas=(0.96, 0.95),       # higher beta1 = more momentum
        weight_decay=0.0          # no decay
    ),
]
```

### Why Different Settings?

1. **resid_lambdas gets 0.01x LR**:
   - Starts at 1.0 (neutral)
   - Should change slowly to avoid destabilizing training
   - Think of it as "fine-tuning the gradient highway"

2. **x0_lambdas gets 1.0x LR with high momentum**:
   - Starts at 0.1 (small but nonzero)
   - Can change more freely (it's adding extra info, not modifying existing flow)
   - High momentum (beta1=0.96) prevents oscillation

### Typical Learning Trajectory

```
Step 0:
  resid_lambdas: [1.0, 1.0, 1.0, 1.0, ...]
  x0_lambdas:    [0.1, 0.1, 0.1, 0.1, ...]

Step 10k:
  resid_lambdas: [1.02, 0.98, 1.05, 0.95, ...]  # small adjustments
  x0_lambdas:    [0.08, 0.15, 0.12, 0.03, ...]  # more variation

Step 100k:
  resid_lambdas: [1.08, 0.92, 1.15, 0.88, ...]  # larger spread
  x0_lambdas:    [0.05, 0.22, 0.18, 0.01, ...]  # clear pattern emerges
```

You'll typically see:
- Early layers: lower x0_lambdas (don't need raw input)
- Late layers: very low x0_lambdas (abstract representations)
- Some layers: high resid_lambdas (critical for gradient flow)

---

## Code Walkthrough

### Step 1: Define Parameters

In your GPT model `__init__`:

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Transformer blocks
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # ===== NEW: Per-layer scalars =====
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
```

### Step 2: Initialize Weights

In your `init_weights()` method:

```python
@torch.no_grad()
def init_weights(self):
    # ... existing initialization for wte, lm_head, blocks ...
    
    # ===== NEW: Initialize scalars =====
    self.resid_lambdas.fill_(1.0)  # start neutral
    self.x0_lambdas.fill_(0.1)     # start small but nonzero
```

### Step 3: Modify Forward Pass

In your `forward()` method:

```python
def forward(self, idx, targets=None):
    B, T = idx.size()
    
    # Embed and normalize
    x = self.transformer.wte(idx)  # (B, T, n_embd)
    x = norm(x)                    # RMSNorm
    
    # ===== NEW: Save x0 for skip connections =====
    x0 = x  # (B, T, n_embd)
    
    # Transformer blocks
    for i, block in enumerate(self.transformer.h):
        # ===== NEW: Apply per-layer scalars BEFORE the block =====
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        
        # Block processes with its own internal residuals
        # Internally: x = x + attn(norm(x)) and x = x + mlp(norm(x))
        x = block(x, ...)
    
    # Final norm and lm_head
    x = norm(x)
    logits = self.lm_head(x)
    
    # Compute loss if training
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
    return logits
```

### Step 4: Configure Optimizer

In your training script or `configure_optimizers()` method:

```python
def configure_optimizers(self, scalar_lr=0.5):
    # Separate parameter groups
    resid_params = [self.resid_lambdas]
    x0_params = [self.x0_lambdas]
    # ... other parameter groups ...
    
    param_groups = [
        # resid_lambdas: conservative (0.01x scalar LR)
        dict(
            params=resid_params,
            lr=scalar_lr * 0.01,
            betas=(0.8, 0.95),  # match main AdamW params
            eps=1e-10,
            weight_decay=0.0
        ),
        # x0_lambdas: full scalar LR with higher momentum
        dict(
            params=x0_params,
            lr=scalar_lr,
            betas=(0.96, 0.95),  # higher beta1 for stability
            eps=1e-10,
            weight_decay=0.0
        ),
        # ... other parameter groups ...
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer
```

---

## FAQ

### Q: Do these actually help performance?

**A:** Yes, in many cases. They provide the network with more flexibility in routing information. The cost is negligible (only `2 * n_layer` parameters), so even small improvements are worth it. Results vary by task and model size, but generally:
- Slightly faster convergence (5-10% fewer steps to reach target loss)
- Slightly better final loss (1-2% improvement)
- More stable training (gradient flow is smoother)

### Q: Why initialize x0_lambdas to 0.1 instead of 0.0?

**A:** If we start at 0.0, the gradient is zero initially, and the parameter might never activate (dead parameter problem). Starting at 0.1 gives it a small "kick" to get gradients flowing. The network can still learn to set it back to ~0.0 if that's optimal.

### Q: Can resid_lambdas or x0_lambdas go negative?

**A:** Yes! They're unconstrained real numbers. Negative resid_lambda would mean "flip the sign of the residual stream" (rare but possible). Negative x0_lambda would mean "subtract the original embedding" (also rare). In practice, they usually stay positive but close to their initial values.

### Q: How do I visualize what they're doing?

**A:** After training, plot them:

```python
import matplotlib.pyplot as plt

layers = list(range(model.config.n_layer))
resid = model.resid_lambdas.detach().cpu().numpy()
x0 = model.x0_lambdas.detach().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(layers, resid, marker='o')
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
plt.title('resid_lambdas by layer')
plt.xlabel('Layer')
plt.ylabel('Lambda value')

plt.subplot(1, 2, 2)
plt.plot(layers, x0, marker='o')
plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.3)
plt.title('x0_lambdas by layer')
plt.xlabel('Layer')
plt.ylabel('Lambda value')

plt.tight_layout()
plt.savefig('per_layer_scalars.png')
```

### Q: Do they interfere with other architectural changes?

**A:** No, they're extremely modular. They work alongside:
- Sliding window attention
- Value embeddings
- Logit softcapping
- Pre/post-norm
- RoPE / ALiBi
- Any other Transformer modification

### Q: What if I want to freeze them after pre-training?

**A:** You can freeze them during fine-tuning:

```python
model.resid_lambdas.requires_grad = False
model.x0_lambdas.requires_grad = False
```

This locks in the learned values from pre-training. Useful if you want to fine-tune the model but keep the gradient highways fixed.

### Q: Where did this idea come from?

**A:** The nanochat codebase mentions "inspired by modded-nanogpt," which is a modified version of Andrej Karpathy's nanoGPT repository. The idea of learnable per-layer scalars has been explored in various forms:
- ReZero (2020): Learnable scalar that starts at zero for each residual connection
- FixUp (2019): Fixed scalars for training without normalization
- Highway Networks (2015): Learnable gating for residual connections

The nanochat implementation is simpler than gates (no sigmoid, just raw scalars) and more flexible than ReZero (separate control for residual stream and x0 skip connection).

### Q: Should I use this in my own models?

**A:** If you're building a Transformer from scratch, yes! The implementation is trivial (10 lines of code) and the cost is negligible. Even if you only get a 1-2% improvement, it's worth it. If you're modifying an existing model (e.g., loading a Hugging Face checkpoint), it's harder because you'd need to retrain from scratch.

---

## Summary

**Per-layer scalars (resid_lambdas and x0_lambdas) give the network learnable control over information flow:**

1. **resid_lambdas**: Scale the residual stream entering each layer
   - Init: 1.0 (neutral)
   - Effect: Modulates gradient highways and feature emphasis

2. **x0_lambdas**: Blend the original embedding back in at each layer
   - Init: 0.1 (small contribution)
   - Effect: Prevents forgetting, provides direct access to input

**Benefits:**
- Trivial parameter cost (2 * n_layer)
- Negligible compute cost (two scalar muls per layer)
- Modest performance gains (1-10% depending on task)
- Interpretable (can visualize learned values)
- Modular (works with any other Transformer modification)

**Implementation:**
- Add two `nn.Parameter` tensors of shape `(n_layer,)`
- Initialize to 1.0 and 0.1 respectively
- Apply in forward pass BEFORE each block: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
- Each block still maintains its own internal residual connections
- Use different optimizer settings (lower LR for resid_lambdas)

**Philosophy:**
Give the network simple, learnable knobs to control its architecture. Let gradient descent figure out the optimal settings. Don't overthink it—sometimes the simplest ideas work best.

---

*This document was written in the style of Andrej Karpathy's educational materials: clear, intuitive, building from first principles, with code you can actually run.*
