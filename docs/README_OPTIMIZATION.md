# Optimization Guide for NanoGPT
## Momentum, Nesterov, Weight Decay, and Muon Explained

---

## Quick Reference

### TL;DR: Our Setup

We use a **hybrid optimizer approach** with:

**AdamW** for:
- Token embeddings (`lr=0.3`) - fast adaptation
- Output head (`lr=0.004`) - stable predictions
- Biases (`lr=0.5`) - quick adjustments

**Muon** for:
- All transformer weight matrices (`lr=0.02`)
- Nesterov momentum (0.85 → 0.95 warmup)
- Decaying weight decay (0.1 → 0.0)

### Key Hyperparameters

```python
# Base learning rates (automatically scaled by batch size and depth)
embedding_lr = 0.3        # Fast: sparse parameters adapt quickly
unembedding_lr = 0.004    # Slow: output distribution stays stable
matrix_lr = 0.02          # Medium: core transformer weights  
scalar_lr = 0.5           # Fast: small bias terms

# Muon momentum schedule
momentum: 0.85 → 0.95     # Warmup over first 300 steps

# Muon weight decay schedule
weight_decay: 0.1 → 0.0   # Linear decay over full training

# AdamW betas
adam_beta1 = 0.8          # First moment
adam_beta2 = 0.95         # Second moment
```

### Automatic Scaling Rules

1. **Batch Size Scaling**: `LR ∝ √(batch_size / 524288)`
2. **Depth Scaling**: `LR ∝ 1/√model_dim` (tuned at model_dim=768)
3. **Weight Decay Scaling**: `WD ∝ 1/depth²` (tuned at depth=12)

### Quick Start

```bash
# Train with depth 12 (recommended)
make ddp-train DEPTH=12 TARGET_FLOPS=1e18

# Scaling law experiment
make ddp-train DEPTH=6 TARGET_FLOPS=1e17

# Customize training (example with 2 GPUs and CORE evals)
make ddp-train NGPUS=2 DEPTH=12 TARGET_FLOPS=1e18 CORE_EVALS=true
```

### What to Watch in Wandb

**Essential metrics**:
- `train_loss` - Should decrease smoothly
- `lr_multiplier` - LR schedule visualization
- `mfu` - Model FLOP Utilization (aim for >40%)

**Learning rates** (Muon mode):
- `lr/embedding` - Token embeddings (~0.3)
- `lr/matrix` - Transformer weights (~0.02)
- `lr/unembedding` - Output head (~0.004)
- `lr/scalar` - Bias terms (~0.5)

**Muon dynamics**:
- `muon/momentum` - Should ramp 0.85→0.95 over 300 steps
- `muon/weight_decay` - Should decay linearly to 0

### Debugging Checklist

| Problem | Solutions |
|---------|-----------|
| **Loss explodes** | ✅ Reduce `matrix_lr` (0.02 → 0.01)<br>✅ Increase gradient clipping<br>✅ Check gradient norm (<5.0) |
| **Training slow** | ✅ Check MFU (>30%)<br>✅ Increase batch size<br>✅ Verify Flash Attention active |
| **Loss plateaus** | ✅ Decrease weight decay<br>✅ Check LR not too low<br>✅ Verify warmdown timing |

---

## Deep Dive: Understanding Optimization from First Principles

### Table of Contents
1. [The Problem: Vanilla Gradient Descent](#1-the-problem-vanilla-gradient-descent)
2. [Momentum: Adding Physics to Optimization](#2-momentum-adding-physics-to-optimization)
3. [Nesterov Momentum: Look Before You Leap](#3-nesterov-momentum-look-before-you-leap)
4. [Weight Decay: Keeping Parameters in Check](#4-weight-decay-keeping-parameters-in-check)
5. [Putting It Together: Muon Optimizer](#5-putting-it-together-muon-optimizer)
6. [Practical Tips and Intuitions](#6-practical-tips-and-intuitions)

---

## 1. The Problem: Vanilla Gradient Descent

Let's start with the simplest optimizer: **Stochastic Gradient Descent (SGD)**.

### The Basic Idea

You have a loss function `L(θ)` where `θ` are your model parameters. You want to minimize this loss. The gradient `∇L(θ)` tells you which direction makes the loss go *up*, so you go in the opposite direction:

```python
# Vanilla SGD (the simplest optimizer)
θ = θ - learning_rate * gradient
```

That's it! Subtract the gradient (scaled by learning rate) from your parameters.

### The Problem: Noisy, Zigzagging Paths

Imagine you're trying to roll a ball down a valley to reach the bottom:

```
         ╱╲              <- Steep walls (high gradient)
        ╱  ╲
       ╱    ╲
      ╱      ╲           <- Valley floor (low gradient)
     ╱________╲          <- Goal!
```

With vanilla SGD:
- **Problem 1**: Each gradient is noisy (computed on a mini-batch)
- **Problem 2**: You zigzag across the valley instead of going straight down
- **Problem 3**: Progress is slow in flat regions, unstable in steep regions

Here's what actually happens:

```python
# Training loop with vanilla SGD
for step in range(1000):
    # Compute gradient on a mini-batch (noisy!)
    loss = model(batch_x, batch_y)
    gradient = compute_gradient(loss)
    
    # Update parameters
    theta = theta - 0.01 * gradient  # lr=0.01
    
    # Problem: Each update is myopic (only looks at current gradient)
    # Result: Zigzag path, slow convergence
```

**Key insight:** We're throwing away information! Each gradient tells us something useful, but we only use it once and forget it.

---

## 2. Momentum: Adding Physics to Optimization

### The Core Idea: Remember the Past

What if instead of following each noisy gradient blindly, we **accumulate** gradients over time? This is momentum.

```python
# Momentum SGD (much better!)
velocity = 0  # Initialize

for step in range(1000):
    gradient = compute_gradient(loss)
    
    # Accumulate gradients into velocity
    velocity = momentum * velocity + gradient
    
    # Update using velocity (not raw gradient)
    theta = theta - learning_rate * velocity
```

### The Physics Analogy

Think of a ball rolling down a hill:
- **Velocity**: How fast the ball is moving (accumulated gradients)
- **Gradient**: The slope pushing on the ball
- **Momentum**: How much the ball "remembers" its previous motion (typically 0.9 or 0.95)

```
Step 1:  velocity = 0.9 * 0 + grad₁ = grad₁
Step 2:  velocity = 0.9 * grad₁ + grad₂ = 0.9·grad₁ + grad₂
Step 3:  velocity = 0.9 * (0.9·grad₁ + grad₂) + grad₃ 
                  = 0.81·grad₁ + 0.9·grad₂ + grad₃
```

The velocity is an **exponential moving average** of past gradients!

### Why This Helps

1. **Noise Cancellation**: Random fluctuations average out
2. **Acceleration**: Consistent directions accumulate speed
3. **Dampening**: Wrong directions get corrected gradually

Visual comparison:

```
Vanilla SGD:                  Momentum:
    ↗↘↗↘↗↘                        ↓
    ↗↘↗↘↗↘                        ↓
    ↗↘↗↘↗↘     vs.               ↓
    ↗↘↗↘↗↘                        ↓
    (zigzag)                   (smooth)
```

### Code Example: Full Implementation

```python
class MomentumSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity for each parameter
        self.velocity = {p: torch.zeros_like(p.data) for p in parameters}
    
    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            
            # Get current gradient
            grad = p.grad.data
            
            # Update velocity: v = momentum * v + grad
            self.velocity[p] = self.momentum * self.velocity[p] + grad
            
            # Update parameter: θ = θ - lr * v
            p.data = p.data - self.lr * self.velocity[p]
    
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()
```

### Two Momentum Conventions: Unnormalized vs Normalized

**Important Note**: There are two common ways to implement momentum, and they behave differently!

**Unnormalized (shown above - simpler to teach):**
```python
velocity = momentum * velocity + gradient
theta = theta - lr * velocity
```

**Normalized (used in modern optimizers like PyTorch and our Muon implementation):**
```python
velocity = momentum * velocity + (1 - momentum) * gradient
theta = theta - lr * velocity
```

**Why the difference matters:**

The `(1 - momentum)` factor normalizes the velocity to keep it roughly the same magnitude regardless of the momentum value.

- **Without normalization**: In steady state with constant gradients `g`, velocity grows to `g / (1 - β)`, which explodes as β → 1
- **With normalization**: In steady state, velocity converges to `g`, independent of β

This means:
- **Unnormalized**: Changing momentum requires retuning learning rate (effective LR = lr / (1 - momentum))
- **Normalized**: Learning rate stays roughly constant across different momentum values (more robust)

**Example with momentum = 0.95:**
```python
# Unnormalized: velocity ≈ 20× the gradient in steady state (1/0.05 = 20)
v = 0.95·v + g  →  v_steady ≈ 20g

# Normalized: velocity ≈ 1× the gradient in steady state
v = 0.95·v + 0.05·g  →  v_steady ≈ g
```

**Which to use?**
- **Unnormalized**: Better for teaching concepts, used in older papers
- **Normalized**: Better for practice, used in modern implementations

Throughout the rest of this README, we use the **normalized** convention for actual code, since that's what PyTorch and Muon use.

### Typical Momentum Values

- `momentum = 0.0`: No momentum (vanilla SGD)
- `momentum = 0.9`: Standard choice (remembers ~10 steps)
- `momentum = 0.95`: High momentum (remembers ~20 steps)
- `momentum = 0.99`: Very high (remembers ~100 steps, can overshoot)

**Rule of thumb**: Higher momentum = smoother but potentially slower to change direction.

---

## 3. Nesterov Momentum: Look Before You Leap

### The Problem with Standard Momentum

Momentum is great, but it has a flaw: **it can overshoot**.

Imagine you're running down a hill with your eyes closed. You build up speed, and even when you reach the bottom, you keep running up the other side because of your velocity!

```
     ╲                    ╱
      ╲        ⚽        ╱   <- Ball overshoots the valley floor
       ╲    ↗           ╱
        ╲↗_____________╱
             (goal)
```

### Nesterov's Brilliant Idea

Instead of:
1. Compute gradient at current position
2. Update velocity
3. Move

Do this:
1. **"Look ahead"** using current velocity
2. Compute gradient at that lookahead position
3. Update velocity based on lookahead
4. Move

It's like opening your eyes and looking where you're about to go!

### The Math

Standard momentum:
```python
velocity = momentum * velocity + gradient(theta)
theta = theta - lr * velocity
```

Nesterov momentum:
```python
# Look ahead: where will momentum take us?
lookahead = theta - momentum * velocity

# Compute gradient at lookahead position
gradient_at_lookahead = gradient(lookahead)

# Update velocity using lookahead gradient
velocity = momentum * velocity + gradient_at_lookahead

# Update parameters
theta = theta - lr * velocity
```

### Practical Implementation Trick

Computing gradient at lookahead position is expensive (requires two gradient calculations). There's a clever rearrangement that PyTorch uses:

```python
class NesterovSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = {p: torch.zeros_like(p.data) for p in parameters}
    
    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            
            grad = p.grad.data
            v = self.velocity[p]
            
            # Update velocity (normalized form)
            # Note: This is (1-momentum)*grad for normalized momentum
            v = self.momentum * v + (1 - self.momentum) * grad
            self.velocity[p] = v
            
            # Nesterov update: use combination of velocity and gradient
            # This is mathematically equivalent to the lookahead form
            nesterov_grad = (1 - self.momentum) * grad + self.momentum * v
            p.data = p.data - self.lr * nesterov_grad
```

### Why Nesterov is Better

1. **Less overshooting**: Corrects trajectory before it's too late
2. **Faster convergence**: Especially in convex or near-convex regions
3. **Better for deep learning**: More stable in high-dimensional spaces

Visual comparison:

```
Standard Momentum:           Nesterov Momentum:
    ╲                ╱           ╲              ╱
     ╲    ↗⚽      ╱              ╲    ⚽      ╱
      ╲↗_________╱                 ╲__↓______╱
     (overshoots)                (corrects early)
```

---

## 4. Weight Decay: Keeping Parameters in Check

### What is Weight Decay?

Weight decay (also called L2 regularization) adds a penalty for large weights:

```
Loss_total = Loss_data + (λ/2) * Σ(θᵢ²)
             └─ fit data ┘   └─ keep weights small ┘
```

### Why Do We Need It?

**Problem**: Models can overfit by using huge weights that memorize training data.

**Solution**: Penalize large weights → forces model to find simpler solutions.

### Implementation: Two Ways

**Method 1: Add to Loss**
```python
# Add weight decay to loss
loss = data_loss + 0.5 * weight_decay * sum(p**2 for p in parameters)
```

**Method 2: Modify Gradient (more common)**
```python
# Equivalent: add weight decay directly to gradient
gradient = gradient + weight_decay * theta

# Then update
theta = theta - lr * gradient
     = theta - lr * gradient - lr * weight_decay * theta
     = (1 - lr * weight_decay) * theta - lr * gradient
```

See that `(1 - lr * weight_decay)`? That's why it's called "weight *decay*" - weights shrink toward zero each step!

### Weight Decay in Momentum

When combined with momentum, weight decay typically applied to the parameter directly:

```python
class SGDWithMomentumAndWeightDecay:
    def step(self):
        for p in self.parameters:
            grad = p.grad.data
            
            # Update velocity (without weight decay)
            self.velocity[p] = self.momentum * self.velocity[p] + grad
            
            # Update parameter with weight decay
            p.data = p.data - self.lr * (
                self.velocity[p] + self.weight_decay * p.data
            )
```

### Typical Values

- `weight_decay = 0.0`: No regularization
- `weight_decay = 0.01`: Light regularization
- `weight_decay = 0.1`: Standard for our setup
- `weight_decay = 1.0`: Heavy regularization (rare)

### Decaying Weight Decay Schedule

Modern practice: **decay the weight decay over training**!

```python
# Start with strong regularization, end with none
weight_decay_t = weight_decay_initial * (1 - t / T)
```

**Intuition**: 
- Early: Need regularization to prevent early overfitting
- Late: Let model learn freely to fit data perfectly

---

## 5. Putting It Together: Muon Optimizer

### What Makes Muon Special?

Muon combines:
1. **Nesterov momentum** (smart acceleration)
2. **Orthogonalization** (unique to Muon - keeps gradients "clean")
3. **Cautious weight decay** (adaptive regularization)

### Muon's Update Rule (Simplified)

```python
def muon_step(params, grads, state, lr=0.02, momentum=0.95, weight_decay=0.1):
    # Step 1: Nesterov momentum (normalized form)
    # Note the (1 - momentum) factor - this keeps velocity magnitude stable
    # regardless of momentum value. See "Two Momentum Conventions" above.
    velocity = momentum * velocity + (1 - momentum) * grads
    nesterov_grad = (1 - momentum) * grads + momentum * velocity
    
    # Step 2: Orthogonalize (this is Muon's secret sauce!)
    # Projects gradient onto orthogonal directions for better conditioning
    ortho_grad = orthogonalize(nesterov_grad)  # Newton-Schulz iteration
    
    # Step 3: Cautious weight decay
    # Only apply if update and weight agree on direction
    params = params - lr * ortho_grad - lr * weight_decay * params
    
    return params
```

**Why (1 - momentum) in Muon?**

Muon uses the **normalized momentum convention** for three key reasons:

1. **Stable learning rate**: When we tune `lr=0.02`, it works across different momentum values (0.85 → 0.95 warmup)
2. **Momentum warmup robustness**: As momentum increases from 0.85 to 0.95, the update magnitude stays consistent
3. **PyTorch compatibility**: Matches the convention used in `torch.optim.SGD(nesterov=True)`

Without the `(1 - momentum)` factor, increasing momentum from 0.85 to 0.95 would effectively reduce the learning rate by 3× (0.15 → 0.05), requiring complex LR compensation.

### Key Hyperparameters in Our Code

```python
# From config.py
matrix_lr = 0.02          # Learning rate for Muon
adam_beta1 = 0.8          # For AdamW (embedding/unembedding)
adam_beta2 = 0.95         # For AdamW
weight_decay = 0.1        # Base weight decay

# From trainer.py - Momentum warmup
def get_muon_momentum(step):
    """Warm up momentum from 0.85 to 0.95 over 300 steps"""
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95

# Why? Start cautious (0.85), then trust momentum more (0.95)

# From trainer.py - Weight decay schedule
def get_muon_weight_decay(step):
    """Decay weight decay from initial to 0 over training"""
    return weight_decay_scaled * (1 - step / total_steps)

# Why? Regularize early, let model learn freely later
```

### Why These Schedules?

**Momentum warmup (0.85 → 0.95)**:
- Step 0-100: `momentum ≈ 0.87` (responsive to gradients)
- Step 100-200: `momentum ≈ 0.92` (building confidence)
- Step 200-300: `momentum ≈ 0.95` (full trust in velocity)
- Step 300+: `momentum = 0.95` (stable)

**Weight decay decay (initial → 0)**:
- Early training: High WD = strong regularization = prevents memorization
- Late training: Low WD = weak regularization = allows perfect fit

---

## 6. Practical Tips and Intuitions

### When to Use What?

**Vanilla SGD**: 
- ❌ Almost never (too noisy)
- ✅ Maybe for tiny problems or pedagogical purposes

**SGD + Momentum**:
- ✅ Good baseline
- ✅ Works well for convex problems
- ⚠️ Can overshoot in tricky landscapes

**Nesterov Momentum**:
- ✅ Better than standard momentum in most cases
- ✅ Especially good for non-convex problems (deep learning!)
- ✅ Used in Muon

**AdamW** (not covered in detail, but worth mentioning):
- ✅ Adaptive learning rates per parameter
- ✅ Good for sparse gradients (NLP, especially embeddings)
- ⚠️ Can generalize worse than SGD+Momentum for vision tasks

**Muon**:
- ✅ Best for large matrix parameters (transformer weights)
- ✅ Combines Nesterov + orthogonalization + smart weight decay
- ⚠️ Requires tuning (but we've done that for you!)

### Debugging Tips

**Loss is noisy/unstable?**
- Try increasing momentum (0.9 → 0.95)
- Or decrease learning rate
- Or increase weight decay

**Loss plateaus early?**
- Decrease weight decay (less regularization)
- Or increase learning rate
- Or decrease momentum (be more responsive)

**Loss spikes/diverges?**
- Decrease learning rate (most common fix)
- Decrease momentum (be more cautious)
- Enable gradient clipping

**Training too slow?**
- Increase learning rate (carefully!)
- Decrease weight decay (if overly regularized)
- Check if using momentum at all

### Visualizing What's Happening

Imagine you're training a model. Here's what each optimizer "sees":

**Vanilla SGD**:
```
Step 1: grad points ↗, move ↗
Step 2: grad points ↘, move ↘  <- Oops! Changed direction
Step 3: grad points ↗, move ↗  <- Changed again!
Result: Zigzag, slow progress
```

**Momentum**:
```
Step 1: grad ↗, velocity ↗, move ↗
Step 2: grad ↘, velocity →, move →  <- Averaged out!
Step 3: grad ↗, velocity ↗, move ↗  <- Smooth curve
Result: Smooth path, fast progress
```

**Nesterov**:
```
Step 1: Look ahead ↗, see grad ↗, move ↗
Step 2: Look ahead →, see grad ↘, correct to →  <- Saw problem early!
Step 3: Look ahead ↗, see grad ↗, move ↗
Result: Even smoother, corrects overshoots
```

### The Big Picture

Training a neural network is like navigating a complicated landscape in the dark:
- **Gradient**: Your local sense of which way is downhill
- **Momentum**: Your memory of where you've been heading
- **Nesterov**: Looking slightly ahead before committing
- **Weight decay**: A rope pulling you back toward the origin (prevents wandering too far)

The art is balancing:
- **Exploration** (high learning rate, low momentum) vs. **Exploitation** (low LR, high momentum)
- **Regularization** (high weight decay) vs. **Capacity** (low weight decay)
- **Speed** (high LR) vs. **Stability** (low LR)

Our schedules (momentum warmup, weight decay decay) try to get the best of both worlds at different training stages!

---

## Summary: Key Takeaways

1. **Momentum = exponential moving average of gradients**
   - Smooths out noise, accelerates in consistent directions
   - Higher momentum = smoother but slower to adapt
   - **Important**: We use normalized form `v = β·v + (1-β)·g` to keep effective LR stable

2. **Nesterov = look before you leap**
   - Computes gradient at predicted next position
   - Reduces overshoot, faster convergence

3. **Weight decay = regularization**
   - Keeps parameters small, prevents overfitting
   - We decay it over training (strong → weak)

4. **Muon = Nesterov + Orthogonalization + Smart WD**
   - Uses momentum warmup (0.85 → 0.95)
   - Uses WD decay (initial → 0)
   - Best for transformer weight matrices

5. **Different parameter types need different LRs**
   - Embeddings: Fast (0.3) - sparse, need quick adaptation
   - Matrices: Medium (0.02) - dense, use Muon
   - Unembedding: Slow (0.004) - output distribution, stay stable

---

## Further Reading

**Papers**:
- Sutskever et al. (2013): "On the importance of initialization and momentum in deep learning"
- Dozat (2016): "Incorporating Nesterov Momentum into Adam" (Nadam paper)
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization" (AdamW paper)
- Jordan et al. (2024): "Muon: Momentum Orthogonalized by Newton-schulz" (Muon paper)

**Visualizations**:
- Alec Radford's optimizer visualization: distill.pub
- CS231n Stanford course notes on optimization

**Code**:
- PyTorch optimizers: `torch.optim`
- Our implementation: `src/gpt_2/muon.py`

---

*Happy optimizing! May your loss curves be smooth and your gradients be non-zero.* 🚀
