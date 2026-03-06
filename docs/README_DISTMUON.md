# DistMuonAdamW: Distributed Hybrid Optimizer

This is a guide to understanding how `DistMuonAdamW` works. It's a distributed optimizer that combines two different optimization algorithms: Muon for most of the model's weights, and AdamW for embeddings. The interesting part is that it achieves this without using PyTorch's DDP wrapper, giving us more direct control over the training process.

## Why a Hybrid Optimizer?

Let's start with the observation that not all parameters in a transformer are created equal. Look at what we actually have:

**2D weight matrices (most of the model):**
- Attention projections: Q, K, V, O
- MLP layers: up-projection, down-projection
- These are big matrices that benefit from gradient orthogonalization

**1D parameters (the rest):**
- Token embeddings (`wte`)
- Language model head (`lm_head`)
- Note: We use RoPE instead of learned position embeddings, and RMSNorm has no learnable parameters

The key insight is that the 2D weight matrices benefit from a specialized optimizer called **Muon**, which orthogonalizes gradients. This gives you more stable training and better conditioning through deep networks. But for embeddings and the language model head, standard **AdamW** works just fine.

## The Distribution Strategy

Training on multiple GPUs introduces a memory problem. If each GPU stores a complete copy of the optimizer state (like in standard DDP), you're not actually saving any memory - you're just replicating everything. 

Instead, we shard the optimizer state across GPUs using a strategy similar to DeepSpeed's ZeRO-2. If you have 4 GPUs, each GPU only stores 1/4 of the optimizer state. This cuts optimizer memory by `1/world_size`.

The algorithm is actually quite simple:
1. **Reduce-scatter**: Each GPU gets its assigned chunk of gradients (averaged across GPUs)
2. **Local compute**: Each GPU updates its chunk of parameters (all GPUs work in parallel)
3. **All-gather**: Each GPU broadcasts its updated parameters back to everyone

No DDP wrapper needed. We handle the gradient synchronization ourselves, which gives us cleaner code and more control.

---

## Step 1: Grouping Parameters by Optimizer Type

First, we need to decide which parameters get Muon and which get AdamW. The rule is straightforward: look at the shape.

```python
# Your model has these kinds of parameters:
wte                   # Token embeddings    → AdamW
lm_head               # Language model head → AdamW
                      # (no wpe: using RoPE instead)
                      # (RMSNorm is parameter-free)
attn.c_attn           # Attention QKV       → Muon (2D)
attn.c_proj           # Attention output    → Muon (2D)
mlp.c_fc              # MLP up-projection   → Muon (2D)
mlp.c_proj            # MLP down-projection → Muon (2D)
```

The rule: if it's a 2D weight matrix, use Muon. If it's embeddings or the language model head, use AdamW. Note that RMSNorm has no learnable parameters (it's just a normalization operation), so there's nothing to optimize there.

---

## Step 2: Batching Same-Shape Parameters (Muon Only)

Here's where we get a significant speedup. Consider that in a 12-layer transformer, you have 12 separate `mlp.c_fc` matrices, all with the exact same shape `(768, 3072)`. You could update them one at a time:

```python
# Updating each layer separately means 12 kernel launches:
for layer in range(12):
    update(model.layers[layer].mlp.c_fc)  # (768, 3072)
```

But this is wasteful. Each kernel launch has overhead, and we're not taking advantage of the GPU's parallel processing capabilities. Instead, we can stack all same-shaped parameters into a single tensor and update them all at once:

```python
# Stack into a single 3D tensor and update in one kernel launch:
stacked = torch.stack([layer.mlp.c_fc for layer in model.layers])  # (12, 768, 3072)
muon_step_fused(stacked)  # Single compiled kernel processes all 12 layers
```

This is the key optimization that makes Muon fast. By batching same-shape parameters together, we go from 12 kernel launches to 1. When you combine this with `@torch.compile`, PyTorch can fuse all the operations into a single, highly optimized kernel.

---

## Step 3: Distributed Training (3 Phases)

Now let's talk about how this works across multiple GPUs. The goal is to shard the optimizer state so each GPU only stores a fraction of it, while still keeping all the model parameters synchronized.

### Phase 1: Reduce-Scatter (Distributing the Gradients)

After you call `loss.backward()`, every GPU has a complete set of gradients for all parameters. But we don't want every GPU to update all parameters - that would mean storing the full optimizer state on each GPU, which defeats the purpose.

Instead, we use reduce-scatter to both average the gradients across GPUs *and* split them up so each GPU gets a different chunk. With 4 GPUs and 12 layers, this looks like:

```
GPU 0: Gets averaged gradients for layers 0-2   (owns 3/12 layers)
GPU 1: Gets averaged gradients for layers 3-5   (owns 3/12 layers)
GPU 2: Gets averaged gradients for layers 6-8   (owns 3/12 layers)
GPU 3: Gets averaged gradients for layers 9-11  (owns 3/12 layers)
```

This is the key to memory savings. Each GPU only needs to store optimizer state (momentum buffers, variance estimates, etc.) for 1/4 of the parameters. The memory usage for optimizer state goes from `N` bytes per GPU to `N/4` bytes per GPU.

### Phase 2: Local Compute (Parallel Updates)

Now each GPU independently updates its assigned chunk of parameters. This is where the actual optimization happens. Since each GPU is working on different parameters, they can all compute in parallel with no communication overhead.

Here's what happens on each GPU (let's say GPU 0, which owns layers 0-2):

```python
def muon_step_fused(grads, params, momentum, variance, lr, wd):
    # 1. Nesterov momentum - smooth out gradient updates
    momentum = 0.95 * momentum + 0.05 * grads
    g = grads + 0.95 * momentum
    
    # 2. Polar Express orthogonalization (5 iterations)
    #    This projects the gradient onto the nearest orthogonal matrix
    #    Stabilizes training by reducing correlations in the gradient
    g = orthogonalize(g)
    
    # 3. Variance reduction - adaptive learning rates per row/column
    #    Similar to Adam but more memory efficient (factored)
    variance = 0.95 * variance + 0.05 * g²
    g = g / sqrt(variance + eps)
    
    # 4. Cautious weight decay - only decay when gradient agrees with weight
    #    This prevents weight decay from fighting the gradient
    mask = (g * params) >= 0
    params -= lr * g + lr * wd * params * mask
```

All 4 GPUs are doing this at the same time, each working on their own chunk. No communication needed during this phase, which means the compute can proceed at full speed.

### Phase 3: All-Gather (Synchronizing Parameters)

At this point each GPU has updated its chunk of parameters, but the other GPUs don't have these updates yet. We need to synchronize so all GPUs have the same complete model.

This is done with all-gather, where each GPU broadcasts its updated chunk to all other GPUs:

```
GPU 0 broadcasts its updated layers 0-2  →  All GPUs receive them
GPU 1 broadcasts its updated layers 3-5  →  All GPUs receive them
GPU 2 broadcasts its updated layers 6-8  →  All GPUs receive them
GPU 3 broadcasts its updated layers 9-11 →  All GPUs receive them
```

After this phase completes, all GPUs have the complete, synchronized model. They're ready to start the next forward pass.

---

## Putting It All Together: Visual Flow

Here's how the whole process flows from start to finish:

```
┌────────────────────────────────────────────────────────────┐
│                     DistMuonAdamW                          │
└────────────────────────────────────────────────────────────┘

1. Group Parameters by Type
   ├─ AdamW group: token embeddings, lm_head
   └─ Muon group:  all 2D weight matrices (attention, MLP)

2. Stack Same-Shaped Parameters
   Example: 12 layers × (768, 3072) → single (12, 768, 3072) tensor

3. PHASE 1: Reduce-Scatter
   ┌─────────────────┐
   │ Full Gradients  │  (12, 768, 3072)
   └────────┬────────┘
            │ split + reduce_scatter
            │ (average across GPUs and distribute chunks)
            ▼
   ┌──┬──┬──┬──┐
   │G0│G1│G2│G3│  each GPU gets its chunk
   └──┴──┴──┴──┘

4. PHASE 2: Compute (all GPUs work in parallel)
   GPU0      GPU1      GPU2      GPU3
   ┌──┐      ┌──┐      ┌──┐      ┌──┐
   │G0│      │G1│      │G2│      │G3│
   └──┘      └──┘      └──┘      └──┘
    │         │         │         │
    │ muon    │ muon    │ muon    │ muon
    │ step    │ step    │ step    │ step
    ▼         ▼         ▼         ▼
   ┌──┐      ┌──┐      ┌──┐      ┌──┐
   │P0│      │P1│      │P2│      │P3│
   └──┘      └──┘      └──┘      └──┘
   updated   updated   updated   updated

5. PHASE 3: All-Gather
   ┌──┬──┬──┬──┐
   │P0│P1│P2│P3│  all GPUs broadcast their chunks
   └──┴──┴──┴──┘
            │
            ▼
   ┌─────────────────┐
   │ Full Parameters │  (12, 768, 3072)
   │  (synchronized) │  ← all GPUs now have identical state
   └─────────────────┘
```

---

## Memory Savings: A Concrete Example

Let's look at actual numbers for a 124M parameter model trained on 4 GPUs.

**Without optimizer sharding (standard DDP approach):**

Each GPU stores a complete copy of everything:
```
Model parameters:     496 MB  (replicated on each GPU)
Optimizer state:      544 MB  (replicated on each GPU)
                      ──────
Per GPU total:        1040 MB
Total across 4 GPUs:  4160 MB
```

**With DistMuon (ZeRO-2 style optimizer sharding):**

Each GPU stores the full model but only 1/4 of the optimizer state:
```
Model parameters:     496 MB  (replicated on each GPU)
Optimizer state:      136 MB  (sharded: 544 MB ÷ 4)
                      ──────
Per GPU total:        632 MB
Total across 4 GPUs:  2528 MB
```

**Result:** We save 39% of total memory. This means you can either train bigger models, use larger batch sizes, or get away with cheaper GPUs with less memory.

---

## Understanding the Muon Algorithm

Let's dig deeper into what Muon is actually doing. The input is a batch of gradients with shape `(N, rows, cols)`, where `N` is the number of same-shaped parameters we're updating together (e.g., 12 for the 12 layers of MLP projections).

### Step 1: Nesterov Momentum

Momentum is a classic technique for smoothing out noisy gradients. Instead of just using the current gradient, we maintain a running average. Nesterov momentum adds a "lookahead" step - we look at where the momentum is taking us, not just where we are now.

```python
momentum_buffer = β * momentum_buffer + (1-β) * G
G = (1-β) * G + β * momentum_buffer  # Nesterov lookahead
```

We use β = 0.95, which means we keep 95% of the old momentum and add 5% of the new gradient. This is fairly heavy momentum, similar to what you'd use with SGD.

### Step 2: Polar Express Orthogonalization

This is where Muon gets interesting. We project the gradient onto the nearest orthogonal matrix. Why? Orthogonal matrices have special properties: they preserve norms, reduce correlations between dimensions, and improve conditioning. In practice, this means more stable gradient flow through deep networks.

The algorithm is called "Polar Express" and uses 5 iterations to approximate the polar decomposition:

```python
X = G / (||G|| * 1.02)  # normalize with slight scaling

# Iterate 5 times to converge to nearest orthogonal matrix
for (a, b, c) in polar_express_coeffs:
    A = X @ X.T            # Gram matrix (measures how far from orthogonal)
    B = b*A + c*(A @ A)    # polynomial correction term
    X = a*X + B @ X        # update X toward orthogonal matrix

G = X  # now approximately orthogonal
```

Think of it this way: the gradient tells you which direction to move in parameter space. By orthogonalizing it, we're ensuring that movement along different dimensions doesn't interfere - it's more like a clean rotation than a stretching or shearing operation.

### Step 3: Variance Reduction (Adaptive Learning Rates)

Just like Adam uses adaptive learning rates for each parameter, Muon adapts the learning rate. But there's a clever trick: instead of storing a variance estimate for every single element (which would use tons of memory), we only store it per row or per column.

```python
v_mean = G².mean(dim=row_or_col)  # Average over row or column
variance = β₂ * variance + (1-β₂) * v_mean

scale = 1 / sqrt(variance + eps)
G = G * scale  # Scale gradient (norm is preserved)
```

For a matrix of shape `(768, 3072)`, full Adam would store 768 × 3072 variance values. Muon only stores either 768 values (one per row) or 3072 values (one per column). This is the factored variance trick - we get most of the benefit of adaptive learning rates with a tiny fraction of the memory cost.

### Step 4: Cautious Weight Decay

Standard weight decay always pulls weights toward zero. But sometimes this fights against what the gradient is trying to do, creating unnecessary oscillations. Cautious weight decay only applies decay when the gradient and weight have the same sign - meaning they're pointing in the same direction.

```python
mask = (G * W) >= 0  # True where gradient and weight agree in sign
W = W - lr*G - lr*wd*W*mask
```

If a weight is positive and the gradient is positive (both pointing "up"), we apply weight decay. If a weight is positive but the gradient is negative (they disagree), we skip weight decay for that element. This lets the gradient do its job without interference, leading to smoother training dynamics.

---

## AdamW Parameters: A Simpler Strategy

For embeddings and the language model head, we use standard AdamW. The distribution strategy depends on the size of the parameter.

**Small parameters (< 1024 elements):**

If a parameter is small (like biases, if they existed), sharding doesn't help much. The communication overhead would outweigh any memory savings. So we just replicate these on all GPUs:

```python
dist.all_reduce(grad)  # All GPUs get the same averaged gradient
# Each GPU maintains its own copy of the optimizer state
```

**Large parameters (≥ 1024 elements):**

For large parameters like embeddings `(50257, 768)` and `lm_head` `(50257, 768)`, we use the same sharding strategy as Muon:

```python
dist.reduce_scatter(grad)  # Each GPU gets its slice of gradients
adamw_step(local_slice)    # Update locally with standard AdamW
dist.all_gather(updated_slice)  # Broadcast back to all GPUs
```

The pattern is identical to Muon, but we use the standard AdamW update (momentum + adaptive learning rates) instead of orthogonalization.

---

## How the Code is Structured

The optimizer's `step()` method orchestrates the three phases we talked about. Here's a simplified view:

```python
class DistMuonAdamW:
    def step(self):
        # Phase 1: Launch all reduce-scatter operations asynchronously
        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'muon':
                future = self._reduce_muon(group)
            else:
                future = self._reduce_adamw(group)
            reduce_infos.append(future)
        
        # Phase 2: Wait for gradients, compute updates, launch all-gathers
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            info['future'].wait()  # Block until this group's gradients arrive
            
            if group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list)
            else:
                self._compute_adamw(group, info, gather_list)
        
        # Phase 3: Wait for all-gathers to complete
        for gather_info in gather_list:
            gather_info['future'].wait()
            # Copy updated parameters back to model
```

The key insight here is that we launch all the reduce-scatter operations first, then wait and compute, then launch all the all-gathers. This overlap of communication and computation keeps the GPUs busy and reduces idle time.

---

## What Gets Stored in Optimizer State

It's useful to understand exactly what state we're storing and where.

**For Muon parameters (stored per parameter group, sharded across GPUs):**

Each GPU stores buffers only for its assigned chunk:
```python
momentum_buffer:         (chunk_size, rows, cols)  # First moment (Nesterov)
second_momentum_buffer:  (chunk_size, rows, 1)     # Second moment (factored variance)
                      or (chunk_size, 1, cols)     # depends on which dimension we factor
```

The key thing is that these are sharded - if you have 4 GPUs, each stores 1/4 of these buffers.

**For AdamW parameters (stored per parameter, sharded for large params):**

For each parameter, we store standard Adam state:
```python
exp_avg:    (param_shape)  # First moment (momentum)
exp_avg_sq: (param_shape)  # Second moment (variance)
```

Again, for large parameters (like embeddings), each GPU only stores its chunk of these buffers.

---

## Hyperparameters

These are the hyperparameters we've found to work well through experimentation (see `docs/README_MUON.md` for details on how these were tuned):

```python
# Muon parameters (for 2D weight matrices)
matrix_lr = 0.02          # Learning rate - higher than typical because gradients are normalized
momentum = 0.95           # Heavy momentum (95%) smooths out training
weight_decay = 0.1        # Fairly aggressive decay, but "cautious" so it doesn't fight gradients
ns_steps = 5              # 5 iterations of Polar Express is enough to approximate orthogonalization
beta2 = 0.95              # Variance decay rate (like Adam's second moment)

# AdamW parameters (for embeddings and lm_head)
embedding_lr = 0.2        # Much higher LR - embeddings can handle it
betas = (0.8, 0.95)       # Standard Adam-style momentum and variance
```

An important caveat: these hyperparameters are scale-dependent. What works well for a 124M parameter model might need adjustment at 1B parameters. The learning rates in particular often need to be tuned based on model size and batch size.

---

## Why This Works

Let's step back and think about why this combination is effective.

**The Muon optimizer brings three main benefits:**

1. **Gradient orthogonalization** keeps training stable through deep networks. By projecting gradients onto orthogonal matrices, we prevent different dimensions from interfering with each other during optimization.

2. **Factored variance** gives us adaptive learning rates (like Adam) but uses way less memory. Instead of storing variance for every parameter element, we only store it per row or column.

3. **Batched updates** let us update many same-shaped parameters in a single fused kernel. This eliminates Python overhead and lets the GPU work efficiently.

**The distribution strategy brings its own benefits:**

1. **Optimizer state sharding** cuts memory usage by `1/world_size`. This means you can train larger models or use bigger batch sizes with the same hardware.

2. **Asynchronous operations** let communication overlap with computation. While one GPU is computing updates, another can be transferring data. This keeps all the GPUs busy.

3. **No DDP wrapper** means we have direct control over the training loop. The code is cleaner and easier to understand, and we can implement optimizations that wouldn't be possible with DDP's abstractions.

Put it all together and you get fast, memory-efficient, stable training at scale.

---

## Comparison to Other Approaches

It's worth understanding where DistMuon fits in the landscape of distributed training strategies.

**Standard DDP + AdamW** is simple but doesn't shard optimizer state. Each GPU stores the full optimizer state, so you're not saving any memory - just replicating everything. Communication is straightforward (all-reduce on gradients), and the code is simple, but you hit memory limits quickly.

**FSDP (Fully Sharded Data Parallel)** shards both the model and optimizer state. This saves more memory than our approach, but it's complex to implement and reason about. You need to manage when parameters are available, handle resharding, etc.

**DeepSpeed ZeRO-2** shards optimizer state (like we do) but is a very complex framework with many abstraction layers. It's powerful but can be hard to debug and customize.

**DistMuon** sits in a sweet spot. We get ZeRO-2 style memory savings for optimizer state, but the implementation is cleaner and more direct. We shard optimizer state but keep the model replicated, which means we don't need complex parameter management. The code is understandable and hackable.

| Approach | Optimizer Memory | Model Memory | Code Complexity |
|----------|------------------|--------------|-----------------|
| DDP + AdamW | Full (replicated) | Full (replicated) | Simple |
| DeepSpeed ZeRO-2 | Sharded | Full (replicated) | Very complex |
| DistMuon | Sharded | Full (replicated) | Medium |
| FSDP | Sharded | Sharded | Complex |

---

## Implementation Details Worth Understanding

There are a few implementation tricks worth knowing about if you're reading or modifying the code.

### Why use `@torch.compile`?

Both the Muon and AdamW update functions are wrapped with `@torch.compile`:

```python
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(...):
    # All 4 steps (momentum, orthogonalization, variance, weight decay)
    # get fused into a single optimized kernel
```

Without compilation, PyTorch would execute each operation separately with Python overhead between them. With compilation, PyTorch traces through the entire function and generates a single, highly optimized kernel. This is orders of magnitude faster.

### Why use 0-D CPU tensors for hyperparameters?

You'll see things like:

```python
self._muon_lr_t = torch.tensor(0.0, device='cpu')
```

This is a trick to prevent recompilation. If we passed the learning rate as a Python float, changing it would cause `torch.compile` to regenerate the entire kernel. By storing it in a tensor, we can update the tensor's value without recompilation. The graph stays the same, only the data changes.

### Why use factored variance?

The memory savings from factored variance are dramatic. Compare:

```python
# Full Adam approach: store second moment for every element
second_momentum_buffer = torch.zeros(N, rows, cols)  # 4N·rows·cols bytes

# Muon approach: store only per row or per column
second_momentum_buffer = torch.zeros(N, rows, 1)     # 4N·rows bytes
```

For a typical `(768, 3072)` matrix, this goes from 9.4 MB down to 12 KB - a 99.9% memory reduction. Across all the parameters in a large model, this adds up significantly.

---

## Debugging and Profiling

Here are some useful techniques for debugging and understanding what's happening during training.

**Verify that gradients are synchronized:**

After calling `optimizer.step()`, all GPUs should have identical parameters. You can check this:

```python
# On GPU 0, check if parameters match GPU 1
assert torch.allclose(model.layer[0].weight, 
                     other_gpu_model.layer[0].weight, rtol=1e-5)
```

If this fails, something went wrong in the all-gather phase.

**Monitor memory usage:**

Keep an eye on how much memory you're actually using:

```python
import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved:  {reserved:.2f} GB")
```

The "allocated" number is what you're actually using. "Reserved" includes PyTorch's memory cache.

**Profile communication overhead:**

To see where time is being spent, use PyTorch's profiler:

```python
with torch.profiler.profile() as prof:
    optimizer.step()
table = prof.key_averages().table(sort_by="cuda_time_total")
print(table)
```

Look for entries like `reduce_scatter` and `all_gather`. If these are taking a large fraction of time, you might have communication bottlenecks (check your network hardware).

---

## References

- **Muon optimizer:** [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- **Polar Express paper:** [arxiv.org/pdf/2505.16932](https://arxiv.org/pdf/2505.16932)
- **modded-nanogpt:** [github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- **nanochat:** [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)

---

## Summary

If you want the short version:

1. We use different optimizers for different parameter types: Muon for 2D weight matrices (most of the model), AdamW for embeddings and the language model head.

2. We batch same-shaped parameters together and update them in one fused kernel operation. This is way faster than updating them individually.

3. We shard optimizer state across GPUs so each GPU only stores 1/N of it. This cuts optimizer memory by a factor of N.

4. The distributed algorithm has three phases: reduce-scatter (distribute gradients), compute (update parameters locally), and all-gather (synchronize parameters).

5. The result is fast, memory-efficient training with direct control over the training loop - no need for PyTorch's DDP wrapper.

---

*Implementation: `src/gpt_2/muon.py` (lines 337-594)*  
*Created: 2026-01-29*
