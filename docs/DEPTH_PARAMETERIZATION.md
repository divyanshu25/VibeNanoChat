# Depth Parameterization

A simple way to configure model architecture with a single parameter, borrowed from [nanochat](https://github.com/karpathy/nanochat).

## The Problem

Normally you'd specify transformer architecture with 3+ numbers:

```python
n_layer = 12    # how many transformer blocks
n_embed = 768   # hidden dimension
n_head = 6      # number of attention heads
```

This gets annoying when you want to sweep model sizes for scaling law experiments. You'd have to manually configure each one:

```python
# 6 different model sizes means 6 different configs
n_layer=6,  n_embed=384,  n_head=3
n_layer=8,  n_embed=512,  n_head=4
n_layer=10, n_embed=640,  n_head=5
n_layer=12, n_embed=768,  n_head=6
n_layer=14, n_embed=896,  n_head=7
n_layer=16, n_embed=1024, n_head=8
```

## The Solution

Just use one number - `depth`. Everything else is calculated automatically:

```python
depth = 12  # that's it, you're done
```

This sets:
```python
n_layer = depth                    # 12 layers
n_embed = depth × 64               # 768 dimensions  
n_head = n_embed / 128             # 6 heads
```

The formula is simple. `n_layer` equals `depth`. `n_embed` is `depth` times an aspect ratio (default 64). `n_head` is `n_embed` divided by head dimension (default 128).

## Quick Reference

Here's what you get with different depths (aspect_ratio=64, head_dim=128):

```
depth=6   -> 6 layers,  384 dims,  3 heads   (~30M params)
depth=8   -> 8 layers,  512 dims,  4 heads   (~60M params)
depth=10  -> 10 layers, 640 dims,  5 heads   (~100M params)
depth=12  -> 12 layers, 768 dims,  6 heads   (~150M params)
depth=14  -> 14 layers, 896 dims,  7 heads   (~210M params)
depth=16  -> 16 layers, 1024 dims, 8 heads   (~280M params)
depth=20  -> 20 layers, 1280 dims, 10 heads  (~560M params)
```

## Usage

Train with a specific depth from command line:

```bash
make ddp-train NGPUS=2 DEPTH=12 TARGET_FLOPS=1e18
```

Or with torchrun directly:

```bash
torchrun --standalone --nproc_per_node=2 src/gpt_2/ddp.py \
    --mode pretraining \
    --depth 12 \
    --target-flops 1e18
```

You can also set it in the config file (`src/gpt_2/config.py`):

```python
depth: int = 12
```

The config file also lets you change the aspect ratio and head dimension if you want different proportions:

```python
depth: int = 12
aspect_ratio: int = 64    # n_embed = depth * aspect_ratio
head_dim: int = 128       # n_head = n_embed / head_dim
```

## Scaling Laws

This is where depth parameterization really shines. You can sweep model sizes with a simple loop:

```bash
for depth in 6 8 10 12 14; do
    make ddp-train DEPTH=$depth TARGET_FLOPS=1e18
done
```

Or sweep both model size and compute budget (Chinchilla-style):

```bash
for flops in 1e18 3e18 6e18; do
    for depth in 6 8 10 12 14; do
        make ddp-train DEPTH=$depth TARGET_FLOPS=$flops
    done
done
```

There's also a Make target that does this automatically:

```bash
make run-scaling-law
```

This runs 15 experiments (5 depths × 3 FLOP budgets).

## Hyperparameter Scaling

An important detail: when you change model size, you should also change the learning rate, weight decay, and batch size. Otherwise small models train with too-small LR and large models with too-large LR (or OOM).

The code does this automatically when you use depth mode:

**Learning Rate:** scales as `1/√n_embed`

```python
lr_scale = (n_embed / 768) ** -0.5
```

This was tuned at depth=12 (n_embed=768). Smaller models get higher LR, larger models get lower LR.

**Weight Decay:** scales as `1/depth²`

```python
wd_scale = (12 / depth) ** 2
```

This was also tuned at depth=12 with WD=0.10. The scaling comes from experiments in nanochat that showed optimal weight decay follows a `1/width²` law.

**Batch Size:** scales based on model size for optimal throughput and memory usage

```python
# Reference: tuned on 2x H100 80GB
if depth <= 8:       batch_size = 64   # maximize throughput for small models
elif depth <= 10:    batch_size = 32   # high throughput
elif depth <= 14:    batch_size = 16   # balanced
elif depth <= 18:    batch_size = 8    # conservative for large models
elif depth <= 22:    batch_size = 4    # tight fit
else:                batch_size = 2    # minimal for huge models
```

The total batch size (524,288 tokens) stays constant - gradient accumulation automatically adjusts. So training quality is the same across all model sizes. Small models get higher batch_size for faster training, large models get lower batch_size to avoid OOM.

You don't have to do anything - these all scale automatically when you specify `--depth`. But it's good to know what's happening under the hood.

## Implementation

The calculation happens in `GPTConfig.__post_init__()` in `src/gpt_2/config.py`:

```python
if self.depth > 0:
    self.n_layer = self.depth
    base_dim = self.depth * self.aspect_ratio
    
    # Round up to nearest multiple of head_dim
    self.n_embed = ((base_dim + self.head_dim - 1) // self.head_dim) * self.head_dim
    
    self.n_head = self.n_embed // self.head_dim
    self.n_kv_head = self.n_head
```

The rounding step ensures `n_embed` is divisible by `head_dim`. This matters for Flash Attention which wants clean divisions. For most depths with aspect_ratio=64 and head_dim=128, the rounding is a no-op. But for odd depths like 7 or 9, you get a small "nudge" upward.

The hyperparameter scaling happens in `src/gpt_2/trainer.py`:
- Learning rate scaling in `_setup_hyperparameters()`
- Weight decay scaling in `_setup_optimizer_and_checkpoint()`

These kick in automatically when you use `--depth` from the command line.

## Advanced

You can override the aspect ratio and head dimension if you want different proportions:

```bash
# Narrower model at same depth
make ddp-train DEPTH=12 ASPECT_RATIO=48 HEAD_DIM=64

# Wider model at same depth  
make ddp-train DEPTH=12 ASPECT_RATIO=80 HEAD_DIM=128
```

This changes the model_dim calculation but keeps the same depth. Useful if you want to explore width vs depth tradeoffs.

## Notes

The implementation is in `src/gpt_2/config.py` (`__post_init__` method) and `src/gpt_2/trainer.py` (hyperparameter scaling). When you run training with `--depth`, you'll see output like:

```
🔢 Using depth-based architecture: depth=12

📐 DEPTH-AWARE SCALING (depth=12)
   n_layer: 12
   n_embed: 768 (base: 768, nudge: +0)
   n_head: 6
   head_dim: 128
   LR scaling: 1.000000 (∝ 1/√(768/768))
   Max LR: 6.00e-04 → 6.00e-04
   WD scaling: 1.000000 (∝ (12/12)²)
   Weight decay: 0.100000 → 0.100000
```

This confirms everything was calculated correctly.

The idea for depth parameterization comes from [nanochat](https://github.com/karpathy/nanochat). The scaling laws for learning rate and weight decay come from empirical experiments documented in nanochat's development log.

If you're doing scaling law experiments (Chinchilla-style), this parameterization makes the sweeps much cleaner. Instead of managing multiple config files or complex scripts, you just loop over depth and FLOP budget.
