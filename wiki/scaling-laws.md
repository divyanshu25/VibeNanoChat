---
title: Scaling laws
tags: [scaling, flops, depth, isoflop, chinchilla]
---

# Scaling laws

> [!ABSTRACT]
> Empirical methodology for finding the optimal model size for a given compute budget. Based on 46+ training runs across model sizes from 77M to 522M parameters.

---

## Contents

- [[#The core question]]
- [[#C = 6ND]]
- [[#Depth parameterization]]
- [[#Running the experiments]]
- [[#The loss scaling law]]
- [[#Optimal allocation]]
- [[#Isoflop curves]]
- [[#Comparison to Chinchilla]]
- [[#Practical guidance]]
- [[#Scripts reference]]

---

## The core question

Given a fixed compute budget C (FLOPs), how should you split it between model size N (parameters) and training data D (tokens)?

---

## C = 6ND

Training compute is approximated as:

```
C = 6 × N × D
```

| Symbol | Meaning |
|--------|---------|
| `C` | Total training FLOPs |
| `N` | Number of model parameters |
| `D` | Number of training tokens |
| `6` | Forward (2×) + backward (4×) per parameter |

### Empirical accuracy

Verified across 46+ runs. The ratio `C_actual / (6ND)` averages **1.001** — essentially perfect.

| Compute budget | Ratio (actual / 6ND) |
|---------------|---------------------|
| 1e18 FLOPs | 0.998 (−0.24%) |
| 2e18 FLOPs | 0.999 (−0.12%) |
| 3e18 FLOPs | 0.998 (−0.23%) |
| 6e18 FLOPs | 1.009 (+0.94%) |

> [!TIP] Practical use
> Given a compute budget and model size, solve for affordable tokens: `D = C / (6 × N)`
>
> Example: 1e19 FLOPs, 500M params → `D = 1e19 / (6 × 500M) = 3.33B tokens`

---

## Depth parameterization

The single `depth` integer controls the entire model size. This makes sweeping model sizes trivial — change one number, everything else auto-adjusts.

```python
n_layer = depth
n_embed = depth × aspect_ratio   # rounded up to multiple of head_dim
n_head  = n_embed // head_dim
```

Defaults: `aspect_ratio = 64`, `head_dim = 64`

| `depth` | Layers | Embed dim | Heads | Approx params |
|---------|--------|----------|-------|--------------|
| 8 | 8 | 512 | 8 | ~77M |
| 10 | 10 | 640 | 10 | ~117M |
| 12 | 12 | 768 | 12 | ~154M |
| 14 | 14 | 896 | 14 | ~210M |
| 16 | 16 | 1024 | 16 | ~270M |
| 18 | 18 | 1152 | 18 | ~341M |
| 20 | 20 | 1280 | 20 | ~522M |

> [!NOTE] Weight decay auto-scaling
> Weight decay scales as `WD × (12/depth)²`, so hyperparameters don't need retuning when you change `depth`. See [[training#Batch scaling]].

---

## Running the experiments

```bash
# Train 6 depths × 4 FLOP budgets = 24 runs
make run-scaling-law

# Fit curves and plot optima
uv run python scripts/plot_isoflop_curve.py

# Verify C = 6ND formula
uv run python scripts/verify_flops_formula.py
```

The sweep trains models at depths N8, N10, N12, N14, N16, N18, N20 across compute budgets 1e18, 2e18, 3e18, 6e18 FLOPs. Results are saved as `scaling_laws_N*_F*.log` files.

---

## The loss scaling law

Training models at 7 depths with a fixed 10:1 data:param ratio and plotting final validation BPB vs compute reveals a near-perfect power law:

```
L(C) = 73.12 × C^(−0.122) + 0.485
```

| Term | Value | Meaning |
|------|-------|---------|
| Coefficient | 73.12 | Scale factor |
| Exponent | −0.122 | How fast loss improves with compute |
| Irreducible loss | 0.485 BPB | Theoretical minimum for this architecture + data |
| R² | 0.999951 | Fit quality (essentially perfect) |

### What the exponent means

| Compute change | Effect on excess loss |
|---------------|----------------------|
| 2× more compute | ~8% improvement |
| 10× more compute | ~24% improvement |
| To halve excess loss | Need ~293× more compute |

### The irreducible floor

The 0.485 BPB constant is real. With infinite compute and this setup, you'd asymptote here. Getting below it requires changing the architecture, data quality, or tokenizer — not more compute.

> [!NOTE] Our exponent vs. prior work
> Our exponent (−0.122) is better than typical: GPT-3 scaling laws (Kaplan et al.) found ~−0.05 to −0.076; Chinchilla found ~−0.05. Likely due to DistMuon's efficiency and FineWeb-Edu's quality.

---

## Optimal allocation

Fitting power laws to optimal models across four compute budgets:

```
N_optimal ∝ C^0.456   (≈ √C)
D_optimal ∝ C^0.544   (≈ √C)
```

Both scale at roughly **√C** — as you double compute, scale both model size and data by ~√2.

### Empirical optimal points

| Compute budget | Optimal N | Optimal D | Tokens/param |
|---------------|----------|----------|-------------|
| 1e18 FLOPs | ~154M | ~1.1B | 7.05 |
| 2e18 FLOPs | ~210M | ~1.6B | 7.5 |
| 3e18 FLOPs | ~270M | ~1.9B | 7.1 |
| 6e18 FLOPs | ~341M | ~2.9B | 8.61 |

---

## Isoflop curves

At a fixed compute budget, plotting validation BPB vs model size produces a **U-shaped curve**:

- Too small → underfitting (not enough capacity)
- Too large → not enough training tokens (overfitting)
- The minimum → optimal model size for that budget

The optimal point shifts right (bigger models) as compute increases.

---

## Comparison to Chinchilla

Chinchilla (Hoffmann et al., 2022) recommends 20 tokens/param. Our results suggest 7–9 tokens/param.

| Factor | Explanation |
|--------|-------------|
| Different scale | Chinchilla studied 400M–70B params; we're at 77M–522M |
| Better optimizer | DistMuon extracts more from fewer tokens than AdamW |
| Higher quality data | FineWeb-Edu > raw Common Crawl |
| Empirical nature | Scaling laws depend on architecture, optimizer, and data |

> [!WARNING]
> Scaling laws are empirical, not theoretical. Our laws apply to *our* setup. If you change the optimizer, data, or architecture, re-run the experiments.

---

## Practical guidance

**Planning a training run:**

1. Decide your compute budget `C` (FLOPs)
2. Use `N_optimal ∝ C^0.456` to find the right model size
3. Compute tokens: `D = C / (6N)`
4. Set `depth` to match the target parameter count
5. Set `target_flops = C` — training steps are auto-calculated

**Don't expect miracles from more compute:**
Doubling compute gives ~8% improvement in excess loss. The last 10% of quality costs as much as the first 90%.

**Know when to stop:**
The 0.485 BPB floor is real. If you need better than ~0.49 BPB, you need better data or a better architecture — more compute won't get you there.

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/plot_scaling_laws.py` | Plot loss vs compute (power law fit) |
| `scripts/plot_isoflop_curve.py` | Plot isoflop curves and optimal points |
| `scripts/verify_flops_formula.py` | Verify C = 6ND empirically |
| `scripts/verify_scaling_constant.py` | Verify the scaling constant |

---

## Related pages

- [[model]] — depth parameterization implementation
- [[training]] — `target_flops` config and batch scaling
- [[evaluation]] — validation loss computation
