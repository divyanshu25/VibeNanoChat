---
title: Training
tags: [training, ddp, optimizer, muon, checkpointing]
---

# Training

> [!ABSTRACT]
> How training works end-to-end: entry point, Trainer class, optimizer, LR schedule, and checkpointing. For data preparation, see [[data]].

---

## Contents

- [[#Quick start]]
- [[#Entry point]]
- [[#Training modes]]
- [[#Training horizon]]
- [[#Batch scaling]]
- [[#Optimizer]]
- [[#LR schedule]]
- [[#Gradient clipping]]
- [[#Checkpointing]]
- [[#Distributed training]]
- [[#W&B logging]]
- [[#Makefile reference]]
- [[#Troubleshooting]]

---

## Quick start

```bash
# 1. Install dependencies
make environment

# 2. Prepare data (~10B tokens)
cd data/fineweb_edu && uv run python prepare_parquet.py --config sample-10BT && cd ../..

# 3. Train (8 GPUs, ~6 hours for depth=12 / 124M params)
make ddp-train NGPUS=8

# Common overrides
make ddp-train NGPUS=4 DEPTH=8                      # Small model, 4 GPUs
make ddp-train NGPUS=8 DEPTH=16 BATCH_SIZE=128      # Larger model
make ddp-train NGPUS=8 TARGET_FLOPS=2e18            # Fixed compute budget
make ddp-train NGPUS=8 CORE_EVALS=true              # Enable benchmark evals
```

---

## Entry point

`src/gpt_2/ddp.py` is the `torchrun` entry point. It:

1. Parses CLI args and builds a `GPTConfig`
2. Calls `training_utilities/` setup helpers (model, dataloader, optimizer, logging, W&B)
3. Instantiates `Trainer` and calls `trainer.train()`

```bash
torchrun --nproc_per_node=$NGPUS src/gpt_2/ddp.py \
    --depth $DEPTH \
    --target_flops $TARGET_FLOPS \
    ...
```

The Makefile wraps this with sensible defaults.

---

## Training modes

| Mode | Flag | Dataloader | Use case |
|------|------|-----------|---------|
| Pretraining | `--mode pretraining` | FineWeb-Edu Parquet BOS | Train from scratch |
| SFT | `--mode sft` | Multiplex (task mixture) | Fine-tune on instructions |

---

## Training horizon

How many steps to train is determined in priority order:

1. **`target_flops`** (if > 0) — compute `num_iterations` to reach the target FLOP budget
2. **`target_param_data_ratio`** (if > 0) — compute `num_iterations` from the data:param ratio
3. **`num_epochs`** — fall back to epoch-based training

> [!NOTE] Default
> `target_param_data_ratio = 10` (10 tokens per parameter). Chinchilla recommends 20; we use 10 based on empirical results with DistMuon and FineWeb-Edu. See [[scaling-laws]].

---

## Batch scaling

When `depth` is set, batch size, LR, and weight decay auto-scale via `src/gpt_2/training_utilities/batch_scaling.py`.

### Weight decay

```
WD = base_WD × (12 / depth)²
```

Deeper models need less regularization. At `depth=12`, WD = `base_WD`. At `depth=6`, WD is 4× higher.

### Effective batch size

Default `total_batch_size = 2^19 ≈ 524,288 tokens` per gradient update, achieved via gradient accumulation across GPUs and micro-batches.

### Learning rate

LRs are **fixed per parameter group** (tuned at `depth=12`) and do not scale with depth. They scale with `sqrt(batch_size)` if batch size is changed from the default.

---

## Optimizer

Source: `src/gpt_2/muon.py`

A single combined optimizer (`MuonAdamW` / `DistMuonAdamW`) with two strategies:

### Muon — for transformer matrices

- Applies **Newton-Schulz orthogonalization** to the gradient before the momentum update
- Produces near-orthogonal updates — prevents weight matrices from becoming rank-deficient
- Used for all 2D weight matrices in transformer blocks
- Default LR: `matrix_lr = 0.02`

### AdamW — for everything else

- Standard adaptive optimizer for embeddings, LM head, biases, scalars
- Sparse updates (only active tokens get gradients) make per-parameter adaptive LR essential here

### Distributed: DistMuonAdamW

In multi-GPU training, `DistMuonAdamW` handles gradient synchronization **inside `optimizer.step()`** — no `DistributedDataParallel` wrapper on the model. Benefits:
- Communication overlaps with Newton-Schulz computation
- ZeRO-2 style optimizer state sharding (each GPU stores 1/N of optimizer state)

> [!IMPORTANT]
> Do not wrap the model in `DistributedDataParallel` when using `DistMuonAdamW`. The optimizer handles all gradient synchronization internally.

---

## LR schedule

Nanochat-style warmup/warmdown (trapezoidal):

```
Step 0 → warmup_end:           LR ramps 0 → peak_lr
warmup_end → warmdown_start:   LR stays at peak_lr
warmdown_start → end:          LR decays to final_lr_frac × peak_lr
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `warmup_ratio` | 0.0 | Fraction of total steps for warmup |
| `warmdown_ratio` | 0.4 | Fraction of total steps for warmdown |
| `final_lr_frac` | 0.0 | Final LR as fraction of peak (0 = full decay) |

---

## Gradient clipping

```python
gradient_clip_norm = 1.0  # Max gradient norm
```

Applied before every optimizer step. Prevents gradient explosions, especially early in training.

---

## Checkpointing

| Parameter | Default | Notes |
|-----------|---------|-------|
| `checkpoint_interval_pretrain` | 5000 steps | Save frequency during pretraining |
| `checkpoint_interval_sft` | 100 steps | Save frequency during SFT |

Checkpoints are saved as `.pt` files in `logs/`:

```
logs/
├── step_05000.pt
├── step_10000.pt
└── ...
```

Each checkpoint contains model weights, optimizer state, and training metadata (step, config).

---

## Distributed training

Uses `torchrun` without wrapping the model in `DistributedDataParallel`:

- Each GPU runs an identical copy of the model
- Gradients are all-reduced inside `DistMuonAdamW.step()`
- Dataloader uses strided reads — each GPU rank reads a different slice of the data

**Scaling:** Works with 1–8 GPUs without code changes. Gradient accumulation auto-adjusts to maintain effective batch size. ~95% scaling efficiency up to 8 GPUs on a single node (NVLink).

---

## W&B logging

When enabled, logged per step:

- Training loss
- Validation loss (every `eval_interval` steps)
- CORE benchmark scores (every `core_eval_interval` steps)
- Learning rate per parameter group
- Gradient norm
- Tokens per second (throughput)

---

## Makefile reference

| Command | Description |
|---------|-------------|
| `make ddp-train` | Start distributed training with defaults |
| `make run-scaling-law` | Sweep depths × FLOP budgets (6 × 4 = 24 runs) |
| `make run-depth-sweep` | Sweep depths at fixed data:param ratio |
| `make gpu-status` | Check GPU utilization, memory, temperature |
| `make kill-gpu` | Kill zombie GPU processes |
| `make format` | Format code with Black and isort |

---

## Troubleshooting

**OOM (Out of Memory)**
Reduce `BATCH_SIZE`. Default is 64; try 32 or 16. Main consumers: activations (~2–4 GB), model params (~500 MB in bfloat16), optimizer state (~500 MB, sharded).

**NaN loss**
Usually LR too high or corrupted data. Try reducing LR or enabling QK-LayerNorm in config.

**Slow data loading**
GPU utilization should be ~95%. If lower: check disk speed, avoid NFS/EFS, ensure data is on local SSD.

**NCCL errors**
All GPUs must be on the same node with NVLink. Check: `nvidia-smi topo -m`. Verify NCCL: `python -c "import torch; print(torch.cuda.nccl.version())"`.

> [!WARNING]
> If training crashes mid-run, zombie CUDA processes may hold GPU memory. Run `make kill-gpu` before restarting.

---

## Related pages

- [[model]] — model architecture and parameter groups
- [[data]] — data preparation and dataloader details
- [[scaling-laws]] — depth parameterization and compute budgets
- [[evaluation]] — eval benchmarks run during training
