---
title: Data
tags: [data, parquet, fineweb, dataloader, sft]
---

# Data

> [!ABSTRACT]
> Data pipeline for VibeNanoChat: datasets, Parquet format, BOS-aligned packing, memory-mapped I/O, and SFT mixing. For how the dataloader is wired into training, see [[training]].

---

## Contents

- [[#Datasets]]
- [[#Parquet schema]]
- [[#BOS-aligned packing]]
- [[#Memory-mapped I/O]]
- [[#Multi-GPU distribution]]
- [[#SFT: multiplex dataloader]]
- [[#Using your own data]]

---

## Datasets

### FineWeb-Edu (primary)

| Property | Value |
|----------|-------|
| Source | Common Crawl, filtered for educational content |
| Sample size | 10B tokens (expandable to 1.3T) |
| Quality | High — Wikipedia, academic blogs, tutorials, textbooks |
| Format | Tokenized Parquet shards |
| Prep script | `data/fineweb_edu/prepare_parquet.py` |

```bash
cd data/fineweb_edu
uv run python prepare_parquet.py --config sample-10BT
cd ../..
```

Available configs: `sample-10BT`, `sample-100BT`, `full` (1.3T tokens).

> [!TIP] Why FineWeb-Edu over raw Common Crawl?
> Educational filtering removes low-quality pages. High-quality tokens are worth more — you need fewer tokens to reach the same benchmark performance, and the filtering reduces noise that can cause training instability.

### OpenWebText (alternative)

Recreation of GPT-2's original training set. Use this to replicate original GPT-2 results. Prep scripts in `data/openwebtext/`.

### SFT task mixture

For supervised fine-tuning, task datasets live in `data/task_mixture/`. The multiplex dataloader mixes them at configurable ratios.

---

## Parquet schema

Training data is stored as tokenized Parquet files. Each file contains rows of packed token sequences.

| Column | Type | Description |
|--------|------|-------------|
| `tokens` | `list[int32]` | Packed token IDs, length = `block_size` |

Documents are tokenized with the **GPT-2 tokenizer** (`tiktoken`), then packed into fixed-length sequences using BOS-aligned best-fit packing.

File layout:

```
fineweb_edu_parquet/
├── train/
│   ├── shard_0000.parquet
│   ├── shard_0001.parquet
│   └── ...
└── val/
    └── shard_0000.parquet
```

---

## BOS-aligned packing

Source: `src/dataloaders/fineweb_edu_parquet_bos_dataloader.py`

Standard packing concatenates documents end-to-end, causing the model to see cross-document token pairs at boundaries. BOS-aligned packing preserves document boundaries:

1. A buffer of documents is loaded (`bos_dataloader_buffer_size = 4096` docs)
2. Documents are packed using **best-fit** into sequences of exactly `block_size` tokens
3. Each document starts with a BOS token; sequences are padded or split at boundaries
4. The model never attends across document boundaries within a sequence

> [!NOTE] Why this matters
> The model learns that BOS marks the start of a new document. This improves generation quality and prevents context from "bleeding" across documents — a subtle but meaningful quality improvement.

### Dataloader config defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `bos_dataloader_buffer_size` | 4096 | Documents buffered for packing |
| `dataloader_persistent_workers` | `True` | Keep workers alive between epochs |
| `num_workers` | Auto | Determined by model depth |
| `prefetch_factor` | Auto | Determined by model depth |

Worker count and prefetch factor are auto-tuned based on `depth` to prevent data loading from becoming a bottleneck.

---

## Memory-mapped I/O

Parquet files are memory-mapped by the OS. The dataloader reads data via page faults — the OS loads only the pages currently needed.

- No upfront deserialization — the full dataset is never loaded into RAM
- Zero-copy reads — data goes directly from disk to GPU memory
- Works at any scale — a 1 TB dataset uses the same RAM as a 1 GB dataset

> [!WARNING]
> Data must be on a **local SSD**. Memory mapping over NFS or EFS is unreliable and slow. Verify disk speed: `dd if=/path/to/data of=/dev/null bs=1M count=1000` (need ≥ 500 MB/s).

---

## Multi-GPU distribution

Each GPU rank reads a **strided slice** of the data:

```
GPU 0: rows 0, N, 2N, 3N, ...
GPU 1: rows 1, N+1, 2N+1, ...
GPU 2: rows 2, N+2, 2N+2, ...
```

where `N = world_size`. No two GPUs see the same data, with no communication required for data distribution.

---

## SFT: multiplex dataloader

Source: `src/dataloaders/multiplex_dataloader.py`

For supervised fine-tuning, multiple task datasets are mixed at runtime:

- Loads multiple datasets simultaneously
- Mixes them at configurable per-dataset ratios
- Handles different sequence formats (instruction-following, chat, etc.)
- Caches tokenized datasets to `sft_cache_dir`

Config:
```python
sft_cache_dir = "<YOURPATH>/sft_cache"
```

---

## Using your own data

1. **Tokenize** your text using the GPT-2 tokenizer (`tiktoken`)
2. **Pack** into sequences of length `block_size` (2048 by default)
3. **Save** as Parquet files with a `tokens` column (`list[int32]`)
4. **Update** `data_dir_pretrain_parquet` in `GPTConfig` to point to your directory

The trainer auto-discovers all `.parquet` files in the configured directory. Reference `data/fineweb_edu/prepare_parquet.py` for the exact tokenization and packing logic.

---

## Related pages

- [[training]] — how the dataloader is wired into the training loop
- [[model]] — tokenizer and vocab size details
- [[architecture]] — system overview
