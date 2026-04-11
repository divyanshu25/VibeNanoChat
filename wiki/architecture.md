---
title: Architecture
tags: [architecture, systems, overview]
---

# Architecture

> [!ABSTRACT]
> High-level map of the VibeNanoChat system: what each module does and how the pieces connect. For the *why* behind design choices, see [[architecture#Key design decisions]].

---

## Contents

- [[#What the system is]]
- [[#Module map]]
- [[#Entry points]]
- [[#Data flows]]
- [[#Key design decisions]]
- [[#Tech stack]]

---

## What the system is

VibeNanoChat is a GPT-2 scale language model training framework built on nanoGPT's clean implementation and nanochat's depth parameterization. The goal: train, evaluate, and experiment with transformer language models from scratch — minimal abstraction, maximum hackability.

| Mode | Description |
|------|-------------|
| **Pretraining** | Next-token prediction on FineWeb-Edu (or OpenWebText) |
| **SFT** | Supervised fine-tuning via multiplex dataloader mixing |

---

## Module map

```
VibeNanoChat/
│
├── src/
│   ├── gpt_2/                         Core model and training
│   │   ├── gpt2_model.py              GPT nn.Module — full architecture
│   │   ├── trainer.py                 Trainer class — training loop
│   │   ├── ddp.py                     torchrun entry point
│   │   ├── config.py                  GPTConfig dataclass — all hyperparameters
│   │   ├── attention.py               Multi-head attention + RoPE application
│   │   ├── block.py                   Transformer block (attn + MLP)
│   │   ├── mlp.py                     Feed-forward network
│   │   ├── muon.py                    Muon / DistMuon / MuonAdamW optimizer
│   │   ├── rope.py                    RoPE precomputation
│   │   ├── kv_cache.py                KV cache for generation
│   │   ├── utils.py                   Shared helpers
│   │   └── training_utilities/        Setup helpers called by ddp.py
│   │       ├── batch_scaling.py       Auto-scale batch size, LR, WD from depth
│   │       ├── hyperparameter_setup.py
│   │       ├── model_setup.py
│   │       ├── dataloader_setup.py
│   │       ├── logging_setup.py
│   │       └── wandb_setup.py
│   │
│   ├── dataloaders/
│   │   ├── fineweb_edu_parquet_bos_dataloader.py   Pretraining dataloader
│   │   └── multiplex_dataloader.py                 SFT mixing dataloader
│   │
│   └── eval_tasks/
│       ├── core/                      Base model evals (multiple-choice LM)
│       ├── chat_core/                 Chat model evals (generative: GSM8K, HumanEval)
│       └── training/                  Eval wrappers called during training
│
├── data/
│   ├── fineweb_edu/                   FineWeb-Edu download + Parquet prep
│   ├── openwebtext/                   OpenWebText prep (alternative dataset)
│   └── task_mixture/                  SFT task mixture datasets
│
├── scripts/
│   ├── chat.py                        CLI chat interface
│   ├── eval_checkpoints.py            Offline eval of saved checkpoints
│   ├── plot_scaling_laws.py           Scaling law visualization
│   └── plot_isoflop_curve.py          Isoflop curve plotting
│
├── debug_tools/
│   ├── interactive_generate.py        Simple interactive text generation
│   └── demo_kvcache_speedup.py        KV cache benchmark
│
├── chat_ui/                           FastAPI web chat interface
│   ├── server.py                      FastAPI app + model/session management
│   ├── asgi.py                        ASGI application wrapper
│   ├── config.py                      Server config (port, checkpoint dir)
│   └── gunicorn_config.py             Production server settings
│
├── tests/
│   ├── unit/                          Unit tests: models, dataloaders, utilities
│   └── integration/                   Integration tests
│
├── docs/                              Deep-dive concept guides (developer-authored)
├── wiki/                              This wiki (LLM-maintained)
└── Makefile                           All training and utility commands
```

---

## Entry points

| Entry point | Command | Purpose |
|-------------|---------|---------|
| `src/gpt_2/ddp.py` | `make ddp-train` | Distributed pretraining or SFT |
| `chat_ui/server.py` | `make chat-server` | Web chat UI |
| `scripts/chat.py` | `uv run python scripts/chat.py` | CLI chat with a checkpoint |
| `debug_tools/interactive_generate.py` | `make interactive-gen` | Simple text generation |
| `scripts/eval_checkpoints.py` | direct invocation | Offline benchmark evaluation |

---

## Data flows

### Training

```
Parquet files on disk  (memory-mapped via mmap)
        │
        ▼
FineWebEduParquetBOSDataloader
  — BOS-aligned document packing
  — Strided reads per GPU rank
        │
        ▼
Batches of (tokens, targets) → GPU
        │
        ▼
GPT.forward() → logits, cross-entropy loss
        │
        ▼
loss.backward()
        │
        ▼
DistMuonAdamW.step()
  — all_reduce gradients internally (no DDP wrapper)
  — Muon for matrices · AdamW for embeddings/head
        │
        ▼
Checkpoint saved every N steps → logs/step_XXXXX.pt
        │
        ▼
CORE / ChatCORE evals on GPU 0 every M steps → W&B + logs
```

### Inference (web UI)

```
HTTP POST /api/chat  (user message)
        │
        ▼
SessionManager — in-memory conversation history
        │
        ▼
ModelManager.generate()
  — autoregressive loop with KV cache
  — GPT.forward() called token-by-token
        │
        ▼
Decoded token string → HTTP response
```

---

## Key design decisions

**No DDP wrapper on the model.**
`DistMuonAdamW` handles gradient synchronization internally via `all_reduce` inside `optimizer.step()`. This avoids `DistributedDataParallel` overhead and enables better overlap of communication with computation.

**Single `depth` knob.**
Model width (`n_embed`) and depth (`n_layer`) are both derived from one integer: `n_embed = depth × aspect_ratio`. Scaling law sweeps change one number; weight decay, LR, and batch size auto-adjust.

**Parquet + mmap for data.**
Training data is stored as Parquet and memory-mapped. The OS loads pages on demand — zero-copy I/O, no upfront deserialization. A 100 GB dataset doesn't consume 100 GB RAM.

**Hybrid optimizer.**
Transformer weight matrices use Muon (Newton-Schulz orthogonalization). Embeddings and the LM head use AdamW. Both live in a single combined optimizer with per-group `kind` flags.

> [!NOTE] Why no DDP?
> Standard DDP wraps the model and all-reduces gradients after `loss.backward()`. DistMuon instead all-reduces inside `step()`, which means the communication can overlap with the Newton-Schulz orthogonalization computation — effectively hiding the gradient sync latency.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python ≥ 3.10 |
| Package manager | `uv` + `pyproject.toml` |
| ML framework | PyTorch (CUDA 12.8) |
| Attention kernel | Flash Attention 3 via `kernels` (H100-optimized) |
| Data format | Apache Parquet (PyArrow) |
| Tokenizer | `tiktoken` — GPT-2 vocab (50,257 + 9 special = 50,266) |
| Experiment tracking | Weights & Biases (`wandb`) |
| Web server | FastAPI + Uvicorn + Gunicorn |
| Testing | pytest + pytest-cov |
| Linting / formatting | Ruff, Black, isort |

---

## Related pages

- [[model]] — detailed model architecture
- [[training]] — training loop and configuration
- [[data]] — data pipeline details
- [[scaling-laws]] — depth parameterization and scaling experiments
