---
title: Evaluation
tags: [evaluation, benchmarks, core, chatcore, mmlu]
---

# Evaluation

> [!ABSTRACT]
> Benchmark strategy for VibeNanoChat: CORE multiple-choice evals for base models, ChatCORE generative evals for chat-tuned models, and how both integrate into the training loop.

---

## Contents

- [[#Overview]]
- [[#CORE benchmarks]]
- [[#ChatCORE benchmarks]]
- [[#Eval schedule]]
- [[#Offline evaluation]]
- [[#Validation loss]]
- [[#Eval architecture]]

---

## Overview

Two evaluation suites run during (and after) training:

| Suite | Type | When | Source |
|-------|------|------|--------|
| **CORE** | Multiple-choice LM scoring | During pretraining | `src/eval_tasks/core/` |
| **ChatCORE** | Generative (free-form output) | During SFT | `src/eval_tasks/chat_core/` |

Both run on **GPU 0 only** while other GPUs continue training — no compute is wasted.

---

## CORE benchmarks

CORE evals score a base model on multiple-choice tasks using **log-likelihood scoring**: the model scores each answer choice by its log-probability; the highest-scoring choice is selected.

### Task list

| Benchmark | What it tests | Random baseline |
|-----------|--------------|----------------|
| MMLU | College-level knowledge across 57 subjects | 25% |
| HellaSwag | Commonsense reasoning — completing everyday scenarios | 25% |
| PIQA | Physical reasoning — how objects interact | 50% |
| WinoGrande | Pronoun resolution requiring world knowledge | 50% |
| ARC-Easy | Grade-school science (easy set) | 25% |
| ARC-Challenge | Grade-school science (hard set) | 25% |
| OpenBookQA | Science questions needing common sense + facts | 25% |
| TriviaQA | Factual recall from reading passages | — |
| *+ 25 more* | See `resources/eval_bundle/EVAL_GAUNTLET.md` | — |

### Scoring

Scores are **normalized**: `0% = random guessing`, `100% = perfect`. This makes scores comparable across tasks with different numbers of answer choices.

### Expected performance (depth=12, 10B tokens)

| Benchmark | Expected range |
|-----------|---------------|
| MMLU | 30–40% |
| HellaSwag | 50–60% |
| PIQA | 65–75% |
| ARC-Challenge | 30–40% |

> [!NOTE]
> For reference: GPT-2 (124M) scored 33.4% on MMLU. These numbers are a sanity check, not a ceiling.

---

## ChatCORE benchmarks

ChatCORE evals score a chat-tuned model on **generative tasks** — the model produces free-form output that is parsed and checked for correctness.

| Benchmark | What it tests | Metric |
|-----------|--------------|--------|
| GSM8K | Grade-school math word problems | Exact match on final answer |
| HumanEval | Python code generation | `pass@k` (functional correctness) |
| MMLU-chat | MMLU in chat format | Accuracy |
| ARC-chat | ARC in chat format | Accuracy |

### ChatCORE config defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `chat_core_num_samples` | 1 | Samples per problem (for `pass@k`) |
| `chat_core_max_examples` | 500 | Max examples per task (for speed) |
| `chat_core_max_tokens` | 512 | Max tokens per generation |
| `chat_core_temperature` | 0.0 | Greedy decoding by default |
| `chat_core_top_k` | 50 | Top-k filtering |

---

## Eval schedule

| Parameter | Default | Notes |
|-----------|---------|-------|
| `eval_interval` | 2000 steps | Validation loss evaluation frequency |
| `core_eval_interval` | 2000 steps | CORE benchmark evaluation frequency |
| `core_eval_max_examples` | 500 | Max examples per CORE task |

Both intervals default to **adaptive** values based on `total_steps` if not overridden — they scale so you always get a reasonable number of eval checkpoints regardless of training length.

### Enabling evals

```bash
# Enable CORE evals during pretraining
make ddp-train NGPUS=8 CORE_EVALS=true

# Enable ChatCORE evals during SFT
make ddp-train NGPUS=8 --mode sft CHAT_CORE_EVALS=true
```

---

## Offline evaluation

To evaluate a saved checkpoint after training:

```bash
uv run python scripts/eval_checkpoints.py --checkpoint logs/step_19531.pt
```

This runs the full eval suite on the checkpoint and prints results.

---

## Validation loss

In addition to benchmarks, **validation loss** (bits per byte, BPB) is computed periodically:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `val_loss_eval_tokens` | 10,485,760 | ~10M tokens for validation loss estimation |

BPB is the primary metric for scaling law experiments. See [[scaling-laws]].

---

## Eval architecture

- Evals run on **GPU 0 only** — other GPUs continue training
- Each task is a separate Python module in `src/eval_tasks/`
- Tasks implement a common interface: `run_eval(model, config) → score`
- Results are logged to W&B and printed to stdout

> [!TIP]
> Use `core_eval_max_examples = 500` (the default) during training for fast evals. For final reporting, set it to `None` to run the full task.

---

## Related pages

- [[training]] — eval intervals and training loop integration
- [[scaling-laws]] — how validation BPB feeds into scaling experiments
- [[chat-ui]] — interactive evaluation via the web UI
