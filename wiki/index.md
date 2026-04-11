---
title: VibeNanoChat Wiki
tags: [index, navigation]
---

# VibeNanoChat Wiki

> [!ABSTRACT] What is this?
> The authoritative technical reference for the VibeNanoChat codebase. Every page is written and maintained by an LLM agent. You read; the agent writes. See [[SCHEMA]] for the workflow.

---

## Start here

| | Page | What you'll find |
|--|------|-----------------|
| 🗺 | [[architecture]] | System map: modules, entry points, data flows, tech stack |
| 🧠 | [[model]] | GPT architecture: transformer blocks, RoPE, sliding window, value embeddings |

---

## Deep-dives

| | Page | What you'll find |
|--|------|-----------------|
| 🏋 | [[training]] | Training pipeline: DDP, MuonAdamW optimizer, LR schedule, checkpointing |
| 🗄 | [[data]] | Data pipeline: FineWeb-Edu, Parquet format, BOS-aligned packing |
| 📊 | [[evaluation]] | Benchmarks: CORE, ChatCORE, scoring, eval intervals |
| 📈 | [[scaling-laws]] | Scaling laws: C=6ND, isoflop curves, depth parameterization |
| 💬 | [[chat-ui]] | Web chat UI: FastAPI server, API endpoints, deployment |

---

## Meta

| | Page | What you'll find |
|--|------|-----------------|
| ⚙️ | [[SCHEMA]] | How agents maintain this wiki (conventions, update workflow) |
| 📋 | [[log]] | Append-only timeline of all wiki activity |

---

## External docs (repo root)

| Doc | Summary |
|-----|---------|
| [README.md](../README.md) | Features, quick start, dataset, Makefile reference |
| [llm_wiki.md](../llm_wiki.md) | The LLM Wiki pattern that inspired this setup |
