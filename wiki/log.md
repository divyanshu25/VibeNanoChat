---
title: Wiki log
tags: [log, meta]
---

# Wiki log

Append-only timeline. Newest entries at the **bottom**.

---

## [2026-04-11] wiki | Initial wiki created

Created `wiki/` with `SCHEMA.md`, `index.md`, and seven topic pages: `architecture.md`, `model.md`, `training.md`, `data.md`, `evaluation.md`, `scaling-laws.md`, `chat-ui.md`.

Covers: GPT architecture with nanochat enhancements (RoPE, sliding window attention, value embeddings, per-layer scalars), hybrid MuonAdamW optimizer, FineWeb-Edu Parquet BOS dataloader, CORE/ChatCORE evaluation suite, scaling law methodology (C=6ND, isoflop curves, depth parameterization), and FastAPI chat UI.

## [2026-04-11] wiki | Styling pass — Otter wiki conventions

Rewrote all pages to match Otter wiki style: YAML frontmatter, Obsidian `[[wikilinks]]`, `> [!ABSTRACT]` / `> [!NOTE]` / `> [!TIP]` / `> [!WARNING]` / `> [!IMPORTANT]` callout blocks, `## Contents` with `[[#section]]` anchor links, and consistent "Related pages" footers. Updated `index.md` as a proper navigation homepage with grouped sections.
