---
title: Wiki schema
tags: [meta, schema, agent]
---

# Wiki schema

> [!ABSTRACT]
> This file tells agents how to maintain `wiki/`. Read it at the start of every wiki-related session. The wiki is the authoritative technical reference for VibeNanoChat — `README.md` covers setup and features only.

---

## Layout

| Path | Role |
|------|------|
| `wiki/SCHEMA.md` | This file — conventions and workflows |
| `wiki/index.md` | Catalog and homepage (update on every ingest or new page) |
| `wiki/log.md` | Append-only timeline of ingests, queries, lint passes |
| `wiki/*.md` | Topic pages — cross-link liberally |

---

## Conventions

- **Filenames:** `kebab-case.md` for new pages.
- **Links:** Use Obsidian wikilinks `[[page-name]]` for internal wiki links. Use relative markdown links `[text](../path)` for links outside `wiki/`.
- **Frontmatter:** Add YAML frontmatter (`title`, `tags`) to every page for Obsidian graph view labeling.
- **Callouts:** Use Obsidian callout syntax for key insights, warnings, and tips: `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`, `> [!IMPORTANT]`, `> [!ABSTRACT]`.
- **Contents:** Every page longer than ~3 sections should have a `## Contents` block with `[[#section]]` anchor links.
- **Truth:** For code behavior, prefer citing actual file paths over guessing. When this wiki contradicts the code, trust the code.
- **No duplication:** Don't restate what's in `README.md`. The wiki owns all technical depth; `README.md` owns setup and feature descriptions.

---

## What belongs here vs. `docs/`

| `wiki/` | `docs/` |
|---------|---------|
| What the system does today | How/why something works from first principles |
| Current config defaults | Deep derivations and math |
| Module responsibilities | Concept explorations and ideas |
| API contracts and schemas | Historical context and references |

---

## Workflows

### Ingest

User points to a source (repo file, doc, or external material). Agent:
1. Reads the source
2. Adds or updates wiki pages
3. Updates `wiki/index.md`
4. Appends an entry to `wiki/log.md`

A single source may touch multiple pages. That's expected.

### Query

User asks a question. Agent:
1. Reads `index.md` to find relevant pages
2. Reads those pages
3. Answers with citations (page + section)
4. If the answer is valuable enough, files it as a new wiki page

### Lint

Periodically, or when asked: check for stale links, orphan pages, claims that may contradict the code, missing cross-references, important concepts that lack their own page.

---

## Log entry format

Prefix each entry for easy grepping:

```markdown
## [YYYY-MM-DD] ingest | Short title
## [YYYY-MM-DD] query | Short topic
## [YYYY-MM-DD] lint | Notes
## [YYYY-MM-DD] wiki | Notes
```
