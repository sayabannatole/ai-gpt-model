# ai-gpt-model

A tiny, local-first **context search model** for small websites.

This project gives you a mini retrieval system (TF-IDF based) that:

1. Crawls a small website.
2. Builds an index of page content.
3. Lets you run context-based search queries against that index.

## Why this is useful

For small internal/docs sites, you often do not need a full vector database or hosted LLM stack. This repo provides a lightweight baseline you can run locally and extend later.

## Quickstart

### 1) Build an index

```bash
python -m src.context_search build https://example.com --max-pages 15 --out context_index.json
```

### 2) Search the index

```bash
python -m src.context_search search "getting started api" --index context_index.json --top-k 5
```

## What is implemented

- A mini retriever (`MiniContextModel`) with:
  - Tokenization
  - TF-IDF vector creation
  - Cosine similarity ranking
- A crawler (`crawl_site`) that:
  - Traverses same-domain links
  - Extracts visible HTML text
  - Builds `PageDocument` objects
- Save/load index as JSON for local testing.

## Run tests

```bash
python -m unittest discover -s tests
```

## Next upgrades

- Add chunking per page section instead of whole page text.
- Add metadata filters (docs/blog/changelog tags).
- Replace TF-IDF with sentence embeddings when you need semantic search.
