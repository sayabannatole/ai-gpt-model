# ai-gpt-model

A tiny, local-first **context search model** for small websites.

This project gives you a mini retrieval system (sentence-embedding based) that:

1. Crawls a small website.
2. Builds an index of section-level content chunks.
3. Lets you run semantic context search queries against that index.
4. Supports metadata filters for `docs`, `blog`, and `changelog` pages.

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

### 3) Filter by metadata tags

```bash
python -m src.context_search search "release notes" --index context_index.json --tags changelog
```

## What is implemented

- A mini semantic retriever (`MiniContextModel`) with:
  - Tokenization
  - Local sentence embeddings
  - Cosine similarity ranking
- A crawler (`crawl_site`) that:
  - Traverses same-domain links
  - Extracts section-level chunks from heading/content boundaries
  - Adds metadata tags (`docs`, `blog`, `changelog`) inferred from URL paths
  - Builds `PageDocument` chunk objects
- Save/load index as JSON for local testing.

## Run tests

```bash
python -m unittest discover -s tests
```
