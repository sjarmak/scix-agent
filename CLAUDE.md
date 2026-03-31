# SciX Experiments

## Purpose

AI/ML experiments on a large scientific literature corpus (NASA ADS metadata). Goal: make scientific knowledge navigable by agents using hybrid retrieval (semantic + structural + symbolic).

## Tech Stack

- **Language**: Python 3
- **Data source**: NASA ADS API (v1)
- **Data format**: JSONL (some gzip/xz compressed)
- **Database**: PostgreSQL 16 + pgvector 0.6.0

## Data

~100GB of ADS metadata across 6 years in `ads_metadata_by_year_picard/`:

| Year | Records | Format    |
| ---- | ------- | --------- |
| 2021 | ~1.1M   | .jsonl.xz |
| 2022 | ~1.1M   | .jsonl    |
| 2023 | ~1.2M   | .jsonl    |
| 2024 | ~1.2M   | .jsonl.gz |
| 2025 | ~430K   | .jsonl    |
| 2026 | 21      | .jsonl.gz |

Each record contains: bibcode, title, abstract, authors, affiliations, keywords, citations, references, DOIs, arxiv_class, doctype, and more (~40 fields).

## Project Structure

```
src/scix/                     — Python package
  field_mapping.py            — JSONL→SQL field mapping + transform_record()
  db.py                       — DB helpers (connection, IndexManager, IngestLog)
  ingest.py                   — Ingestion pipeline (JSONL→PostgreSQL via COPY)
scripts/
  setup_db.sh                 — Idempotent database creation + schema application
  ingest.py                   — CLI for ingestion pipeline
migrations/
  001_initial_schema.sql      — papers, citation_edges, paper_embeddings, extractions
  002_ingest_log.sql          — Ingestion progress tracking
tests/                        — pytest suite (53 tests)
ads_metadata_year_*.py        — ADS API harvest scripts (by year range)
ads_metadata_by_year_picard/  — Raw JSONL data files (~100GB)
```

## Security

- ADS API key must be in `ADS_API_KEY` env var, never hardcoded
- Add `.env` to `.gitignore`
- Data files (_.jsonl, _.jsonl.gz, \*.jsonl.xz) should be in `.gitignore`

## Conventions

- Use `pytest` for testing
- Use `black` + `ruff` for formatting/linting
- Type annotations on all function signatures
- Immutable data structures preferred (frozen dataclasses, NamedTuples)
