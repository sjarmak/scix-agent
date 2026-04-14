# SciX Experiments

## Purpose

AI/ML experiments on a large scientific literature corpus (NASA ADS metadata). Goal: make scientific knowledge navigable by agents using hybrid retrieval (semantic + structural + symbolic).

## Tech Stack

- **Language**: Python 3
- **Data source**: NASA ADS API (v1)
- **Data format**: JSONL (some gzip/xz compressed)
- **Database**: PostgreSQL 16 + pgvector 0.8.2 (iterative scan, parallel HNSW builds, halfvec)

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
src/scix/                     — Python package (47 modules)
  mcp_server.py               — MCP server (22 tools)
  search.py                   — Hybrid search with RRF fusion
  db.py                       — DB helpers (connection pool, IndexManager, IngestLog)
  ingest.py                   — JSONL→PostgreSQL via COPY
  field_mapping.py            — ADS JSONL→SQL field mapping + transforms
  embed.py                    — INDUS / SPECTER2 embedding pipeline
  graph_metrics.py            — PageRank, HITS, community detection
  extract.py                  — LLM entity extraction
  session.py                  — Agent working set management
  sources/                    — OpenAlex, ar5iv, S2 source modules
  jit/                        — JIT entity resolution (cache, router, NER)
  eval/                       — Retrieval evaluation framework
scripts/                      — CLI tools (78 scripts)
migrations/                   — PostgreSQL schema (44 migrations)
tests/                        — pytest suite (107 test files)
docs/                         — Documentation
  prd/                        — Product requirement docs
  premortem/                  — Premortem risk analyses
  ADR/                        — Architecture decision records
  paper_outline.md            — ADASS 2026 paper outline
  briefing.md                 — Technical briefing document
  figures/                    — Data visualizations
results/                      — Evaluation results (JSON + reports)
ads_metadata_by_year_picard/  — Raw JSONL data files (~100GB, gitignored)
```

## Security

- ADS API key must be in `ADS_API_KEY` env var, never hardcoded
- Add `.env` to `.gitignore`
- Data files (_.jsonl, _.jsonl.gz, \*.jsonl.xz) should be in `.gitignore`

## Testing — Database Safety

**CRITICAL**: The default DSN (`dbname=scix`) points at the production database with 32M papers and 299M citation edges. Integration tests that write/delete data MUST use `SCIX_TEST_DSN`.

```bash
export SCIX_TEST_DSN="dbname=scix_test"
pytest tests/  # integration tests now run against scix_test
```

- `scix_test` database has the full schema (all migrations applied) but no data
- Tests that write to the DB check `is_production_dsn()` and skip if DSN points at production
- **Never run `pytest` against production** without `SCIX_TEST_DSN` set — read-only tests are safe, but integration tests will be skipped

## Agent Autonomy

When running as a Gas City worker (GC_AGENT env var is set):

- Execute autonomously — do NOT ask for confirmation before proceeding
- Do NOT present plans and wait for approval — plan internally then execute
- Close beads (`bd close <id>`) when work is complete
- If blocked, send mail to mayor (`gc mail send mayor "blocked on <reason>"`) and stop

## Conventions

- Use `pytest` for testing
- Use `black` + `ruff` for formatting/linting
- Type annotations on all function signatures
- Immutable data structures preferred (frozen dataclasses, NamedTuples)

## Architecture & Research Context (April 2026)

### Project Identity

- Agent-native navigation layer on full 32.4M-paper ADS corpus, 299M citation edges
- PostgreSQL + pgvector, 22 MCP tools, hybrid search (INDUS + text-embedding-3-large + BM25 via RRF)
- Leiden community detection at multiple resolutions, entity extraction pipeline
- Targeting ADASS 2026 paper
- Embedding pipeline running: multiprocessing GPU pipeline doing binary COPY at 2-5K rec/s
- 32M papers embedded with INDUS

### Dual-Model Embedding Strategy (Decided)

- **INDUS** (nasa-impact/nasa-smd-ibm-st-v2, 768d): domain-specific scientific similarity — trained on 2.66M ADS title-abstract pairs, outperforms larger general models on scientific retrieval
- **text-embedding-3-large** (3072d, flex to 1024d): asymmetric query-document retrieval — general-purpose, strong on diverse queries
- Fused via RRF — this is the decided architecture, not up for debate
- SPECTER2 was the original model; INDUS replaced it as the primary dense signal for the full 32M corpus

### Embedding Landscape (March 2026 MTEB)

| Model                  | nDCG@10 | Dims             | Context | Cost          | License     |
| ---------------------- | ------- | ---------------- | ------- | ------------- | ----------- |
| Gemini Embedding 001   | 67.7    | 3072 (flex->768) | -       | ~$0.15/1M tok | Proprietary |
| Qwen3-Embedding-8B     | ~67     | 7168 (flex)      | 32K     | FREE          | Apache 2.0  |
| Cohere Embed v4        | ~65     | 1536 (flex)      | 128K    | $0.10/1M tok  | Proprietary |
| Voyage 3.5             | ~65     | 2048 (flex)      | 32K     | $0.06/1M tok  | Proprietary |
| text-embedding-3-large | ~63     | 3072 (flex)      | 8K      | $0.13/1M tok  | Proprietary |

- Matryoshka representations: models can be truncated to smaller dims with 98.37% quality at 8.3% size
- For chunk-level full-text retrieval: Voyage 3.5 ($0.06/1M) or self-hosted Qwen3-Embedding-0.6B (Apache 2.0)
- Voyage-context-3 (July 2025): contextualized chunk embeddings, +14.24% over text-embedding-3-large on chunks

### pgvector Path

- pgvector 0.8.2 installed (upgraded from 0.6.0 on 2026-04-06), built from source
- Iterative scan enabled: auto-activates with relaxed_order for filtered queries in vector_search()
- Parallel HNSW builds available (max_parallel_maintenance_workers=7), halfvec (float16), binary quantization
- HNSW sufficient up to 5M vectors (m=16, ef_construction=64, ef_search=100, iterative_scan=relaxed)
- For 30M+ vectors: pgvectorscale StreamingDiskANN — SSD-backed index, 471 QPS at 99% recall on 50M vectors, 28x lower p95 latency than Pinecone, 75% less cost
- halfvec (float16) is safe; binary quantization causes >40% nDCG@10 loss on scientific retrieval — use only as first-pass filter
- Store at 768d in pgvector — fits block-size limits, no TOAST overhead, matches INDUS native output

### Hybrid Search (Implemented)

- Dense (pgvector HNSW) + BM25 (tsvector) fused via RRF — 49-67% error reduction vs dense-only
- RRF constant k=60 is standard
- For true BM25: consider ParadeDB pg_search or Timescale pg_textsearch (March 2026)
- Reranking stage: Cohere Rerank 3.5 or BAAI/bge-reranker-large reduces errors by up to 67% with contextual embeddings

### Tool Count Concern

- 22 MCP tools currently exposed
- A-RAG paper (Feb 2026) shows 3 hierarchical tools (keyword, semantic, chunk read) achieve 94.5% on HotpotQA
- Premortem flagged >15 tools degrades agent tool selection accuracy
- Consider grouping into composite tools or using a meta-tool for discoverability

### Citation Graph Validation

- CG-RAG (SIGIR 2025): dual graph (intra-doc sections + inter-doc citations) enables multi-hop retrieval that neither sparse nor dense retrieval achieves alone — CITE in paper Section 5
- HippoRAG 2 (ICML 2025): Personalized PageRank on knowledge graph +7% over SOTA embeddings, 10-30x cheaper than iterative retrieval
- Our citation_chain, co_citation_analysis, bibliographic_coupling tools already implement CG-RAG pattern
- Leiden with CPM (not modularity) at multiple resolutions (0.001/0.01/0.1) — recommended by CWTS

### Paper Priorities (Section References)

- Section 3.3 edge resolution (17.8% -> 99.6%) should be LEAD result, not buried — "Graph analytics on partial corpora are fundamentally misleading"
- Section 4.4: 50-query eval is core retrieval contribution, add INDUS as third model
- Section 5.4: giant component + community quality metrics need measuring
- Cite: CG-RAG, A-RAG, INDUS (arXiv 2405.10725) in related work

### Full-Text Future Work

- When implementing: Anthropic contextual retrieval (prepend section context to chunks) — 35% reduction in retrieval failure
- Chunk at 512 tokens with section boundaries, never split abstracts
- Parse with GROBID (section structure) + Docling (tables) or Marker (equations)
- RAPTOR-style hierarchical indexing: recursive clustering + summarization, +20% on high-level queries
- Late chunking not recommended for scientific papers (length exceeds sweet spot)

### Key Citations for Paper

- CG-RAG (SIGIR 2025) — validates graph intelligence layer
- A-RAG (arXiv 2602.03442, Feb 2026) — validates agentic retrieval with minimal tools
- INDUS (arXiv 2405.10725) — NASA-specific embedding landscape
- HippoRAG 2 (ICML 2025) — PPR on knowledge graphs
