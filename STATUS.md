# Project Status — Updated 2026-04-03 18:15 EDT

## Running Pipelines

### SPECTER2 Embedding (started ~6:05 PM EDT Apr 3)

- **Script**: `.venv/bin/python3 scripts/embed_from_file.py --input /tmp/to_embed.tsv`
- **Log**: `embed_run_full.log`
- **Rate**: ~508 rec/s (GPU-bound, 100% RTX 5090)
- **Total**: 26,960,172 papers
- **ETA**: ~9 AM EDT Apr 4
- **Architecture**: multiprocessing — main process (file read + tokenize + GPU), writer process (binary COPY)

**CRITICAL post-completion steps:**

```bash
# 1. Verify
tail -5 embed_run_full.log
psql -d scix -c "SELECT count(*) FROM paper_embeddings WHERE model_name = 'specter2';"
# Expected: ~32M (5.4M existing + 27M new)

# 2. Re-enable WAL (table is UNLOGGED for bulk speed)
psql -d scix -c "ALTER TABLE paper_embeddings SET LOGGED;"

# 3. Rebuild HNSW index (dropped for bulk speed)
psql -d scix -c "
CREATE INDEX CONCURRENTLY idx_embed_hnsw_specter2 ON paper_embeddings
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'specter2';
"

# 4. Cleanup
psql -d scix -c "DROP TABLE IF EXISTS _to_embed;"
rm /tmp/to_embed.tsv
```

## Database State

| Setting            | Value                               | Note                            |
| ------------------ | ----------------------------------- | ------------------------------- |
| `shared_buffers`   | 16GB                                | Changed from 128MB this session |
| `paper_embeddings` | **UNLOGGED**                        | Must SET LOGGED after embedding |
| HNSW indexes       | **DROPPED**                         | Must rebuild after embedding    |
| `_to_embed` table  | Exists (27M rows, UNLOGGED)         | Drop after embedding            |
| `paper_metrics`    | 32.4M rows, PageRank+HITS populated | Communities are NULL            |

## Corpus

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| Papers              | 32,390,237 (years 1800-2026) |
| Citation edges      | 299,253,213 (99.6% resolved) |
| SPECTER2 embeddings | 5.4M done + 27M in progress  |
| OpenAI embeddings   | 0                            |
| Extractions         | 0                            |

## What's Done

- Full corpus ingestion (32.4M papers from 227 JSONL files)
- PageRank + HITS on full 32M-node citation graph (stored in paper_metrics)
- Knowledge enrichment code: 22 MCP tools, entity extraction pipeline, dual-model search, session state
- 64 tests passing
- Code review: 5 HIGH + 4 MEDIUM issues fixed
- Paper outline: `docs/paper_outline.md` (ADASS 2026)
- PRDs: `docs/prd/prd_knowledge_enrichment.md` (premortem-annotated)
- PostgreSQL tuned (shared_buffers 16GB, work_mem 256MB, maintenance_work_mem 2GB)
- Embedding pipeline optimized from 32→508 rec/s
- Repo: https://github.com/sjarmak/scix-agent

## What's Next (priority order)

### Immediate (after embedding completes)

1. Run post-completion steps above (LOGGED, HNSW rebuild, cleanup)
2. Leiden communities on giant component (previous attempt OOM'd at 43GB — see below)
3. Taxonomic communities for all 32M papers (quick SQL UPDATE from arxiv_class)
4. Apply migration 009 to live DB

### Discovered: Full-Text Body Not Ingested

The ADS harvest scripts pull the `body` field (full paper text), and it's in all the JSONL files. **55% of papers have body text, median ~65K chars.** But the papers table has no `body` column and field_mapping.py doesn't map it. This is a major untapped asset for:

- Full-text search (tsvector on body)
- Chunk-and-embed for RAG-style retrieval
- Better entity extraction (body has methods/instruments that abstracts omit)
- The paper — changes "full-text is future work requiring ADS access" to "we already have it"

Action: add `body TEXT` column (migration 010), update field_mapping.py, re-run ingestion. Storage: ~2TB (1.3TB free on disk — tight). Consider separate `paper_bodies` table or external storage.

### Leiden OOM Fix

Previous attempt OOM'd at 43GB RSS running Leiden on 32M nodes. The graph + undirected copy + partition exceeds 62GB. Fix: run only on giant component. Script `scripts/graph_full.sh` Phase 2 attempts this but hasn't succeeded yet. With 99.6% edge resolution, the giant component may be 25-30M nodes — still large. Options:

- Cap with `maintenance_work_mem`
- Lower Leiden resolution parameters
- Run on a sampled subgraph

### Demo & Paper

- OpenAI embedding pilot (10K papers, $0.13)
- Entity extraction micro-pilot (100 papers, ~$2)
- 50-query retrieval benchmark
- Agent demo transcript
- Paper draft

## Environment

- **GPU**: RTX 5090 32GB, CUDA 13.1
- **Python**: `.venv/bin/python3` (torch 2.11.0+cu130) — NOT system python
- **torch install**: use `--index-url https://download.pytorch.org/whl/cu130`
- **DB**: PostgreSQL 16 + pgvector 0.8.0+, `dbname=scix`
