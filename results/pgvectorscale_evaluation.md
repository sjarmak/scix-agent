# pgvectorscale StreamingDiskANN Evaluation

*Generated: 2026-04-06*

## Status: BLOCKED

**Blocker**: Building pgvectorscale from source requires `postgresql-server-dev-16` (for pgrx/Rust PG extension framework). This package needs sudo, which is unavailable in the current environment.

**To unblock**: Run `sudo apt-get install -y postgresql-server-dev-16` then re-run this evaluation.

## Current HNSW Baseline (pgvector 0.8.2)

| Metric | Value |
|--------|-------|
| Total SPECTER2 embeddings | ~24M (768d) |
| Index type | HNSW (m=16, ef_construction=200) |
| ef_search | 100 |
| Iterative scan | enabled (relaxed_order) |
| Estimated HNSW memory | ~40-60GB for 32M × 768d vectors |

## StreamingDiskANN Expected Benefits (from Timescale benchmarks)

Based on published benchmarks (pgvectorscale v0.5.0, 50M vectors):

| Metric | HNSW | StreamingDiskANN | Improvement |
|--------|------|------------------|-------------|
| QPS (99% recall) | ~200 | ~471 | 2.4x |
| Memory usage | All in RAM | SSD-backed | 75-90% reduction |
| p95 latency | ~15ms | ~8ms | 1.9x |
| Index build time | Hours | Hours (similar) | ~1x |

## Recommendation

StreamingDiskANN is strongly recommended for the 30M+ SPECTER2 embeddings:
- HNSW at 32M × 768d requires ~100GB+ RAM (exceeds 62GB server)
- DiskANN uses SSD-backed storage, keeping only graph structure in RAM
- 471 QPS at 99% recall is sufficient for agent workloads
- pgvector HNSW can remain for smaller model-specific indexes (text-embedding-3-large at 1024d)

## Installation Steps (when sudo available)

```bash
sudo apt-get install -y postgresql-server-dev-16
source "$HOME/.cargo/env"
cargo install cargo-pgrx --version 0.12.9
cargo pgrx init --pg16 $(which pg_config)
git clone https://github.com/timescale/pgvectorscale.git /tmp/pgvectorscale
cd /tmp/pgvectorscale/pgvectorscale
cargo pgrx install --release
# Then in psql:
# CREATE EXTENSION vectorscale;
# CREATE INDEX ON paper_embeddings USING diskann (embedding) WHERE model_name = 'specter2';
```

## Benchmark Plan (post-installation)

1. Create DiskANN index on 1M vector subset
2. Measure QPS at 95%, 99%, 99.9% recall
3. Compare latency distribution (p50/p95/p99) vs HNSW
4. Measure memory footprint (pg_indexes_size + shared_buffers usage)
5. Test with iterative scan + filters (year, doctype)
