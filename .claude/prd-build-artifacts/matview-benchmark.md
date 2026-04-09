# Materialized View Scale Benchmark

**Date**: 2026-04-06 16:35 UTC
**Total benchmark time**: 181s

## Synthetic Data Scale

| Table | Target Rows |
|-------|------------|
| entities | 1,000,000 |
| document_entities | 10,000,000 |
| entity_aliases | 500,000 |
| entity_identifiers | 1,200,000 |
| entity_relationships | 200,000 |
| datasets | 5,000 |
| dataset_entities | 50,000 |
| document_datasets | 100,000 |
| papers (synthetic) | 2,000,000 |

## Results

| View | Rows | CREATE (s) | REFRESH CONCURRENTLY (s) | Query Latency (ms) | Status |
|------|------|-----------|-------------------------|-------------------|--------|
| agent_document_context | 2,000,000 | 19.4 | 22.9 | 0.03 | PASS |
| agent_entity_context | 1,000,000 | 4.4 | 6.6 | 0.02 | PASS |
| agent_dataset_context | 5,000 | 2.7 | 2.8 | 0.03 | PASS |

## Verdict

All materialized views complete REFRESH CONCURRENTLY within 30 minutes at target scale. Materialized views are viable for agent context.

## Fallback Strategy (Contingency)

Documented unconditionally so downstream beads can adopt without re-running the benchmark if production scale or row distribution changes the picture. Apply if any REFRESH CONCURRENTLY at production scale exceeds 30 min, or if production row counts grow materially beyond the 10M document_entities tested here.

1. **Incremental summary tables**: trigger-maintained summary tables updated on INSERT/UPDATE/DELETE (no full refresh; constant per-row cost). Best when write rate is moderate and read latency must be sub-ms.
2. **Partitioned refresh**: partition `document_entities` by `entity_type` or by hash of `bibcode`, build per-partition matviews, refresh partitions independently in parallel. Cuts wall-clock refresh by N-way parallelism and isolates hot partitions.
3. **Partial materialization**: only materialize frequently-queried slices (e.g., top 100K entities by doc_count, or only the last 5 years of papers). Combine with on-demand JOIN for the long tail.
4. **pgvectorscale StreamingDiskANN for vector columns**: if any agent context view embeds dense vectors and memory pressure becomes a bottleneck, switch the vector index to pgvectorscale (SSD-backed, ~6 GB RAM for 1M vectors at 768d vs. ~100 GB for HNSW in memory).
5. **Logical replication subscriber**: replicate the source tables to a read-only subscriber and build matviews there to remove refresh contention from the primary write path.

### Selection guidance

- Refresh time linear in input rows: prefer #2 (partitioned).
- Refresh time dominated by JSONB aggregation: prefer #1 (incremental triggers).
- Long tail of low-traffic entities: prefer #3 (partial).
- Vector memory pressure: layer #4 on top of any of the above.
- Refresh blocks live writes: layer #5 on top of any of the above.

## Observations

- `agent_document_context` is the largest view (one row per paper with aggregated entities)
- `agent_entity_context` uses LATERAL subquery for doc count to avoid cross-join explosion
- `agent_dataset_context` is smallest due to limited dataset count
- All views use JSONB aggregation with DISTINCT to avoid duplicates from multi-way JOINs
- UNIQUE INDEX on primary key enables REFRESH CONCURRENTLY

## Reproducibility

```bash
python scripts/matview_benchmark.py
```

Benchmark runs in an isolated `matview_bench` schema and cleans up after itself.
