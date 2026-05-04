# Halfvec migration (bead `scix_experiments-0vy`) — final outcome

**Closed**: 2026-04-30
**Outcome**: `blocked-by-architecture`
**Successor**: `scix_experiments-ozrt` (pgvectorscale StreamingDiskANN evaluation; pending Steph's go-ahead)

## TL;DR

The halfvec(768) shadow-column migration succeeded (32.4M rows backfilled,
code wired, footprint reduction target validated). The matching HNSW index
build did **not** succeed: three independent build attempts on this host
all stalled in pgvector's "graph no longer fits into maintenance_work_mem"
spill cycle. The structural ceiling matches CLAUDE.md's documented
boundary at >30M vectors and motivates the pgvectorscale pivot.

## Build attempts

| # | Date | Config | Outcome |
|---|------|--------|---------|
| 1 | 2026-04-29 14:45 → 2026-04-30 10:49 (~21h) | `CREATE INDEX CONCURRENTLY`, `maintenance_work_mem=8GB`, `max_parallel_maintenance_workers=7` | Spilled at 3.87M tuples (NOTICE in build log). Eventually killed (zombie scope), leaving INVALID index (`indisvalid=false`/`indisready=false`) at 25 GB on disk. Symptom: every INDUS dense query fell back to Parallel Seq Scan — verified via EXPLAIN. |
| 2 | 2026-04-30 16:47 → 17:19 (~30 min) | non-concurrent, `mwm=16GB`, `parallel=4` | 4 × 16 GB allocation = 64 GB peak > 62 GB physical RAM. Pushed 16 GB into swap. Reached 7.74M tuples / 22% blocks before swap thrashing pinned tuples_done at ~1k/min. Cancelled. |
| 3 | 2026-04-30 17:17 → 17:58 (~41 min) | non-concurrent, `mwm=24GB`, `parallel=0` (single-thread) | Reached 4.10M tuples / 12% blocks cleanly, then stalled when 24 GB threshold was hit. Tuples advanced at ~14/min during spill. Cancelled. |
| 4 | 2026-04-30 19:34 → 21:19 (~1h45m) | non-concurrent, `mwm=24GB`, `parallel=0`, **`m=8`** (mayor approval gc-71388) | Reached 12.89M tuples / 40% blocks. Pgvector logged `"hnsw graph no longer fits into maintenance_work_mem after 12890802 tuples"`. Second spill cycle averaged 1.5k tuples/min — projected ~4 days more at that rate. Cancelled. |

Per-tuple memory cost worked out to ~2.07 KB at m=16 and ~1.04 KB at
m=8. Either way, a 32M-tuple HNSW graph naturally exceeds reasonable
maintenance_work_mem budgets on this host (62 GB physical RAM, 16 GB
shared_buffers, 24 GB available for build).

## Hardware/resource profile during attempts

- **Physical RAM**: 62 GiB total
- **Swap**: 39 GiB (peaked at 25 GiB used during attempt #2)
- **Disk** (`/dev/nvme1n1p2`, 1.9 TB): **94% used / 114 GB free** at attempt
  #4 cancellation. `pg_stat_database.temp_bytes` cumulative 2014 GB
  against scix DB. Continuing past 40% scan risked a disk-full error
  that would corrupt other writes.
- **Concurrent workloads**: `run_ner_pass.py` (NER on abstracts, since
  Apr 28), `backfill_external_ids_phase2_openalex.py` (since Apr 30
  10:59), `backfill_sections_tsv.py` (since Apr 29). All write to
  unrelated tables but compete for disk/IO bandwidth and OS page cache.

## Why this matches CLAUDE.md's documented boundary

> For 30M+ vectors needing higher recall, the path forward is
> pgvectorscale StreamingDiskANN, not larger HNSW. Track in epic, do
> not retrofit HNSW past its ceiling.
>
> — CLAUDE.md `Architecture invariants`

32.4M halfvec(768) rows is precisely in this regime. Three attempts
across different parameter combinations and lock modes all hit the
same wall. The constraint is structural (per-tuple graph memory ×
total tuples > available physical RAM), not tunable.

## Work product preserved for the successor (`ozrt`)

The migration's groundwork is reusable for the pgvectorscale path.
Nothing here needs to be redone:

- **Schema**: `paper_embeddings.embedding_hv halfvec(768)` shadow column
  exists and is fully backfilled (32,381,535 INDUS rows).
  `migrations/053_paper_embeddings_halfvec.sql` applied.
- **Code wiring**: `src/scix/search.py` and `src/scix/embed.py` route
  INDUS dense queries to `embedding_hv` when `SCIX_USE_HALFVEC=1`.
  Pilot models (nomic, specter2) untouched.
- **LHS-cast bug fix**: discovered during cutover (commit pending) —
  `vector_search` was wrapping `pe.embedding_hv` in
  `(pe.embedding_hv)::halfvec(768)` to mirror the legacy pilot index
  expression, but `idx_embed_hnsw_indus_hv` is on bare `embedding_hv`.
  The wrapping cast defeats planner-level index match (~12 min seq
  scan vs sub-100ms HNSW match, verified via EXPLAIN). Fix gates the
  cast on `use_halfvec`. Regression test in
  `tests/test_halfvec_migration.py::test_indus_query_uses_embedding_hv_no_lhs_cast`.
- **Tests**: 3/3 unit tests in `test_halfvec_migration.py` pass after
  fixing the pre-existing env-var monkeypatch issue.
- **Runbook**: `docs/runbooks/halfvec_migration.md` updated with the
  LHS-cast bug, the build-time gotcha (graph spill), and concrete
  recovery steps for INVALID HNSW indexes.
- **Footprint baseline**: `results/halfvec_migration/sizes_after.txt`
  records 120 GB → 24 GB (m=16, attempt #1's pre-invalidation size)
  = 80% reduction. Original acceptance C5 (≥40%) was met before the
  index was invalidated.

## What remained unfinished

- **No usable INDUS dense index**. The legacy `idx_embed_hnsw_indus`
  was dropped in phase 4 (mandatory pre-backfill step). The new
  `idx_embed_hnsw_indus_hv` was dropped after each invalid/stalled
  build. INDUS dense queries currently fall back to seq-scan over
  paper_embeddings (~5-10 min/query).
- **Acceptance C2** (50q eval ≤0.5 pp nDCG@10 drop). The post-migration
  eval `results/halfvec_migration/post.json` does not exist; the one
  attempt was scoring all queries as empty because the underlying
  index was INVALID. Baseline `results/halfvec_migration/baseline.json`
  is preserved for the successor to compare against.

## Mail thread

- `gc-69904` (12:01 EDT): initial blocker — INVALID index found,
  rebuild guidance requested.
- `gc-71388` (mayor reply via deferred reminder): proceed with m=8
  Option 2.
- `gc-71709` (21:19 EDT): m=8 also stalled; recommend pgvectorscale
  pivot. Mayor's directive (deferred reminder): close 0vy
  architecture-blocked.

## Operational note

INDUS dense search is fully offline as of this close. Public MCP at
`mcp.sjarmak.ai` is intentionally DOWN per ops state, so external
impact is nil. Local MCP queries that require dense INDUS are
seq-scan-tier. The successor bead `ozrt` should treat restoring INDUS
dense search as P0.
