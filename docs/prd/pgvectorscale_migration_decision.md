# pgvectorscale Migration Decision

## Context

SciX currently runs 32.4M INDUS (768d) vectors in a pgvector 0.8.2 HNSW
index on the production `scix` database. HNSW is RAM-resident and hybrid
retrieval p95 sits around 80 ms today, but three pressures are
compounding: (1) filtered queries (year, arxiv_class) rely on
`iterative_scan` as a workaround rather than native label filtering,
(2) the current corpus already sits 6x past the documented 5M-vector
HNSW sweet spot, so any additional embedding work (second model,
chunk-level embeddings) risks pushing the index past RAM and into steep
latency cliffs, and (3) CLAUDE.md calls out pgvectorscale
StreamingDiskANN as the recommended path at 30M+ vectors, citing large
latency and cost wins that have not been validated on our own hardware
and eval set.

The parent research PRD
([docs/prd/prd_pgvectorscale_migration_benchmark.md](./prd_pgvectorscale_migration_benchmark.md))
carves the benchmark work: install pgvectorscale in an isolated
`scix_pgvs_pilot` database, copy the 32M INDUS vectors in, build an
HNSW baseline and three StreamingDiskANN variants (V1 halfvec-only,
V2 default + SBQ, V3 tuned + SBQ), and run the 50-query eval and
filtered-query harness against each.

This document is the decision output of that benchmark. It is written
**before** the measured numbers exist so that the PASS / FAIL bars are
committed up front and cannot drift once results land. Placeholder
values are marked `TBD` and must be filled in from the JSON outputs in
`results/pgvs_benchmark/` before the doc concludes.

## Decision Criteria

A pgvectorscale StreamingDiskANN variant is "go" only if it satisfies
**all three** numeric bars below. Any variant that fails even one bar
is out; failing all three across all variants lands us on the No-Go
Path.

- **C1. Unfiltered nDCG@10 within 1% of HNSW baseline.** The candidate
  variant's nDCG@10 on the 50-query unfiltered eval must satisfy
  `|Δ nDCG@10| <= 0.01` against the HNSW baseline. This is the SBQ
  honesty gate — binary-quantization latency wins do not offset a
  retrieval-quality regression on scientific text.
- **C2. Filtered p95 latency at least 30% lower than HNSW.** On both
  the F1 (`year = 2024`, ~10% selectivity) and F2
  (`arxiv_class && '{astro-ph.GA,astro-ph.SR}'`, ~20% selectivity)
  filter configurations, the candidate variant must show
  `>= 30%` p95 latency reduction vs the HNSW baseline (which relies
  on `iterative_scan`).
- **C3. Index size <= 1.5x HNSW.** On-disk size of the candidate
  variant's index (as reported by `pg_total_relation_size` on the
  index relation) must be no more than `1.5x` the HNSW baseline's
  on-disk size.

The three bars are intentionally orthogonal — quality (C1), latency
under real filter traffic (C2), storage cost (C3) — so "two of three"
is explicitly not enough.

## Results (TO FILL IN)

All numbers below are placeholders until the benchmark scripts land
their JSON outputs in `results/pgvs_benchmark/`.

### Unfiltered retrieval quality

Source: `scripts/bench_pgvectorscale.py` →
`results/pgvs_benchmark/retrieval_quality.json`.

| Variant             | nDCG@10 | ΔnDCG@10 vs HNSW | Recall@10 | Recall@20 | MRR  | p50 (ms) | p95 (ms) | C1 PASS/FAIL (within 1%) |
| ------------------- | ------- | ---------------- | --------- | --------- | ---- | -------- | -------- | ------------------------ |
| HNSW baseline       | TBD     | 0.0 (reference)  | TBD       | TBD       | TBD  | TBD      | TBD      | reference                |
| DiskANN V1 (halfvec)| TBD     | TBD              | TBD       | TBD       | TBD  | TBD      | TBD      | TBD                      |
| DiskANN V2 (+SBQ)   | TBD     | TBD              | TBD       | TBD       | TBD  | TBD      | TBD      | TBD                      |
| DiskANN V3 (tuned+SBQ)| TBD   | TBD              | TBD       | TBD       | TBD  | TBD      | TBD      | TBD                      |

### Filtered p95 latency

Source: `scripts/bench_pgvectorscale_filtered.py` →
`results/pgvs_benchmark/filtered_queries.json`.

| Variant              | F1 p95 (ms) year=2024 | F2 p95 (ms) arxiv_class | % reduction vs HNSW (F1) | % reduction vs HNSW (F2) | C2 PASS/FAIL (>= 30%) |
| -------------------- | --------------------- | ----------------------- | ------------------------ | ------------------------ | --------------------- |
| HNSW baseline        | TBD                   | TBD                     | 0 (reference)            | 0 (reference)            | reference             |
| DiskANN V1 (halfvec) | TBD                   | TBD                     | TBD                      | TBD                      | TBD                   |
| DiskANN V2 (+SBQ)    | TBD                   | TBD                     | TBD                      | TBD                      | TBD                   |
| DiskANN V3 (tuned+SBQ)| TBD                  | TBD                     | TBD                      | TBD                      | TBD                   |

### Index size comparison

Source: `scripts/build_hnsw_baseline.py`,
`scripts/build_streamingdiskann_variants.py` →
`results/pgvs_benchmark/hnsw_baseline.json` and
`results/pgvs_benchmark/streamingdiskann_builds.json`.

| Variant              | Build wall-time | Peak RSS | On-disk bytes | Total relation size | Ratio vs HNSW | C3 PASS/FAIL (<= 1.5x) |
| -------------------- | --------------- | -------- | ------------- | ------------------- | ------------- | ---------------------- |
| HNSW baseline        | TBD             | TBD      | TBD           | TBD                 | 1.0x (ref)    | reference              |
| DiskANN V1 (halfvec) | TBD             | TBD      | TBD           | TBD                 | TBD           | TBD                    |
| DiskANN V2 (+SBQ)    | TBD             | TBD      | TBD           | TBD                 | TBD           | TBD                    |
| DiskANN V3 (tuned+SBQ)| TBD            | TBD      | TBD           | TBD                 | TBD           | TBD                    |

### Concurrent stress (informational)

Source: `scripts/bench_pgvectorscale_concurrent.py` →
`results/pgvs_benchmark/concurrent_stress.json`. Not part of the
numeric go/no-go — used to sanity-check that the winning variant does
not fall apart under 10- and 50-thread load.

| Variant              | Threads | Sustained QPS | p50 (ms) | p95 (ms) | p99 (ms) |
| -------------------- | ------- | ------------- | -------- | -------- | -------- |
| HNSW baseline        | 10      | TBD           | TBD      | TBD      | TBD      |
| HNSW baseline        | 50      | TBD           | TBD      | TBD      | TBD      |
| DiskANN winner       | 10      | TBD           | TBD      | TBD      | TBD      |
| DiskANN winner       | 50      | TBD           | TBD      | TBD      | TBD      |

### Cold-start (informational)

Source: `scripts/bench_pgvectorscale_coldstart.py` →
`results/pgvs_benchmark/cold_start.json`. Also informational — we care
whether StreamingDiskANN's SSD-backed design recovers from a Postgres
restart faster than a cold HNSW mmap.

| Variant              | Cold query 1 (ms) | Cold query 5 (ms) | Cold query 10 (ms) | Warm steady-state p50 (ms) |
| -------------------- | ----------------- | ----------------- | ------------------ | -------------------------- |
| HNSW baseline        | TBD               | TBD               | TBD                | TBD                        |
| DiskANN winner       | TBD               | TBD               | TBD                | TBD                        |

### Selected variant

`TBD` — fill in with the single variant (V1 / V2 / V3) that clears all
three bars, or `none` if the decision lands on No-Go.

## Go Path

Triggered only if one specific pgvectorscale variant clears **C1 and
C2 and C3**. If go, this section becomes the seed of the migration
runbook and `docs/prd/prd_pgvectorscale_migration_build.md` gets
fleshed out from its stub.

### Migration plan sketch

1. **Production extension install.** Run
   `CREATE EXTENSION vectorscale;` on the production `scix` database
   in a scheduled maintenance window. Capture the exact version pin
   (`SELECT extversion FROM pg_extension WHERE extname='vectorscale';`)
   and record it alongside the Postgres major version in the ops log.
2. **Build the new index concurrently.** Use `CREATE INDEX
   CONCURRENTLY` (or the pgvectorscale equivalent) with the exact
   parameter set from the winning variant on the `paper_embeddings`
   table, `WHERE model_name='indus'`. Estimated wall-clock on 32M
   rows: `TBD` (fill in from the pilot build-time number in the Index
   Size table above). Disk headroom: `TBD` (use the on-disk-bytes
   number).
3. **Dual-read cutover window.** Ship a flag in `src/scix/search.py`
   that lets the dense retrieval path hit either the HNSW index or
   the new DiskANN index. Default to HNSW, then flip a small % of
   traffic (e.g., 5% → 25% → 100%) over a **one-week** dual-read
   window while monitoring nDCG@10 on the live 50-query eval, p95
   latency, and error rate.
4. **Monitoring period.** During dual-read, watch: (a) absolute p95
   latency on the new index vs the HNSW baseline captured on the same
   day, (b) any query whose top-10 differs substantially between the
   two indexes (delta-set logging), (c) extension-level errors in
   `pg_stat_statements` and Postgres logs.
5. **Final cutover.** After a clean dual-read week with no quality
   regression, flip the flag to DiskANN-only, leave the HNSW index
   intact for one additional week as a hot-swap fallback, then drop
   it in a follow-up window.

### Rollback plan

- **Trigger:** any of (i) live nDCG@10 drops >1% over a rolling
  24-hour window, (ii) p95 latency exceeds the HNSW baseline by >10%
  for >1 hour, (iii) any extension-level crash.
- **Action:** flip the `src/scix/search.py` flag back to HNSW. The
  HNSW index stays built and valid for one week post-cutover
  precisely so rollback is a config change, not a rebuild.
- **Time budget:** rollback must be achievable in **<= 30 min** from
  trigger detection to 100% HNSW traffic. The flag change is
  runtime; no deploy required.
- **Post-rollback:** open an incident ticket, capture the failure
  signature, and decide whether to re-benchmark (see escalation
  trigger in No-Go Path) or drop the DiskANN index entirely.

### Operational risk

- **Extension-version coupling.** pgvectorscale is a Rust extension
  built against a specific Postgres major version. Any future Postgres
  major-version upgrade must be preceded by confirming pgvectorscale
  ships a compatible build, and the upgrade runbook must install the
  matching vectorscale version in the new cluster before restoring
  data. Pin the version in infrastructure-as-code.
- **Backup compatibility.** `pg_dump` and `pg_basebackup` must capture
  the extension state. Verify in the pilot that a `pg_dump` / restore
  round-trip preserves the DiskANN index (or that a post-restore
  rebuild is fast enough to be acceptable). Document the answer in
  the migration-build PRD.
- **Upstream dependency.** pgvectorscale is Apache 2.0, actively
  maintained by Timescale. It does **not** require TimescaleDB itself
  — vectorscale stands alone. Still, we take on one more extension to
  keep in sync across Postgres upgrades.

### Estimated wall-clock

- Build new index on 32M rows: `TBD` (from benchmark build time).
- Dual-read period: 1 week default.
- Final cutover window: single maintenance window, ~30 min.
- HNSW index retention post-cutover: 1 week, then drop.

### Next step after go

`docs/prd/prd_pgvectorscale_migration_build.md` (currently a stub)
gets fleshed out: reserved migration number confirmed, explicit SQL
migration files drafted, `src/scix/search.py` dual-read flag spec'd,
rollback script committed, monitoring dashboard linked.

## No-Go Path

Triggered if no pgvectorscale variant clears all three bars.

### Cost of staying on HNSW

- **RAM-bound growth ceiling.** HNSW quality degrades sharply once
  the index spills from RAM. 32M INDUS vectors fit today; adding a
  second embedding model or chunk-level embeddings likely does not.
  Staying on HNSW means accepting a hard ceiling on embedding
  expansion until a re-benchmark.
- **iterative_scan overhead on filtered queries.** Filtered-query p95
  is already a known soft spot; this path stays soft. Agent traffic
  that leans heavily on year or arxiv_class filters will feel it
  first.
- **Rebuild cost on parameter tuning.** Any future HNSW tuning
  (`m`, `ef_construction`) on 32M vectors is a multi-hour index
  rebuild. DiskANN would have given us cheaper per-parameter
  iteration.

### Next escalation trigger

Document the explicit conditions under which to re-benchmark rather
than leaving the question open:

- **Scale trigger.** Re-run the benchmark when total INDUS vector
  count exceeds **50M** (planned embedding-model or chunk-embedding
  expansion would cross this line).
- **Latency-SLO trigger.** Re-run the benchmark if filtered-query p95
  exceeds the current SLO for **>1 week** sustained.
- **Upstream trigger.** Re-run the benchmark on any pgvectorscale
  release that claims material SBQ quality improvements, since SBQ
  retrieval quality is the bar the current benchmark is most likely
  to fail.

### Documented negative result

If this section is the active path, link to the concrete benchmark
results (`results/pgvs_benchmark/retrieval_quality.md`,
`filtered_queries.md`) and summarize which bar(s) each variant
missed. That summary becomes the evidence cited by the next
escalation when re-benchmark is triggered — "we already ran this,
here is what changed."

## Runbook References

- [docs/ops/pgvectorscale_install.md](../ops/pgvectorscale_install.md)
  — install + pilot DB bootstrap.
- `scripts/bench_pgvectorscale.py` — unfiltered quality bench.
- `scripts/bench_pgvectorscale_filtered.py` — filtered bench
  (F1 year, F2 arxiv_class).
- `scripts/bench_pgvectorscale_concurrent.py` — concurrent stress.
- `scripts/bench_pgvectorscale_coldstart.py` — cold-start bench.
- `scripts/build_hnsw_baseline.py` — HNSW baseline build.
- `scripts/build_streamingdiskann_variants.py` — DiskANN V1/V2/V3
  build.
- Parent PRD:
  [docs/prd/prd_pgvectorscale_migration_benchmark.md](./prd_pgvectorscale_migration_benchmark.md).
- Follow-up build PRD (stub):
  [docs/prd/prd_pgvectorscale_migration_build.md](./prd_pgvectorscale_migration_build.md).
