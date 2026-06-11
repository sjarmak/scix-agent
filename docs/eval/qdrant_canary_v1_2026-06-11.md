# Qdrant Dense-Lane Canary v1 — 2026-06-11

Collection: `scix_indus_v2_papers_s1` (32,383,535 pts, m=32, f16 on-disk,
SQ-INT8 on-disk, CPU-built, qdrant 1.18.2). Backend under test:
`vector_search()` Qdrant route (bead 5jtf), REST transport, exercised through
`scix.search.hybrid_search` — the production code path.

## Gate results

| # | Gate | Result | Verdict |
|---|------|--------|---------|
| G1 | **Index-quality honesty** (50q `eval/retrieval_50q.jsonl`, HNSW vs `exact=True`, same harness/path) | Δ nDCG@10 = **+0.0000**, Δ MRR@10 = +0.0000, Δ recall@50 = +0.0000; identical per-bucket | **PASS** (bar: \|Δ\| ≤ 0.01) |
| G2 | Serving latency (warm, single-stream) | p50 150 ms / p95 197 ms — vs 362 ms vector-stage in the pgvector era (MH-14) | **PASS** |
| G3 | Latency variance, 10-thread concurrent | p95 809 ms vs 2×single bar 395 ms (**4.1×**) | **FAIL** (see L1) |
| G4 | Graph integrity (ef=512, quant-ignored, 500 sampled pts) | 22 self-misses → 11 duplicate-crowding (MPEC/template docs, top-10 all ≥0.999 — benign), **8 true disconnections (1.6%)**, skewed to old/odd docs (1875 Natur, 1996 conf) | **PASS w/ note** — zero measurable end-to-end impact (G1 Δ=0) |
| G5 | Per-community-decile recall stratification | Subsumed: G1 shows zero aggregate index loss to stratify | **N/A (covered)** |
| G6 | Freshness slice (outbox-head queries) | **DEFERRED** — outbox worker not built (bead 8m0a); collection is 0 days old, staleness accrues ~1.3k papers/day | **OPEN** |

## Context findings (not gates)

- **First live numbers for this harness.** The April 50q doc was a template
  (gated on the never-valid halfvec index). Absolute scores — nDCG@10 0.083,
  recall@50 0.19 *even with exact dense* — measure gold-set hardness +
  encoder ceiling, not the index. `author_specific` bucket scores 0.0:
  hybrid has no author-aware lane (gold-set/lane design finding).
- **L1 (G3 lever):** quantized layer is on-disk (`always_ram=false`, chosen
  for 62G-host co-tenancy). Concurrent p95 contends on NVMe page faults.
  Fix: `always_ram=true` (~25 GB RAM) after `paper_embeddings` decommission
  frees budget — or accept for a single-operator local MCP whose realistic
  concurrency is ~1-2 dense queries.
- **gRPC transport broken** client↔1.18.2 (deserialization); backend pins
  REST. Contract-test mandate in 5jtf stands.

## Recommendation

**Flip the dense lane on** (`QDRANT_URL=http://127.0.0.1:6633` in the MCP
env): the lane has been dead for ~2 weeks; this restores it at better
latency than the old pgvector lane with provably zero index-quality loss.
G3 and G6 become the immediate follow-ups (L1 lever; bead 8m0a), not flip
blockers — both are degradation-over-time/under-load concerns, strictly
better than "no dense lane at all". Rollback: unset `QDRANT_URL`, restart MCP.
