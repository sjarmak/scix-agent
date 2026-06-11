# ADR-013: Serve the INDUS Dense Lane from Qdrant; Supersede the Single-Postgres-Substrate Pin for Vector ANN

- **Status**: Accepted (2026-06-11) — flipped to production the same day (`QDRANT_URL` in `.mcp.json` + global config), after canary `results/qdrant_canary_v1.md`.
- **Deciders**: Stephanie Jarmak (operator approval at every external step); investigation + execution in the 2026-06-08→11 dense-lane-restoration session.
- **Scope**: kNN serving for `model_name='indus'` in `vector_search()` (`src/scix/search.py`). Postgres remains authoritative for **everything else**: papers, BM25 (both lanes), citation graph, entities, sessions, and the `paper_embeddings` table itself (rollback source of truth through the 30-day soak).
- **Related beads**: `kj37` (closed — dense lane restored), `12rp` (DDL/benchmark divergence, P1), `qlvv` (pgvectorscale parallel-build validation), `pkcd` (Qdrant pilot + GPU A/B), `5jtf` (gated backend), `o9ib` (canary), `8m0a` (outbox sync).
- **Related ADRs**: ADR-008 (Qdrant payload schema — now applies to the v2 collection lineage).
- **Supersedes**: the CLAUDE.md retrieval-stack line "single PostgreSQL + pgvector is the whole substrate — no separate vector DB", **for the dense-ANN serving role only**.

## Context

The corpus (32.4M INDUS 768d vectors) sits ~6× past pgvector HNSW's
documented comfort zone. The serving HNSW index (120 GB, RAM-resident by
design on a 62 GB host) worked for months in a degraded-but-acceptable mode
(~362 ms vector stage, MH-14) by riding the OS page cache. The planned
successor was pgvectorscale StreamingDiskANN (kj37), chosen because it was
the only scale path that preserved the single-substrate pin.

The cutover failed, expensively. Causal chain (each link verified in-session):

1. **Schema landmine**: `paper_embeddings.embedding` is dimensionless
   `vector` (multi-model table design). DiskANN requires a fixed dimension →
   the rebuild DDL cast `(embedding)::vector(768)` → a true **expression
   index**, which pgvectorscale **cannot scan** (`assertion failed:
   attnum > 0`, reproduced at 50k rows in seconds). A properly-typed
   `halfvec(768)` column (`embedding_hv`, migs 053/054, fully backfilled)
   existed but the rebuild script was not written against it.
2. **Benchmark/production divergence**: the pgvs pilot validated
   `halfvec_cosine_ops` on a properly-typed pilot table; prod ran a different
   opclass on a different storage type. The go/no-go numbers described a
   configuration production never ran (bead 12rp).
3. **No scannability validation**: no scratch-index query was run before
   committing to a 56 h build. The index built `valid=t` and crashed every
   scan. The planner auto-selected it, so it would have broken (not merely
   not helped) all dense queries.
4. **Extension immaturity at our envelope**: 0.9.0's parallel build never
   engages on this install (even force-GUC'd; suspected
   `shared_preload_libraries` requirement — bead qlvv); builds are
   single-threaded with builder-cache spill (I/O-bound death-crawl for the
   final 40%); gRPC clients fail deserializing 1.18-era responses.
5. **One-way door**: the serving HNSW index was dropped before the
   replacement was validated → ~2 weeks of lexical-only serving, whose
   measured cost is documented in `results/litreview_rerun_2026-06-11.md`
   (two lit reviews materially changed by the missing dense lane).

Replacement evaluated the same day (bead pkcd): Qdrant 1.18.2, local NVMe,
CPU-built HNSW (GPU/Vulkan builds produce ~0.5–0.75% disconnected nodes on
the 5090 — measured A/B, GPU rejected), f16 on-disk originals + SQ-INT8.

## Decision

1. **Dense kNN for INDUS serves from Qdrant** (`scix_indus_v2_papers_s1`,
   127.0.0.1-only, docker `--memory 40g`, NVMe storage — never NAS).
   `vector_search()` routes via the `QDRANT_URL` env gate; REST transport
   pinned; rollback = unset the var and restart the MCP.
2. **Postgres `paper_embeddings` remains the source of truth.** Qdrant is a
   derived cache: full rebuild from PG measured at **3.2 h** (vs 56 h for the
   failed DiskANN build), making rebuild-from-source a real recovery
   strategy. Decommission of pgvector serving artifacts only after the
   PRD's 30-day soak.
3. **CPU index builds only** until the GPU disconnected-node defect is
   resolved upstream (repro: `scripts/qdrant_gpu_ab.py`).
4. **Sections/chunks lanes target Qdrant collections**, not pgvector
   (`zsbd` rerouted) — payload-filtered HNSW is precisely the
   methodology-search requirement pgvector served worst (`iterative_scan`).

## Evidence (canary, 2026-06-11)

| Gate | Result |
|---|---|
| Quality: 50q hybrid, HNSW vs `exact=True`, production code path | **Δ = +0.0000** on nDCG@10 / MRR / recall@50, every bucket |
| Warm serving latency | p50 150 ms (vs 362 ms pgvector-era) |
| 10-thread p95 variance | **FAIL** (4.1× vs 2× bar) — lever: `always_ram` for SQ-INT8 (~25 GB) post-decommission |
| Graph integrity at 32M | 1.6% true disconnection (old/odd docs) + benign duplicate-crowding; zero end-to-end impact |
| Freshness | open until outbox (`8m0a`) lands; staleness accrues ~1.3k papers/day |

## Consequences

- Two stateful systems on a single-operator host. Admission price (from the
  migration premortem, now mandatory): rebootability drill (M4), pinned
  `qdrant-client` + contract test (gRPC break proved the point), outbox lag
  visibility, dated decommission calendar.
- The single-substrate pin survives for everything non-ANN. Returning dense
  to Postgres remains possible (PRD MH-12) if pgvectorscale matures *and* a
  fixed-dim column migration is done — revisit only with the validation
  rules below.
- CLAUDE.md retrieval-stack section must be updated to reference this ADR.

## Validation rules this failure bought (binding for future index/lane work)

1. **No index is trusted until one query has returned from it.** Any new
   index type/config gets a ≤50k-row scratch build + forced-index-scan smoke
   test *before* any multi-hour build.
2. **Benchmark DDL must be byte-identical to production DDL** (opclass,
   column type, storage params), or the benchmark is treated as fiction.
3. **Never drop a serving index before its replacement is validated
   scannable** — disk pressure is solved some other way first.
4. **Architecture pins are re-examined when they start selecting immature
   components.** "The only option that preserves the pin" is a trigger to
   question the pin, not to accept the option.
