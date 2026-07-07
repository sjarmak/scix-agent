---
name: scix-vector-serving-qdrant
description: >
  The INDUS dense lane's serving substrate: the Qdrant collection
  scix_indus_v2_papers_s1, the QDRANT_URL gate in vector_search(), the
  REST-only client pin, the bibcode-only payload + Postgres post-filter
  over-fetch pattern, SCIX_QDRANT_EXACT, point-id contracts, ADR-013 and why
  dense ANN does NOT serve from pgvector/pgvectorscale (the DiskANN
  catastrophe), the ADR-014 sparse lane (parked, not dead), and the
  ADR-015/016 storage line as it bears on the dense lane. Load when working on
  Qdrant collections, vector kNN serving, the dense-lane gate, Qdrant payload
  schema/backfills, the scix-qdrant container, or anything that would put
  dense ANN back in Postgres. NOT for the RRF fusion stack or lane weighting —
  use scix-retrieval-architecture. NOT for the embed/ingest pipeline or the
  s7cy daily-sync fire — use scix-embedding-pipeline. NOT for Postgres index
  builds, disk reclamation mechanics, or NAS placement rules — use
  scix-index-and-storage-discipline.
---

# SciX vector serving on Qdrant

Facts dated 2026-07-07, verified read-only against commit `452ab86` on branch
`bd/0yp5-external-copy-accuracy-audit` (not `main`; the repo's git history is
truncated — pre-2026-06-11 decisions live only in ADRs, PRDs, and beads).

**Jargon, once:** _lane_ = one retrieval signal fused into hybrid search by
RRF (Reciprocal Rank Fusion). _Dense lane_ = kNN (k-nearest-neighbor) over
768-dimensional INDUS embeddings. _ANN_ = approximate nearest neighbor.
_HNSW_ = the graph-based ANN index type. _Payload_ = the JSON metadata Qdrant
stores per point. _Over-fetch_ = asking Qdrant for more hits than needed so a
downstream filter still leaves `limit` survivors. _halfvec_ = pgvector's
float16 column type. _SQ-INT8_ = Qdrant scalar quantization to 8-bit ints.
_Soak_ = a mandated observation window before destructive cleanup. _Outbox_ =
a Postgres queue table draining writes to a second system.

## When not to use this skill

| You want                                                     | Use instead                                                    |
| ------------------------------------------------------------ | -------------------------------------------------------------- |
| How RRF fuses lanes, k=60, lane weighting, eval numbers      | `scix-retrieval-architecture`                                  |
| Running/repairing the embed pipeline, the s7cy live fire     | `scix-embedding-pipeline`                                      |
| Postgres index build discipline, disk reclamation, NAS rules | `scix-index-and-storage-discipline`                            |
| Querying the system as an agent (MCP tools)                  | `scix-mcp` (query-side; parts are stale — see doc-drift table) |
| Changing any ADR-pinned axis                                 | `scix-change-control` first, always                            |

## The dense lane at a glance

| Axis                   | Value                                                                  | Source                                                                                             |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Serving collection     | `scix_indus_v2_papers_s1`                                              | `src/scix/search.py:51`                                                                            |
| Gate                   | `QDRANT_URL` env set AND model is `indus`                              | `search.py:59-65` (`_qdrant_dense_gated`)                                                          |
| Live endpoint          | `http://127.0.0.1:6633` (this installation)                            | `.mcp.json`, `docs/eval/qdrant_canary_v1_2026-06-11.md`                                            |
| Transport              | REST only, `prefer_grpc=False`, client timeout 30 s                    | `search.py:79`; gRPC read path breaks (below)                                                      |
| Point count at cutover | 32,383,535 (2026-06-11)                                                | canary doc; runtime count unverified read-only                                                     |
| Collection config      | m=32, f16 on-disk originals, SQ-INT8 on-disk, CPU-built, Qdrant 1.18.2 | canary doc                                                                                         |
| Point id               | `str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode))`                         | `scripts/qdrant_full_load.py:116`, `scripts/qdrant_outbox_sync.py:106`                             |
| Payload                | `{"bibcode": <bibcode>}` only — nothing else                           | `_vector_search_qdrant` docstring; enrichment parked (bead 2xi8)                                   |
| Vector                 | single unnamed 768d vector (NOT named)                                 | `qdrant_full_load.py`; ADR-014 notes mixing in a named sparse vector requires a collection rebuild |
| Exact-mode control     | `SCIX_QDRANT_EXACT=1` (eval only)                                      | `search.py:552-557`                                                                                |
| Dimensionality         | 768d, ADR-pinned; do not change                                        | CLAUDE.md retrieval pins                                                                           |
| Quantization           | halfvec/f16 safe; **binary quantization banned** (>40% nDCG@10 loss)   | CLAUDE.md retrieval pins                                                                           |

## Serving-path anatomy (`src/scix/search.py`)

`vector_search()` (line 619) checks `_qdrant_dense_gated(model_name)` first.
Gated → `_vector_search_qdrant()` (line 527). Not gated → the legacy pgvector
path, which **queries the dropped `paper_embeddings` table and cannot work**
(see "Rollback reality" below).

What `_vector_search_qdrant` does, in order:

1. Builds SQL filter clauses from `SearchFilters` (year, doctype, entity
   filters, etc.). Filters are **not** pushed into Qdrant — the serving
   collection's payload carries only `bibcode`, so there is nothing to filter
   on in-engine.
2. **Over-fetch:** `fetch_n = min(limit * 10, 500)` when any filter is
   present, else `fetch_n = limit` (line 550). The 10x is a heuristic; a very
   selective filter (e.g. one bibstem + one year) can still exhaust the 500
   cap and return fewer than `limit` results. That is expected behavior, not
   a bug — check `metadata["fetch_n"]` in the `SearchResult`.
3. Search params: `hnsw_ef = max(int(ef_search), limit)` normally;
   `SCIX_QDRANT_EXACT=1` forces `SearchParams(exact=True)` — full-scan exact
   kNN, the eval-only control for isolating HNSW approximation loss
   (canary gate G1 used it; Δ = +0.0000 vs HNSW). Far too slow for serving;
   never set it in the MCP env.
4. `client.query_points(collection, query=embedding, limit=fetch_n,
timeout=120)`. Points missing a `bibcode` payload are skipped with a
   warning, not crashed on.
5. **Postgres post-filter join:** `SELECT ... FROM papers p WHERE p.bibcode =
ANY(%s)` plus the filter clauses, then results are re-assembled in Qdrant
   rank order, truncated to `limit`. Scores are cosine similarity, same
   semantics as pgvector's `1 - (vec <=> query)`.
6. Returns `metadata={"backend": "qdrant_dense", "fetch_n": ...}` and
   `timing_ms={"vector_ms": ..., "qdrant_ms": ...}` — use these to confirm
   which backend actually served a query.

The client is a module-global singleton (`_qdrant_dense_client`); a changed
`QDRANT_URL` needs a process restart to take effect.

**Why REST-only:** qdrant-client's gRPC path fails deserializing Qdrant
1.18.2 _responses_ (verified 2026-06-11, bead `pkcd`; noted in the canary
doc). Reads/queries pin REST. Bulk _upserts_ via gRPC did work
(`scripts/qdrant_full_load.py` uses `prefer_grpc=True`, gRPC port 6634) —
the break is response deserialization, not the write path. Do not "fix" the
serving client to gRPC without re-verifying against the running server
version; ADR-013 mandates a pinned client + contract test for exactly this.

## Two collection lineages, two point-id schemes (trap)

There are TWO Qdrant lineages in this codebase. Mixing their id schemes
creates orphan points that can never be matched or updated.

|             | v2 serving lineage                                                                                                                                                         | v1 pilot lineage                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Collections | `scix_indus_v2_papers_s1`                                                                                                                                                  | `scix_papers_v1` (400K pilot), `scix_chunks_v1`                                                         |
| Code        | `src/scix/search.py`, `src/scix/qdrant_dense.py` (uncommitted), `scripts/qdrant_full_load.py`, `scripts/qdrant_outbox_sync.py`, `scripts/backfill_qdrant_filter_fields.py` | `src/scix/qdrant_tools.py`, `scripts/qdrant_upsert_pilot.py`, `scripts/backfill_qdrant_is_retracted.py` |
| Point id    | `str(uuid5(NAMESPACE_URL, bibcode))` — string UUID                                                                                                                         | `blake2b(bibcode, digest_size=8)` unpacked to int, `>> 1` (`qdrant_tools.bibcode_to_point_id`)          |
| Payload     | `bibcode` only                                                                                                                                                             | full ADR-008 schema (7 indexed + 5 metadata fields)                                                     |
| Vector      | single unnamed vector                                                                                                                                                      | named vector `indus` (`using="indus"` required in queries)                                              |
| Serves      | the production dense lane                                                                                                                                                  | MCP discovery/recommendation + chunk tools                                                              |

`qdrant_tools.py`'s docstring ("Qdrant holds a pilot subset") predates
ADR-013 and describes only its own v1 lineage — do not read it as a statement
about the dense lane. ADR-008's rich payload schema is the _contract for the
pilot lineage and for any future enrichment of v2_; the v2 serving collection
does not carry it today.

Whether the v1 pilot collections live in the same server instance as v2
(port 6633) is **unverified** in this pass (checking requires querying the
server). Operator check: `curl -s http://127.0.0.1:6633/collections` — a
live-service call; do not run casually from an agent session.

## Why Qdrant, not pgvector/pgvectorscale (ADR-013)

Full text: `docs/ADR/013_dense_lane_qdrant.md`. The short version a newcomer
must internalize before proposing "just use pgvector, it's already there":

The corpus (32.4M × 768d) sits ~6x past pgvector HNSW's documented comfort
zone. The planned fix — pgvectorscale StreamingDiskANN — failed
catastrophically on 2026-06-11 (bead `12rp`, closed):

1. `paper_embeddings.embedding` was a **dimensionless** `vector` column
   (multi-model table). DiskANN needs a fixed dim, so the rebuild DDL cast
   `(embedding)::vector(768)` — a true **expression index that pgvectorscale
   cannot scan** (`assertion failed: attnum > 0`, reproducible at 50k rows in
   seconds).
2. The benchmark validated `halfvec_cosine_ops` on a properly-typed pilot
   table; prod built `vector_cosine_ops` on a different storage type. The
   go/no-go numbers described a config production never ran.
3. Nobody ran a scratch-scale query before committing to the build. The 56 h
   index built `valid=t` and crashed every scan — and the planner
   auto-selected it, so it would have _broken_ all dense queries, not merely
   failed to help.
4. The serving HNSW index had been dropped before the replacement was
   validated → **~2 weeks of lexical-only serving** (measured cost:
   `results/litreview_rerun_2026-06-11.md`).

Qdrant was evaluated and cut over the same day (bead `pkcd`): CPU-built HNSW
(GPU/Vulkan builds produce ~0.5–0.75% disconnected nodes on this stack —
measured A/B, `scripts/qdrant_gpu_ab.py`, GPU rejected), 3.2 h full rebuild
from PG vs 56 h for DiskANN. Canary
(`docs/eval/qdrant_canary_v1_2026-06-11.md`): quality Δ = +0.0000 on
nDCG@10/MRR/recall@50 vs exact; warm p50 150 ms (vs 362 ms pgvector-era);
10-thread p95 variance FAILED (4.1x vs 2x bar) — the known lever is
`always_ram=true` for the SQ-INT8 layer (~25 GB RAM), held pending the
storage/RAM window (`results/qdrant_ram_relief_options.md`, bead `uy40`).

**The four validation rules this failure bought** (ADR-013 §Validation rules,
binding for ALL future index/lane work, mirrored in CLAUDE.md):

1. No index is trusted until one query has returned from it — ≤50k-row
   scratch build + forced-index-scan smoke test before any multi-hour build.
2. Benchmark DDL must be byte-identical to production DDL, or the benchmark
   is fiction.
3. Never drop a serving index before its replacement is validated scannable.
4. Re-examine architecture pins when they start selecting immature
   components.

Returning dense ANN to Postgres is not forbidden forever (ADR-013 names the
conditions: pgvectorscale matures AND a fixed-dim column migration), but it
is an ADR-level change routed through `scix-change-control` — never a
refactor.

## Rollback reality (changed since ADR-013 — read this)

ADR-013's rollback contract was: `paper_embeddings` in Postgres stays the
source of truth through a 30-day soak; rollback = unset `QDRANT_URL`, restart
the MCP, serve from pgvector again.

**That contract is dead as of ~2026-06-29/30.** `paper_embeddings` (and
`embedding_outbox`) were dropped in prod out-of-process — outside the ADR-015
staging, before the 2026-07-11 soak gate, with no NAS archive (bead `s7cy`,
OPEN, P1; verified in-bead 2026-07-03: `to_regclass('public.paper_embeddings')
= false`, no `/mnt/scix_offload/paper_embeddings*`). Consequences for THIS
skill's scope:

- Unsetting `QDRANT_URL` no longer degrades to pgvector — it routes
  `vector_search()` to a code path that raises
  `psycopg.errors.UndefinedTable`. **Qdrant is now the only copy of the
  32.4M INDUS vectors on this host.** Treat the collection and its NAS
  snapshots as production data, not a derived cache.
- The migration-070 outbox freshness path (`scripts/qdrant_outbox_sync.py`,
  `daily_sync.sh` step 7, bead `8m0a`) is dead — its trigger source table is
  gone. New papers since ~2026-06-30 (~1–3k/day, ~83K at last bead count)
  have no dense vector; the dense lane is silently stale for recent content.
- A committed docstring (`scripts/qdrant_reload_with_payload.py`, committed
  2026-06-15) says the table was "DROPPED 2026-06-14". That date is
  inconsistent with ADR-015 (2026-06-22, treats the table as present) and
  with bead `s7cy`'s verified timeline (last clean daily_sync 2026-06-29).
  Trust the bead + ADR timeline; treat the script docstring's date as suspect.

**PROVISIONAL pending Stephanie (discovery Q2):** a direct-to-Qdrant
remediation exists but is **uncommitted in the working tree** —
`src/scix/qdrant_dense.py`, `migrations/072_indus_qdrant_synced.sql` (a
bibcode+timestamp watermark table replacing the 195 GB staging table),
`scripts/seed_indus_qdrant_synced.py`, and a rewritten `src/scix/embed.py`.
It is in-flight, not canon. Do not build on it, do not commit it, and do not
teach it as the standard path until it lands. Its `upsert_dense()` docstring
itself flags that it re-introduces the bibcode-only-payload wipe hazard
(next section) if the collection is ever enriched. Full pipeline story:
`scix-embedding-pipeline`.

## Payload history: the wipe hazard and the write-path pathology

Two hard-won facts govern any payload work on the serving collection:

1. **A full-point upsert REPLACES the payload** (bead `e4xv`, closed). An
   upsert with `payload={"bibcode": ...}` wipes any richer payload a point
   carried. Today that is a no-op (every v2 point carries only `bibcode`),
   but the moment the ADR-008 enrichment backfill runs, every re-embed/
   recovery upsert becomes a silent payload strip. The e4xv fix (full
   ADR-008 payload in the outbox drain, shared contract module
   `src/scix/qdrant_payload.py`) was committed on branch
   `bd/e4xv-outbox-payload-fix` (commit `134215b`) and is **NOT on this
   branch's HEAD** — `git ls-files` shows no `qdrant_payload.py`. If you
   enrich the collection, land payload-preserving writes first.
2. **Per-point `set_payload` on the serving collection costs ~47–63 s/op to
   apply** — intrinsic to its ~1.3M-point on-disk segments, proven by WAL
   recovery replay (`results/qdrant_writepath/root_cause_and_fix.md`, bead
   `tqg4`). Enrichment therefore does NOT patch points in place; the shipped
   pattern is `scripts/qdrant_reload_with_payload.py` — scroll vectors out of
   the existing collection read-only, join payload from Postgres by bibcode,
   bulk-upsert into a NEW collection, build payload indexes AFTER the load.
3. The full 32.4M enrichment backfill is parked as bead `2xi8` (frozen,
   operator-gated, scix-batch only). Until it runs, in-engine filtering is
   impossible and the over-fetch + PG post-filter pattern above is
   load-bearing — do not remove it.

## ADR-014: the Qdrant sparse BM25 lane — parked, not dead

**PROVISIONAL pending Stephanie (discovery Q5):** treat as
parked-for-relanding; check bead + branch state before touching.

`docs/ADR/014_qdrant_sparse_lexical_lane.md` (status: Proposed). The pilot
(2026-06-11, 52k universe) PASSED its pre-registered system-level gate:
swapping the Postgres tsvector lexical lane for Qdrant FastEmbed BM25
improved the fused hybrid by +0.06–0.10 nDCG@10 (big win on
`author_specific`; small real loss ~−0.04 on `title_matchable` scientific
tokens). Not Accepted because (a) an AND-vs-OR query-parsing confound
inflates the magnitude and (b) it needs confirmation at 32M scale.

Phase 2 harness is **built and committed**; the 32M build is **held on NVMe
headroom** (`results/adr014_phase2_blocked_on_disk.md`: ~40–46 GB peak needed,
gate is `free ≥ estimate + 20 GB`). It unblocks when the storage line (next
section) frees disk. Runnable-today pieces are listed in the ADR (streaming
build, `pg_or` attribution arm, tokenizer tuning via
`scripts/_sparse_bm25.py`). If accepted, it forces a **v2 collection rebuild
as named vectors** — the current single-unnamed-vector collection cannot host
a named sparse vector alongside. Do not start that build without the ADR
flipping to Accepted and change-control sign-off.

## The ADR-015/016 storage line (as it bears on this lane)

Depth lives in `scix-index-and-storage-discipline`; what a dense-lane
operator must know:

- **ADR-015** (Proposed) staged the retirement of `paper_embeddings`' INDUS
  footprint: Stage 1 = drop the two INDUS HNSW indexes
  (`migrations/071_drop_paper_embeddings_indus_indexes.sql`, ~120 GB+ back to
  the OS at commit), soak-gated to **2026-07-11**; Stage 2 = row reclaim,
  deliberately not pre-authored. The out-of-process `DROP TABLE` (s7cy)
  bypassed this staging entirely — bead `s7cy` also records
  `schema_migrations` maxing at 68 with 069–072 applied by hand, unrecorded
  (there is no auto-migration runner).
- **ADR-016** (Accepted 2026-06-29) seals `papers_fulltext` ≤2024 to a NAS
  cold tier, reclaiming ~470–493 GB. That reclaim is what unblocks the
  dense-lane RAM relief (`always_ram` for SQ-INT8, ~25 GB — fixes the failed
  canary gate G3) and the ADR-014 Phase-2 build. Both are parked on ONE
  operator maintenance window; the weekly audit's `BLOCKED_CHECK` is RED
  (blocked on human). Do not attempt to "helpfully" run any of it.
- **Snapshots:** `scripts/qdrant_snapshot_to_nas.sh` (default endpoint
  `:6633`) — the container's `/qdrant/snapshots` must be bind-mounted to a
  NAS path so the ~78 GB tar never stages on NVMe (the API
  download-then-move pattern always materializes the tar locally first;
  that mode is deliberately unsupported). Live Qdrant storage stays on local
  NVMe, never NAS (NFS unsafe for live writes). Heavy; operator-run under
  `scix-batch`; do not run casually.

## Operating the container

- **Binding policy (security, incident-backed):** REST and gRPC ports bind
  `127.0.0.1` ONLY. Open-source Qdrant has no auth unless
  `QDRANT__SERVICE__API_KEY` is set; on 2026-04-25 the container was found
  LAN-reachable because a bare `docker run -p 6333:6333` defaults HostIp to
  `0.0.0.0` (bead `s1a`). `scripts/preflight_qdrant_security.sh
[container-name]` verifies bindings read-only via `docker inspect` — run it
  after every container start.
- **Port drift warning:** the committed `deploy/qdrant-compose.yml` and
  `docs/runbooks/qdrant.md` describe the ORIGINAL pilot container
  (`scix-qdrant`, ports 6333/6334, `.qdrant_storage/`) and predate ADR-013
  (compose last touched 2026-04-26). The live serving instance for this
  installation is on **6633 (REST) / 6634 (gRPC)** with storage in
  `.qdrant_v2_storage/` (`.qdrant_gpu_storage/` is the rejected GPU-build
  experiment). Every current config path — `.mcp.json`, the canary doc,
  script defaults — says 6633. How the 6633 container is launched is **not
  captured in any committed file found this pass** (unverified); recover it
  with `docker inspect` on the running container before any restart, and
  treat writing a v2 compose file as an open, bead-worthy gap.
- ADR-013 records the serving container as `docker --memory 40g`, NVMe
  storage, 127.0.0.1-only.
- Restart discipline: this host co-runs production services — container
  stop/start is an operator action, not an agent action. Rollback to
  pgvector does not exist (see Rollback reality); a dead Qdrant means a dead
  dense lane until the container returns.

## Doc-drift table (do not trust these on this topic)

| Artifact                                               | Stale claim                                                         | Reality                                                                                   |
| ------------------------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `README.md` / CLAUDE.md header                         | "Single PostgreSQL instance / no separate search engine"            | Qdrant serves the dense lane (ADR-013)                                                    |
| `.claude/skills/scix-mcp`                              | OpenAI embedding lane; trycloudflare/bearer connection              | OpenAI lane removed; public deploy decommissioned 2026-06-12 (intentional, not an outage) |
| `src/scix/qdrant_tools.py` docstring                   | "Postgres+pgvector is source of truth; Qdrant holds a pilot subset" | True only of the v1 pilot lineage; the v2 serving collection is now the only vector copy  |
| `deploy/qdrant-compose.yml`, `docs/runbooks/qdrant.md` | ports 6333/6334, `.qdrant_storage/`                                 | live serving instance is 6633/6634, `.qdrant_v2_storage/`                                 |
| `scripts/qdrant_reload_with_payload.py` docstring      | "paper_embeddings DROPPED 2026-06-14"                               | drop verified ~2026-06-29/30 (bead s7cy); date suspect                                    |
| `vector_search()` docstring / pgvector branch          | implies a working pgvector path                                     | targets the dropped table; only the Qdrant branch works                                   |

Fix stale docs only under an explicit bead (PROVISIONAL pending Stephanie,
discovery Q5) — never silently.

## Provenance and maintenance

Verified 2026-07-07 by source-reading only (no DB/Qdrant connections, no
scripts executed) at commit `452ab86`, branch
`bd/0yp5-external-copy-accuracy-audit`. Runtime facts (point counts, live
port bindings, collection inventory) are quoted from the canary doc and bead
records, not re-measured.

Re-verify before trusting:

```bash
git rev-parse --short HEAD && git branch --show-current   # pin drift
grep -n "_QDRANT_DENSE_COLLECTIONS\|SCIX_QDRANT_EXACT\|fetch_n = min" src/scix/search.py  # gate, exact flag, over-fetch
grep -n "prefer_grpc" src/scix/search.py scripts/qdrant_outbox_sync.py    # REST pin
git ls-files src/scix/ | grep qdrant                       # qdrant_dense.py committed yet? (s7cy fix landing)
git log --oneline -3 -- migrations/072_indus_qdrant_synced.sql docs/ADR/013_dense_lane_qdrant.md
bd show s7cy | head -5                                     # is the live fire still open?
bd show 2xi8 | head -3                                     # payload enrichment still parked?
grep -n "Status" docs/ADR/014_qdrant_sparse_lexical_lane.md docs/ADR/015_offload_drop_paper_embeddings_indus.md | head
```
