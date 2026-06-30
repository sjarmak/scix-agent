# ADR-016: Time-Partitioned Cold-Text Tier on NAS

- **Status**: Accepted (2026-06-29); **amended 2026-06-29 post-sizing** (see Amendment A1)
- **Deciders**: SciX maintainers (Stephanie)
- **Scope**: `papers.body`, `papers_fulltext` JSONB columns, `src/scix/search.py`
  (body BM25 query + `read_paper`/`read_paper_section`), a new annual seal job,
  a DS read-through cache, ops cadence.
- **Supersedes**: none
- **Extends / relates**: storage tiering policy (CLAUDE.md), `docs/prd/qdrant_nas_migration.md`
  (this ADR covers the *text* corpus, the PRD covers *vectors*), ADR-010
  (`sections_tsv` stored expression index), ADR-011 (drop `papers.raw`, still
  Proposed — complementary in-place reclaim, not part of this ADR), ADR-015
  (offload `paper_embeddings` to Qdrant).

## Amendment A1 (2026-06-29, post-sizing)

The read-only sizing pass corrected a wrong assumption in the original Decision
(that `sections_tsv` is small and cheap to keep hot). Measured reality:

- `papers_fulltext` is **493 GB = 4.5 GB heap + 29 GB indexes + 460 GB TOAST**.
- The TOAST holds **`sections` ≈ 214 GB AND `sections_tsv` ≈ 217 GB** —
  co-located in the same TOAST relation.
- `sections_tsv` backs only the **`section_retrieval`** lane
  (`mcp_handlers/sections.py`), which is **not one of the 15 exposed MCP tools**
  and whose GIN (`idx_papers_fulltext_sections_tsv`, 27 GB) has **`idx_scan=0`**
  (`pg_stat_database.stats_reset` is NULL and peer indexes show millions of
  scans, so the zero is real, not a reset artifact). `sections_tsv` is
  **rederivable** from the section text via `to_tsvector`.
- Row split: **sealed (≤2024) = 14,599,062; hot (≥2025) = 342,425 (2.3%)**.
- No FK references `papers_fulltext`; one view (`v_papers_fulltext_route_inputs`)
  depends on it.

**Consequence — keeping `sections_tsv` is what makes reclaim infeasible.** Because
`sections` and `sections_tsv` are co-TOASTed, reclaiming the 214 GB of `sections`
while keeping the 217 GB of `sections_tsv` requires rewriting the whole 460 GB
TOAST (`VACUUM FULL`/`pg_repack` scratch ≈ live size ≈ 220 GB) — impossible at
25 GB free.

**Amended decision.** For **sealed years**: seal the section *text* to NAS (as
before) **and drop `sections_tsv`** — defer the unexposed `section_retrieval`
lane; it can be rebuilt from the NAS shards (or a cold lexical index) if/when it
is exposed. **Hot years keep** `sections` + `sections_tsv` (342 K rows, ~10 GB —
cheap, preserves the forward option).

**Amended reclaim mechanism (replaces "partitioned rebuild" framing).** Because
hot is only 2.3% of rows, rebuild as a small fresh table — no partitioning
machinery, no 220 GB scratch:

1. Full-scale **seal builds** of all sealed years to NAS (read-only on PG),
   verified by row-count + per-bibcode checksum parity.
2. `CREATE TABLE papers_fulltext_hot AS SELECT … WHERE year ≥ 2025` (~10 GB,
   fits in 25 GB free); add PK + (hot-only) `sections_tsv` GIN.
3. Ship the cold read route (`read_paper_section` / section consumers: sealed
   bibcode → NAS shard) **before** any drop.
4. Metadata swap in one txn: drop `v_papers_fulltext_route_inputs`, rename old →
   `papers_fulltext_old`, rename hot → `papers_fulltext`, recreate view + index.
5. Verify reads (hot via PG, sealed via NAS), then **`DROP TABLE
   papers_fulltext_old`** → reclaims ~470 GB instantly (whole-relation drop, zero
   scratch).

This also removes the **Phase 2 / Phase 3 split's** rationale for `papers_fulltext`
(Phase 2's stored-`body_tsv` work still applies to `papers.body` in Phase 3).
Phase 1 now ends with the monolith dropped and ~470 GB reclaimed.

## Amendment A2 (2026-06-30, execution-time)

Two things surfaced while executing Phase 1b; both are handled.

- **Omitted shard columns are immaterial.** The seal preserves `bibcode`,
  `sections`, `figures`, `tables`, `equations`, `source`, `parser_version`,
  `canonical_bibcode`, `suppressed_by_publisher`. It omits `inline_cites`,
  `source_version`, `arxiv_version` (verified **0 rows with data** across all
  14,599,062 sealed rows) and `parsed_at` (a write-only parse timestamp, read by
  no serving path). Net: **zero scientific-content loss**; the dropped columns
  are intentional.
- **Routing view + populator.** `v_papers_fulltext_route_inputs.has_fulltext_row`
  is a LEFT JOIN to `papers_fulltext`; after the drop it reads `false` for sealed
  papers. Its only consumer is the **manual** `populate_papers_fulltext.py`
  backfill (no cron path). To stop it from treating 14.6M sealed papers as
  fulltext gaps (re-fetch/re-parse), `iter_candidate_papers` now floors on
  `p.year >= HOT_WINDOW_START_YEAR` (`scix.coldtext.HOT_WINDOW_START_YEAR`). The
  view is recreated as-is. Trade-off: a late-arriving pre-2025 paper won't get
  body-parsed by the populator — acceptable (sealed years are closed).

**Reclaim shape (disk safety).** The box is at 98% with WAL on the same volume
and a disk-full/PG-OOM history, so 1b uses the **hot-only rebuild** (Design A):
`CREATE papers_fulltext_hot` + INSERT 342,425 hot rows + indexes + FK (small
WAL), one-txn rename swap, then `DROP papers_fulltext_old` (instant). The
alternative — keeping 14.6M slim sealed stubs to preserve the view natively —
was rejected as a 14.6M-row rewrite (heavy WAL on a near-full disk).

## Amendment A3 (2026-06-30, post-Phase-1): body phases (2 & 3) superseded — won't do

Phase 1 reclaimed ~493 GB; DS is at 72% (532 GB free), so the disk crisis that
motivated the body phases is resolved. Measurement then showed Phase 2/3 is
**net-negative for `papers.body`** and they are cancelled.

Sampled (papers, 411 GB): `body` (raw) ~208 GB; a **materialized `body_tsv`
~236 GB**; `raw` JSONB ~11 GB; `tsv` (title/abstract) ~30 GB.

Body-BM25 is **actively used** (`ix_papers_body_tsv` idx_scan=1258 — opposite of
the dead `sections_tsv` we dropped in Phase 1), so the lexical lane cannot be
dropped. Today it serves from an **expression GIN (39 GB)** that recomputes
`to_tsvector(body)` at query time — there is no stored tsvector. Dropping raw
`body` (Phase 3) requires a **stored** `body_tsv` (Phase 2), and a tsvector
*with positions* is **larger than the prose it indexes**:

- Now: `body` 208 + GIN 39 = **247 GB** in PG.
- After 2+3: `body_tsv` 236 + GIN ~39 = **275 GB** in PG (**+28 GB**), plus a 32M-row
  online rewrite, heavy WAL, and hours of ACCESS-EXCLUSIVE/maintenance risk — to
  reclaim nothing.

Decision: **do not implement Phase 2 or Phase 3.** Body stays in Postgres;
body-BM25 keeps its expression GIN; `read_paper` full-body keeps reading
`papers.body` from PG. Rejected variants: positionless `body_tsv` (~80 GB net,
but changes ranking and needs an eval — not worth it with no disk pressure);
`raw`-column drop (ADR-011, only ~11 GB).

Phase 4 (operationalize the annual `papers_fulltext` seal + read-through cache +
monitoring) remains valid for the cold tier built in Phase 1.

## Context

DS (`/dev/nvme1n1p2`, 1.9 TB NVMe — the live serving disk) is at **99%, 25 GB
free**. NAS (`/mnt`, NFS) has **48 TB free**. The scix DB is **1198 GB**, and the
bulk is full-text *display content*:

| Object | Size | Content |
| --- | --- | --- |
| `papers_fulltext` | 493 GB | parsed `sections`/`figures`/`tables`/`equations` JSONB + `sections_tsv` (14.9M rows) |
| `papers.body` (subset of 411 GB table) | ~210 GB | raw body text (32.4M rows; ~14.9M with body) |

This content is **read-mostly** and only needed at `read_paper` time. Everything
that powers *search* — metadata, the `tsv`/`sections_tsv` tsvectors and their GIN
indexes, the citation graph (`citation_edges`, 299M edges), the entity graph, and
the full INDUS dense collection in Qdrant (83 GB on DS) — is small and
latency-critical.

The raw ADS JSONL source is already mirrored to `/mnt/scix_offload`, and
`weekly_pg_backup.sh` already treats `papers`/`papers_fulltext` as re-harvestable.
So the 1.2 TB is a *serving copy*, not the only copy.

**Constraint**: NFS is unsafe for live *write* workloads (storage tiering policy)
— but safe for **read-only** files. The naive "mount the PG/Qdrant data dir on
NAS" is therefore off the table; a read-only, write-once cold tier is not.

Bodies span ~1991–2026. `daily_sync.sh` only ever appends the **current year**.

## Decision

Adopt a **time-partitioned, text-only cold tier**:

1. **Hot window = last 2 years (2025–2026).** Their raw `body` and
   `papers_fulltext` JSONB stay in PG on DS, writable by `daily_sync`.
2. **Sealed years = 2024 and older.** Their `body` raw text and `papers_fulltext`
   JSONB are moved to **read-only, per-year shards on NAS** (`/mnt/scix_coldtext/v1/{year}/`,
   columnar — DuckDB or Parquet — keyed by `bibcode`), then removed from PG.
3. **Everything that powers search stays hot on DS for every year**: metadata,
   `tsv`, `sections_tsv`, a new stored `body_tsv` (see Prerequisite), all GIN
   indexes, citation graph, entity graph, and the full Qdrant dense collection.
4. **`read_paper` / `read_paper_section` route by year**: PG for hot years, NAS
   shard for sealed years, fronted by a small **DS read-through LRU cache**.
5. **Sealing is an annual idempotent batch.** A year is sealed once `daily_sync`
   has stopped appending to it (after a grace period).

### Why this shape

- **Search never touches NAS** — all tsvectors/indexes/dense vectors remain on DS,
  so query latency is unchanged and there is **no cross-tier RRF fanout**.
- **No iSCSI dependency** — sealed shards are write-once/read-only, which NFS
  handles safely. (Cold *vectors* would need an iSCSI block LUN; we explicitly
  decline that — Qdrant is only 83 GB and fits on DS.)
- **`daily_sync` is unaffected** — it appends the current year, which is hot.
- **Bounds DS growth** to the hot window going forward.

## Prerequisite schema change (gates Phase 3)

Body BM25 currently uses an **expression** GIN, `to_tsvector('english', body)`,
and the query **recomputes** `to_tsvector(p.body)` at runtime (`search.py:495`).
Both depend on raw `body` being materialized in PG. To let raw body leave PG for
sealed years while preserving body BM25 for **all** years, we must first
**materialize a stored `body_tsv` column** (covering all years), rebuild the GIN
on it, and switch the query to `body_tsv @@ ...`.

`papers_fulltext.sections_tsv` is already a stored, independent column (ADR-010),
so its JSONB can be sealed with **no** schema change — which is why it goes first.

## Non-goals

- Moving the live PG cluster or Qdrant onto NAS (NFS or otherwise).
- A cold dense-vector tier / iSCSI LUN (Qdrant is 83 GB; stays hot on DS).
- Any change to retrieval ranking or the RRF fusion.
- Off-host backup (ADS upstream is the DR tier; storage tiering policy).

## Migration phases

Ordered so each phase is independently shippable and Phase 1 creates the DS
headroom the later schema work needs (the box has only 25 GB free).

- **Phase 1 — Seal `papers_fulltext` JSONB for years ≤2024** (no schema change).
  Stream (COPY) to NAS year-shards, verify row-count + per-bibcode checksum
  parity vs PG, route `read_paper_section` cold→NAS, null the JSONB for sealed
  years, `VACUUM` per-year. Biggest immediate win; minimal DS scratch (streams to
  NAS).
- **Phase 2 — Materialize stored `body_tsv`** on `papers` (with Phase-1
  headroom), rebuild the body-BM25 GIN on it, switch `search.py`, verify BM25
  parity on the gold query set.
- **Phase 3 — Seal `papers.body` raw text for years ≤2024** to NAS shards, route
  `read_paper` cold→NAS, null `body` for sealed years, `VACUUM`.
- **Phase 4 — Operationalize**: idempotent annual seal job + DS read-through LRU
  cache + monitoring + tests; wire the seal into the ops cadence.

## Validation gates (mirroring ADR-013 discipline)

- No PG deletion until the shard has returned **one** read for a sampled bibcode.
- Shard build verified by **row-count + per-bibcode checksum** parity vs PG before
  any PG `body`/JSONB nulling.
- `read_paper` content parity test: sealed-year fetch byte-equals pre-seal content
  on a sample.
- BM25 parity (Phase 2): stored-column query returns the same top-k as the
  expression query on the gold set.
- All heavy steps run under `scix-batch` (cgroup memory bounds) with
  `max_parallel_workers_per_gather=0` and bounded `work_mem` (PG-side OOM rules).

## Consequences

- **+** Frees ~600–700 GB off DS progressively; bounds future DS growth.
- **+** Search path and latency unchanged; no iSCSI; cold shards immutable +
  checksummed.
- **−** `read_paper` on sealed years pays an NFS read (mitigated by DS LRU cache;
  human/agent-paced, not a hot loop).
- **−** Adds a stored `body_tsv` column (~30–60 GB) — a fraction of the ~210 GB
  raw body reclaimed.
- **−** New annual seal job + read-router code path + cache to maintain.
