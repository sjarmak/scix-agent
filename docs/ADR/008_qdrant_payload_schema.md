# ADR-008: Canonical Qdrant payload schema for `scix_papers_v1`

- **Status**: Accepted (2026-04-26)
- **Deciders**: SciX Agent maintainers
- **Scope**: Every script and library code path that writes to or reads from the Qdrant collection `scix_papers_v1` (and its successors)
- **Supersedes**: The first naming pass in `docs/prd/qdrant_nas_migration.md` MH-2 ("year, primary_class, community_id_semantic, is_retracted") — superseded 2026-04-25 by reconciliation with the live pilot
- **Related**: PRD `docs/prd/qdrant_nas_migration.md` (gitignored), bead `scix_experiments-rdx`, memory `community_labels_pipeline.md`

## Context

The `scix_papers_v1` pilot collection (400K INDUS rows, m=16, no quant) was populated by `scripts/qdrant_upsert_pilot.py` using payload field names that match the live Postgres column names (`arxiv_class`, `community_semantic_medium`, etc.). The first PRD draft for the Qdrant NAS migration used a different naming scheme (`primary_class`, `community_id_semantic`, etc.).

Two naming conventions in flight is one too many. A second person writing an upserter will pick whichever name they grep first and create a third dialect. We pick the live pilot's names as canonical because:

1. They match the Postgres source columns, so the upsert SQL is column-for-column readable.
2. They are already populated on 400K live points; renaming would invalidate every payload index and break in-flight filtered queries.
3. The PRD has been edited to match (MH-2 in `qdrant_nas_migration.md`).

`is_retracted` is the one field added that does **not** mirror a Postgres column directly — it is derived from `papers.retracted_at IS NOT NULL` (per migration 058's denormalized convenience column) so agents can filter retracted papers out of retrieval without joining back to Postgres.

## Decision

**The canonical payload schema for `scix_papers_v1` is exactly the seven indexed fields and five non-indexed metadata fields below. Any new upserter, backfill, or read path MUST use these names.** Any change to this schema requires updating this ADR first.

### Indexed payload fields (7)

| Field name                    | Qdrant type | Source SQL                                       | Why indexed                                                      |
|-------------------------------|-------------|--------------------------------------------------|------------------------------------------------------------------|
| `year`                        | `integer`   | `papers.year`                                    | Recency / time-window filtering, the most common agent filter.   |
| `doctype`                     | `keyword`   | `papers.doctype`                                 | Article-vs-thesis-vs-erratum filtering.                          |
| `arxiv_class`                 | `keyword`   | `papers.arxiv_class` (text array)                | Discipline filtering. Stored as a list; Qdrant matches any-of.   |
| `bibstem`                     | `keyword`   | `papers.bibstem` (text array)                    | Venue-tier filtering (e.g. `ApJ` vs `arXiv`).                    |
| `community_semantic_coarse`   | `integer`   | `paper_metrics.community_semantic_coarse`        | Hierarchical-community filtering (broad partition).              |
| `community_semantic_medium`   | `integer`   | `paper_metrics.community_semantic_medium`        | Hierarchical-community filtering (working partition per memory). |
| `is_retracted`                | `bool`      | `(papers.retracted_at IS NOT NULL)`              | Retraction-aware retrieval — see absence semantics below.        |

### Non-indexed metadata (carried in payload, never queried by index)

These accompany every point so a single Qdrant search can return enough context for an MCP response without a Postgres round-trip:

| Field name        | Source SQL                          | Notes                              |
|-------------------|-------------------------------------|------------------------------------|
| `bibcode`         | `papers.bibcode`                    | Primary key. Always present.       |
| `title`           | `papers.title`                      | For result rendering.              |
| `first_author`    | `papers.first_author`               | First-author shorthand.            |
| `citation_count`  | `papers.citation_count`             | For ranking heuristics.            |
| `pagerank`        | `paper_metrics.pagerank`            | For ranking heuristics.            |

### Citation-Leiden community fields (intentionally absent)

`community_id_coarse` / `community_id_medium` (the citation-Leiden partition columns) are **not** in the payload schema. Per memory `citation_leiden_phase_b_incomplete.md`, those columns hold sentinel `-1` (12.4M rows) plus NULL (20M rows) on prod — the citation partition has never completed Phase B. Including them would surface garbage to filtered queries. Re-evaluate when Phase B lands.

### Absence semantics for `is_retracted`

A point's `is_retracted` value reflects the retraction state at the time of upsert. New retractions (rows that pick up a non-NULL `retracted_at` after upsert) are propagated by a backfill — `scripts/backfill_qdrant_is_retracted.py` — not by realtime CDC. **Filter writers should use `must_not(match_value=true)` to exclude retracted papers**; this excludes `is_retracted=true` and tolerates points where the field is absent (treated as "not known retracted").

## Consequences

- Every existing reader/writer that touched the legacy names (`primary_class`, `community_id_semantic`) must be retargeted; the only known instance was the PRD itself, fixed 2026-04-25.
- Adding a new indexed field is non-trivial: it requires an ADR amendment, an `upsert_pilot.py` change, and a backfill script for the existing 400K rows. Don't add fields casually.
- The 7-index ceiling is intentional — every payload index is RAM and disk overhead. Fields that don't get filtered against in production agent traces should stay in unindexed metadata.

## Backfill discipline

Any one-time field addition (the `is_retracted` rollout is the worked example) MUST:

1. Land an `idx_<field>` payload index on the live collection via `client.create_payload_index` — idempotent, safe to re-run.
2. Run a backfill script that sets the field on existing points using `client.set_payload` keyed by `bibcode_to_point_id(bibcode)` — same hash function as the upserter.
3. Be re-runnable without duplicating writes.
4. Update the upserter so future inserts already carry the field.

`scripts/backfill_qdrant_is_retracted.py` is the reference implementation.
