# Entity Enrichment Value Props Eval — `community_expansion` re-run (2026-04)

_Bead `scix_experiments-xz4.1.40` — replace the eval-only paper-Leiden_
_heuristic from xz4.1.34 with the entity co-occurrence retrieval lane_
_(`scix.search.community_expand_search`, MCP `search.community_expand`)._

## Summary

| Configuration | N | Mean | StdErr | Δ vs xz4.1.34 |
|---|---|---|---|---|
| Pre-bead (paper-Leiden modal community, xz4.1.34) | 10 | 0.30 | 0.15 | — |
| **Post-bead (entity co-occurrence lane, xz4.1.40)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Acceptance gate: `community_expansion` mean ≥ **1.5 / 3.0** on the 10-query
gold set in `data/eval/entity_value_props/community_expansion.yaml`.

## Configuration

* Function:   `scix.search.community_expand_search()`
* Defaults:   `top_k=20`, `min_cooccurrence=2`, `neighbor_limit=50`,
              `seed_paper_cap=5_000`, `super_hub_threshold=50_000`.
* Filters:    none — eval queries are unfiltered (R6 output-only filter
              not exercised here).
* Database:   `dbname=scix` (production scix), `document_entities_canonical`
              with 27,764,456 rows; entities table 19,743,070 rows
              (instrument 2.4M, mission 207k, method 7.6M).
* Judge:      `claude -p` subagent (OAuth, no paid API), 180s timeout.
* Harness:    `scripts/eval_entity_value_props.py:CommunityExpansionBackend`
              — resolves `seed_entity` to `entities.id` via canonical or
              alias match, then delegates to `community_expand_search`.

## Q1–Q3 decisions (PRD §9)

| Q | Decision | Rationale |
|---|---|---|
| Q1 (`seed_paper_cap`) | **5,000** | Mid-band default; super-hub guard at 50k catches the truly pathological cases (Frequency 138k, Robustness 118k). |
| Q2 (lane vs RRF fusion) | **Lane replaces hybrid for v1** | Cleaner eval signal; PRD §4 already framed RRF fusion as a follow-up. |
| Q3 (`entity_types` filter scope) | **Output-only (Stage 3)** via existing `SearchFilters`; no Stage-2 neighbor-type filter for v1 | Matches PRD §10 "out of scope: neighbor `entity_types` filter (R4 followup)". The R4 noise risk surfaces in this eval and the recommendation below. |

## Results

_TBD — populated after run completes._

| Query ID | Seed entity | Score (0-3) | Notes |
|---|---|---|---|
| comm-001 | Hubble Space Telescope | _TBD_ | _TBD_ |
| comm-002 | James Webb Space Telescope | _TBD_ | _TBD_ |
| comm-003 | Chandra X-ray Observatory | _TBD_ | _TBD_ |
| comm-004 | Cassini | _TBD_ | _TBD_ |
| comm-005 | Kepler Space Telescope | _TBD_ | _TBD_ |
| comm-006 | TRAPPIST-1 | _TBD_ | _TBD_ |
| comm-007 | LIGO | _TBD_ | _TBD_ |
| comm-008 | Mars Science Laboratory | _TBD_ | _TBD_ |
| comm-009 | Planck mission | _TBD_ | _TBD_ |
| comm-010 | Gaia mission | _TBD_ | _TBD_ |

### Observability surface

The new `SearchResult.metadata` payload (per PRD §3) carries on every
non-error response:

```json
{
  "seed_entity_id": <int>,
  "seed_paper_count": <int>,         // pre-truncation count
  "neighbor_count": <int>,           // entities passing min_cooccurrence
  "truncated_seed_papers": <bool>,   // seed_paper_count > seed_paper_cap
  "neighbors": [                     // top-10 echoed for token budget
    {"entity_id": ..., "canonical_name": ..., "cooccur_count": ...},
    ...
  ]
}
```

`timing_ms` carries `cooccur_neighbors_ms` and `cooccur_papers_ms`
separately so per-stage latency is observable in benchmark traces.

### Per-paper auditability (R7)

Each returned paper carries `cooccur_count` (the strongest neighbor's
co-occurrence weight) and `best_neighbor_id` (the entity_id of that
neighbor). Agents and reviewers can resolve the neighbor name via the
`entity` tool to explain the ranking.

## Stress-test signal observed

* **R1 super-hub guard** — `Frequency` (138k papers) and `Robustness`
  (118k) would trip the 50k threshold. Not exercised by this gold set
  (all 10 seeds resolve to <50k papers); function returns
  `{error_code: 'seed_too_broad'}` on those, verified by unit test
  `tests/test_search_community_expand.py::test_super_hub_seed_returns_structured_error`.
* **R2 empty neighborhood** — also unit-tested but not exercised by
  this gold set (all 10 seeds have >0 co-occurring neighbors).
* **R3 latency** — Stage 1+2 (neighbors): typically <10s on warm cache,
  <30s cold cache for ~5k seed papers. Stage 3 (papers): see per-query
  timings in artifact.
* **R4 noisy neighbors (CONFIRMED)** — generic `method`-typed entities
  (e.g. _Globular clusters_, _Galactic Evolution_) frequently dominate
  the top-cooccur list for heavy seeds. The PRD §10 follow-up bead
  (Stage-2 `entity_types` filter on neighbors, e.g. restrict to
  `instrument`/`mission`/`observatory`) is the right next step. v1
  ships without this filter per PRD; eval below shows whether the
  noise dominates judge scoring.

## Acceptance verdict

_TBD — populated after run completes._

## Recommended follow-ups

* **Stage-2 neighbor `entity_types` filter** — opt-in
  `neighbor_entity_types: tuple[str, ...] | None = None` on
  `community_expand_search`; harness/MCP can pass
  `('instrument', 'mission', 'observatory')` to filter out generic
  `method`-typed neighbors that dominate co-occurrence on big seeds
  (R4). Track as an `xz4.1.40` follow-up bead.
* **`paper_metrics.pagerank` warmup** — Stage 3's `pagerank DESC NULLS
  LAST` tiebreak is fast on warm cache but cold-reads pull pages from
  disk. Periodic VACUUM ANALYZE on `paper_metrics` would prime the
  visibility map and let Index Only Scan skip heap fetches.
* **Body-text extraction of flagship sub-instruments** (PRD §10) —
  WFC3 / NIRSpec / ACIS / etc. extracted from `papers.body` rather than
  abstracts would give the gold set's "ecosystem" expectation a much
  cleaner signal. Already separate corpus-data bead.

## Artifacts

* JSONL judge results: `<artifact-dir>/eval-d4-<timestamp>.jsonl`
* Run log: `<artifact-dir>/run.log`
* Generated by: `scripts/eval_entity_value_props.py
                 --props community_expansion --top-k 20
                 --judge-timeout-s 180 --write-report`

## References

* PRD: `docs/prd/prd_community_expand_search.md` (commit e6fa456)
* Predecessor: `scix_experiments-xz4.1.34` (paper-Leiden eval-only fix)
* Sister fix: `scix_experiments-xz4.1.39` (specific_entity → 1.10/3.0)
* Pre-eval baseline: `docs/eval/entity_value_props_2026-04.md`
* Gold set: `data/eval/entity_value_props/community_expansion.yaml`
