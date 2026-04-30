# PRD: `community_expand_search()` — entity co-occurrence retrieval lane

**Bead**: scix_experiments-xz4.1.38 (Option C, decided 2026-04-22)
**Status**: Draft 2026-04-29
**Predecessors**: xz4.1.34 (eval-only narrow fix), xz4.1.33 (flagship_seed)
**Author**: scix-worker-2-gc-65767
**Type**: PRD-before-build

## TL;DR

Replace the paper-community heuristic shipped in xz4.1.34
(`CommunityExpansionBackend` in `scripts/eval_entity_value_props.py`) with a
real product feature in `src/scix/search.py`: an **entity co-occurrence**
retrieval lane that takes a seed entity, finds entities most frequently
co-mentioned on seed-linked papers, and returns papers that mention those
neighbor entities. Wire it into the MCP `search` tool behind a
`community_expand: bool = False` flag.

The reframing (from "paper community" to "entity neighborhood") is what
matches the gold-set expectations: "HST ecosystem" = WFC3, STIS, ACS, COS,
STScI — sibling **entities** linked through shared papers, not papers in
the same Leiden cluster.

## 1. Goal & non-goals

### Goal

Given a seed entity (resolved or explicitly passed), return up to `top_k`
papers ranked by:

1. Seed → neighbor-entity co-occurrence count (descending).
2. Within neighbor entities, paper PageRank (descending, NULLS LAST).
3. Excluding papers already linked to the seed entity itself, so the result
   is genuinely "siblings" rather than "seed mentions".

### Non-goals

- Default-on promotion of the lane in MCP `search` (eval-gated like
  `enable_alias_expansion`, `enable_ontology_parser`).
- Corpus extraction of flagship sub-instruments from body text
  (WFC3, NIRSpec, ACIS, etc.) — separate corpus-data bead.
- ML-learned community assignment / GNN embeddings — premature.
- Replacing `paper_metrics.community_semantic_*` columns or the
  `explore_community` / `lit_review` callsites that read them — those still
  serve `graph_context` and topic-bucket use cases.

## 2. Why entity co-occurrence (not paper communities)

The xz4.1.34 fix used `paper_metrics.community_semantic_medium`: pick the
modal Leiden community across papers linked to the seed, then return the
top-PageRank papers in that community.

This scored **0.00 → some-positive** in dev runs but is fundamentally the
wrong shape. Three structural problems:

1. **Topic buckets are too broad.** "HST ecosystem" sits inside the
   semantic community for "observational astronomy" or "UV spectroscopy",
   which contains thousands of unrelated missions. The judge wants WFC3,
   not generic UV papers.
2. **Modal-community selection collapses information.** A heavy seed (HST,
   ~10k papers) spans many topics. The single modal community discards 90%
   of the structure the seed actually carries.
3. **Doesn't generalize to the ecosystem framing.** Co-occurrence is what
   the gold-set asks for: "papers about WFC3" co-occur with "papers about
   HST" because the same papers mention both. The Leiden clustering
   doesn't capture this directly.

Entity co-occurrence is a direct query against the structure the gold-set
rewards.

## 3. Function design

### Signature

```python
def community_expand_search(
    conn: psycopg.Connection,
    seed_entity_id: int,
    *,
    top_k: int = 20,
    min_cooccurrence: int = 2,
    neighbor_limit: int = 50,
    seed_paper_cap: int = 5_000,
) -> SearchResult: ...
```

`SearchResult` matches the rest of `search.py` — `papers`, `total`,
`timing_ms`, `metadata`. No new return type.

### SQL sketch (to be benchmarked, not committed verbatim)

```sql
-- Stage 1: cap the seed-paper sample (super-hub guard).
WITH seed_papers AS (
  SELECT bibcode
  FROM document_entities_canonical
  WHERE entity_id = %(seed_entity_id)s
  ORDER BY fused_confidence DESC
  LIMIT %(seed_paper_cap)s
),
-- Stage 2: rank co-occurring neighbor entities by paper-overlap count.
neighbors AS (
  SELECT dec.entity_id, COUNT(*) AS cooccur_count
  FROM document_entities_canonical dec
  JOIN seed_papers sp ON sp.bibcode = dec.bibcode
  WHERE dec.entity_id <> %(seed_entity_id)s
  GROUP BY dec.entity_id
  HAVING COUNT(*) >= %(min_cooccurrence)s
  ORDER BY cooccur_count DESC
  LIMIT %(neighbor_limit)s
),
-- Stage 3: papers mentioning a neighbor (excluding seed-linked papers),
-- ordered by best-neighbor-cooccur-count then PageRank.
candidate_papers AS (
  SELECT
    dec.bibcode,
    MAX(n.cooccur_count) AS best_cooccur,
    MAX(pm.pagerank)     AS pagerank
  FROM neighbors n
  JOIN document_entities_canonical dec ON dec.entity_id = n.entity_id
  JOIN paper_metrics pm                ON pm.bibcode    = dec.bibcode
  WHERE NOT EXISTS (
    SELECT 1 FROM document_entities_canonical dec_seed
    WHERE dec_seed.bibcode = dec.bibcode
      AND dec_seed.entity_id = %(seed_entity_id)s
  )
  GROUP BY dec.bibcode
)
SELECT p.bibcode, p.title, p.first_author, p.year, p.citation_count, p.abstract,
       cp.best_cooccur, cp.pagerank
FROM candidate_papers cp
JOIN papers p ON p.bibcode = cp.bibcode
ORDER BY cp.best_cooccur DESC, cp.pagerank DESC NULLS LAST
LIMIT %(top_k)s;
```

`document_entities_canonical` is the right source over raw
`document_entities`: it's deduplicated, fused-confidence-ranked, and
indexed on `(entity_id, fused_confidence DESC)` plus `(bibcode, entity_id)`.
The mat-view also already filters out low-confidence noise.

### `metadata` payload

`SearchResult.metadata` should expose, at minimum:

- `seed_entity_id`
- `seed_paper_count` (post-cap)
- `neighbor_count` (rows from Stage 2)
- `neighbors`: top-10 `(entity_id, canonical_name, cooccur_count)` for
  observability / agent surfacing. **Do not** include all neighbors —
  blows token budget for super-hubs.
- `truncated_seed_papers: bool` (true when seed had > `seed_paper_cap`).

`timing_ms` should record `cooccur_neighbors_ms` and `cooccur_papers_ms`
separately so we can spot regressions.

## 4. MCP wiring

### `search` tool changes

Add to `inputSchema.properties`:

```jsonc
"community_expand": {
  "type": "boolean",
  "default": false,
  "description":
    "When true, run a community-expansion lane in addition to (or instead of) "
    "hybrid retrieval. The seed entity is taken from filters.entity_ids[0] when "
    "provided; otherwise the server attempts to resolve the query string to a "
    "single unambiguous entity. If neither resolves, the tool returns a "
    "structured {error_code: 'community_expand_no_seed', ...} response. Off by "
    "default — gated like alias_expansion until per-prop eval lands."
}
```

### Seed-resolution decision tree

1. **Explicit**: `filters.entity_ids` has exactly one entry — use it.
2. **Inferred**: free-text resolution via the existing entity tool's
   `resolve` action, requiring a single unambiguous match. Multiple
   matches → return disambiguation envelope (reuse the existing
   `disambiguate=true` machinery — do not invent a second envelope).
3. **No match**: structured error `community_expand_no_seed` with hint
   pointing at the `entity` tool.

### Composition with hybrid

For v1, when `community_expand=true`, the lane **replaces** standard
hybrid retrieval rather than RRF-fusing into it. Reasons:

- The two lanes target different semantics (lexical/dense match vs.
  graph-neighborhood). RRF fusion of incompatible signal types tends to
  dilute both.
- It makes the eval signal clean: a community_expansion result is or
  is not driven by the new lane.

A future bead can add `community_expand_weight` to fuse if eval shows
benefit.

## 5. Eval harness wiring

`scripts/eval_entity_value_props.py:CommunityExpansionBackend`:

- Drop `_modal_community` and `_community_siblings`.
- `retrieve()` calls `community_expand_search(conn, entity_id, top_k=top_k)`
  and converts `SearchResult.papers` → `list[RetrievalDoc]`.
- Keep `_resolve_entity_id` (unchanged) — it's the same entity-name → id
  resolver and isn't tied to the old paper-community logic.
- `min_cooccurrence` and `neighbor_limit` use defaults; eval doesn't tune
  them.

Acceptance: rerun produces a `docs/eval/entity_value_props_2026-04.md`
delta where `community_expansion` mean ≥ 1.5 / 3.0 across the 10-query
gold set.

## 6. Risks and stress tests

### R1: super-hub seed (P0)

`Frequency` (138k papers) and `Robustness` (118k) are real entities in the
corpus. Seed-paper count alone could OOM the planner.

- **Mitigation 1**: `seed_paper_cap` (default 5,000) caps Stage 1.
- **Mitigation 2**: `LIMIT neighbor_limit` (default 50) caps Stage 2 — a
  super-hub is unlikely to have a meaningful "neighborhood" anyway, so
  truncating to top-50 by cooccur_count is the right default.
- **Premortem**: a super-hub is also unlikely to be a *useful* seed. The
  unscoped-broad-query guard pattern from bead `uerc` is the right model:
  if `seed_paper_count > 50_000`, return a structured
  `{error_code: 'seed_too_broad', hint: 'narrow to a more specific entity'}`
  envelope. Do not silently return mediocre results.

### R2: empty neighborhood (P1)

A long-tail entity (a freshly-extracted method with one paper) has no
co-mentioned siblings.

- **Behaviour**: return `SearchResult(papers=[], total=0,
  metadata={seed_entity_id, seed_paper_count, neighbor_count: 0})`.
- **Do not** fall back to standard hybrid — we'd hide the signal and
  agents would learn to trust the lane on inputs where it doesn't apply.
- The eval gold-set queries are all heavy seeds (HST, JWST, ...) so this
  case isn't directly tested but must not crash.

### R3: latency budget (P1)

Two-stage CTE on a 94M-row mat-view will stress HNSW's neighbour ef and
buffer pool. Targets:

- p50 ≤ 800 ms, p95 ≤ 2 s for typical seeds (≤ 5k papers).
- Statement timeout per the existing `SCIX_TIMEOUT_SEARCH` (30 s) — but
  if the MCP-level p95 routinely exceeds 5 s the lane needs an EXPLAIN
  pass before merge.

### R4: noisy neighbors (P2)

`Temperature` (85k papers, observable) co-occurs with everything. Without
filtering, neighbors will skew toward universal "concepts" rather than
ecosystem-specific siblings.

- **Mitigation**: the SQL already filters `entity_id <> seed_entity_id`.
  Beyond that, v1 ships without a neighbor-type filter — eval will
  expose whether the universal-concept noise dominates.
- If eval shows the noise problem, follow-up bead adds an
  `entity_types` filter parameter (e.g. only neighbours that are
  instruments / missions / methods).

### R5: stale `document_entities_canonical` (P2)

The mat-view is refreshed on a cadence (see migration 044). A new
flagship_seed entity won't appear in cooccur results until the next
refresh.

- **Mitigation**: document this in the function docstring and the MCP
  tool description. Operators have an existing playbook for forcing
  refresh.

### R6: filter interaction (P2)

What happens when the caller passes `filters.year_min`, `entity_types`,
or `arxiv_class` together with `community_expand=true`?

- **Decision**: filters apply at Stage 3 (candidate_papers) only. The
  neighborhood is computed on the whole graph; the year/discipline
  filter narrows the *output* papers. This matches user intent: "show me
  HST-ecosystem papers from 2024" should still let the neighborhood
  include WFC3 even if 2024-WFC3 papers are scarce.
- Document this decision in the function docstring.

### R7: ranking explainability (P2)

`best_cooccur DESC, pagerank DESC` is opaque. Agents and human reviewers
can't tell why paper X ranked above Y.

- **Mitigation**: include `cooccur_count` and the `best_neighbor_id` in
  per-paper metadata so the response is auditable.

## 7. Implementation plan (TDD)

1. **Tests first** (`tests/test_search_community_expand.py`):
   - `test_seed_resolution_pulls_neighbors` — fixture: 1 seed, 3
     neighbors, 5 papers; verify Stage 2 ranking is correct.
   - `test_excludes_seed_linked_papers` — papers tagged with both seed
     and a neighbor must not appear.
   - `test_pagerank_tiebreak` — two neighbors with equal cooccur, papers
     with NULL vs. real PageRank → NULL last.
   - `test_empty_neighborhood_returns_empty_result` — seed with one
     isolated paper.
   - `test_super_hub_seed_returns_structured_error` —
     `seed_paper_count > 50_000` → error envelope.
   - `test_min_cooccurrence_filters_singletons` — `min_cooccurrence=2`
     drops neighbors with only 1 shared paper.
2. **Function** in `src/scix/search.py` — public API alongside
   `hybrid_search`, `lit_review`, etc.
3. **MCP wiring** in `mcp_server.py` — schema + dispatch in the `search`
   action, including the `community_expand_no_seed` and `seed_too_broad`
   error envelopes (mirror the patterns from `uerc` / `x5jg`).
4. **Integration test** (`tests/test_mcp_search_community_expand.py`):
   round-trip via the MCP dispatch layer with a stub conn.
5. **Eval harness rewrite** —
   `scripts/eval_entity_value_props.py:CommunityExpansionBackend`
   delegates to the new function.
6. **Eval re-run** — produce
   `docs/eval/entity_value_props_2026-04_communityexpand.md` showing
   `community_expansion` mean ≥ 1.5 / 3.0.

## 8. Acceptance criteria

- [ ] `community_expand_search()` exported from `scix.search` with
      docstring + type signatures.
- [ ] Unit tests (six cases above) pass.
- [ ] MCP `search` tool accepts `community_expand: bool` arg; integration
      test asserts wire format.
- [ ] `CommunityExpansionBackend` retires its in-harness SQL and
      delegates to the new function.
- [ ] Eval mean ≥ 1.5 / 3.0 on 10-query gold set; report committed.
- [ ] Function listed in `src/scix/search.py` public-API comment block
      (top of file).

## 9. Open questions (decide before merge)

- **Q1**: Is `seed_paper_cap=5_000` the right default? Frame as "what's
  the median paper count for a useful seed?" — instruments cluster
  around 100–10k, so 5k is mid-band. Could be 10k. Benchmark Stage 1
  query plan with a 138k-paper seed before locking.
- **Q2**: Should the lane fuse via RRF instead of replacing hybrid? v1
  says replace; followup bead reconsiders if eval shows complementary
  signal.
- **Q3**: Should `entity_types` filtering happen on neighbors (Stage 2)
  or only on output papers (Stage 3)? Default v1: output-only. R4
  flag says revisit if "universal concept" neighbors dominate top-k.

## 10. Out of scope (followup beads)

- Default-on promotion of `community_expand` once eval clears.
- `community_expand_weight` / RRF fusion mode.
- Neighbor `entity_types` filter (R4 followup).
- Caching of (seed → neighbor) lists for hot-path seeds.
- Body-text extraction of flagship sub-instruments (WFC3, NIRSpec, ACIS,
  ...) — referenced as a corpus-data dependency by xz4.1.38 but a
  separate bead.

## References

- Parent: `scix_experiments-xz4.1` — entity-enrichment production
  rollout.
- Predecessor (eval-only fix): `scix_experiments-xz4.1.34`.
- Data dependency: `scix_experiments-xz4.1.33` (flagship_seed in prod).
- MCP error-envelope precedents: `scix_experiments-uerc`,
  `scix_experiments-x5jg`.
- Premortem framework: `docs/premortem/premortem_entity_enrichment_strategy.md`.
- Gold set: `data/eval/entity_value_props/community_expansion.yaml`.
