# PRD: `entity_clusters` — Cross-Ontology Entity Deduplication

**Status:** Draft · **Bead:** `scix_experiments-xz4.4` (epic `xz4`) · **Origin:** Open Question 7 in `docs/prd/archived/prd_entity_enrichment_strategy.md` — deferred as a follow-on until M13 shipped (it has).

**Pairs with:** `docs/prd/prd_wikidata_backfill.md` (N2). That PRD treats clustering as a *sub-step of the Wikidata harvest* (§5). This PRD promotes cross-ontology dedup to a **standing, source-agnostic capability** of the entity graph and owns the read-path changes the Wikidata PRD leaves unspecified.

---

## 1. Problem

A single real-world concept is represented as multiple rows in `entities`, one per harvest source, because the table's natural key is `(canonical_name, entity_type, source)` (migration `021_entity_graph.sql`). There is no mechanism that records "these N rows are the same thing."

Concrete instance (the motivating case): **Mars** appears as four distinct `entities` rows —

| source | canonical_name | entity_type |
|--------|----------------|-------------|
| `ssodnet` | Mars | observable / body |
| `gcmd` | Mars | observable |
| `wikidata` | Mars | observable |
| `vizier` | Mars | observable |

All four are real, all four are harvested by scripts that exist today (`scripts/harvest_ssodnet.py`, `harvest_vizier.py`, GCMD via `harvest_full.py`, `enrich_wikidata_multi.py`). The consequences:

1. **Agent-facing duplication.** `agent_entity_context` (materialized view, migration `055_agent_entity_context_rewrite.sql`) emits **one row per `entities.id`**, so the `entity_context` MCP tool surfaces Mars four times. This is the UX problem flagged in the epic.
2. **Resolver fan-out.** `EntityResolver.resolve()` (`src/scix/entity_resolver.py`) returns one `EntityCandidate` per matching row and is documented as "Never limited to one result." A mention of "Mars" yields four candidates that the caller cannot collapse, inflating `document_entities` link counts and skewing any per-entity aggregation.
3. **Fragmented signal.** Aliases, identifiers, and `document_entities` links accrue against whichever source row happened to match, so no single row is "complete." A query against the GCMD Mars misses the Wikidata QID attached to the Wikidata Mars.

This is purely a **read/identity** problem. The underlying rows are correct and provenance-bearing; we must not destroy them. We need a layer on top that says "these rows co-refer, and here is the preferred representative."

### 1.1 Non-goals

- **Not** merging or deleting source `entities` rows. Provenance per source is load-bearing for harvest re-runs and audit (`entity_audit_log`, migration `025`). Clustering is additive.
- **Not** a Wikidata feature. Wikidata is one of ~10 sources that participate; the capability must work for an all-ontology graph even if the Wikidata backfill never runs.
- **Not** entity *disambiguation* at link time (homograph "Mercury" planet vs. element). That is `src/scix/jit/disambiguator.py`'s job and stays separate; see §7.3.

---

## 2. Current State (grounded)

| Component | File | Relevance |
|-----------|------|-----------|
| `entities` natural key `(canonical_name, entity_type, source)` | `migrations/021_entity_graph.sql` | Source of the per-source row multiplicity |
| `entity_identifiers` (`id_scheme`, `external_id`) | `migrations/021_entity_graph.sql` | QID-based clustering input; PK `(id_scheme, external_id)` |
| `entity_aliases` | `migrations/021_entity_graph.sql` | Alias-overlap clustering input |
| `entity_relationships` w/ `same_as` predicate | `migrations/021`, `050_entity_relationships_evidence.sql` | Existing weak signal; **not** a cluster table |
| `ambiguity_class` / `link_policy` enums | `migrations/028_entity_schema_hardening.sql` | Gate over-clustering of homographs |
| `EntityResolver.resolve()` | `src/scix/entity_resolver.py` | Read path that fans out per source |
| `resolve_entities()` (M13 canonical writer) | `src/scix/resolve_entities.py` | Sole writer to `document_entities`; integration point |
| `agent_entity_context` matview | `migrations/055_*.sql`, `src/scix/views.py` | Where duplicate rows surface to agents |
| Harvest sources in use | `src/scix/sources/`, `scripts/harvest_*.py` | `aas, ascl, gcmd, physh, pwc, spase, spdf, ssodnet, vizier, wikidata` |

There is **no** existing cluster/merge mechanism (confirmed: no `cluster` table, no merge job). The `same_as` predicate in `entity_relationships` records pairwise assertions but has no canonical-representative concept and no transitive-closure guarantee, so it cannot answer "give me the one Mars."

---

## 3. Design

### 3.1 Schema — migration `069_entity_clusters.sql`

Two LOGGED tables (never `UNLOGGED` — see `feedback_unlogged_tables`; we lost 32M embeddings to that mistake once).

```sql
-- 069_entity_clusters.sql
BEGIN;

CREATE TABLE IF NOT EXISTS entity_clusters (
    cluster_id    SERIAL PRIMARY KEY,
    canonical_id  INT  NOT NULL REFERENCES entities(id) ON DELETE RESTRICT,
    label         TEXT NOT NULL,
    entity_type   TEXT NOT NULL,           -- denormalized for type-scoped queries
    member_count  INT  NOT NULL DEFAULT 1, -- maintained by the clustering job, not a trigger
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS entity_cluster_members (
    cluster_id    INT  NOT NULL REFERENCES entity_clusters(cluster_id) ON DELETE CASCADE,
    entity_id     INT  NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    confidence    REAL NOT NULL DEFAULT 1.0 CHECK (confidence > 0.0 AND confidence <= 1.0),
    match_method  TEXT NOT NULL,
    added_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cluster_id, entity_id)
);

-- An entity belongs to at most one cluster: enforce single membership.
CREATE UNIQUE INDEX IF NOT EXISTS uq_cluster_member_entity
    ON entity_cluster_members(entity_id);

CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster
    ON entity_cluster_members(cluster_id);

COMMIT;
```

Design decisions and *why* (deltas from the Wikidata PRD §5.2):

- **`canonical_id` uses `ON DELETE RESTRICT`, not `CASCADE`.** A cluster must never lose its representative silently; deleting the canonical entity must fail loudly so the clustering job re-elects a representative first. (The Wikidata PRD specified `CASCADE`, which would orphan the cluster's identity.)
- **Single-membership invariant** via `uq_cluster_member_entity`. An entity in two clusters makes "the one Mars" ambiguous again. Cross-cluster co-reference is resolved by *merging* clusters, not by dual membership.
- **`entity_type` denormalized onto `entity_clusters`** so the read path can type-scope without a join, and so a cluster's type mismatch (a `mission` clustered with an `instrument`) is queryable for audit. Members must share a compatible type (§3.3 Phase 3).
- **`member_count` is maintained by the job, not a trigger.** Per ZFC/anti-slop: a trigger here is hidden behavior; the batch clustering job already rewrites membership transactionally and can set the count in the same statement.
- **No `UNLOGGED`, no partitioning.** Cluster cardinality is O(10⁴), trivially small next to `document_entities`.

### 3.2 Clustering job — `scripts/cluster_entities.py`

A standalone, idempotent batch job that (re)builds clusters over the **entire** `entities` table — not just newly harvested rows. Runnable via `scix-batch` (it is a full-table scan; see `compass-memory-isolation`). Must accept `--dry-run` (report cluster deltas, write nothing) and `--source <s>` (re-cluster only rows touching one source, for incremental post-harvest runs).

Idempotency contract: running it twice with no intervening harvest produces zero membership changes.

### 3.3 Clustering algorithm — three deterministic phases

Mechanical, calibrated-threshold matching only — the ZFC "duplicate/similarity detection with calibrated thresholds" allowed exception. No semantic LLM judgment in the hot path.

**Phase 1 — Identifier exact (`match_method='id_exact'`, confidence 1.0).**
Group entities sharing any `(id_scheme, external_id)` in `entity_identifiers` (Wikidata QID, but also DOI, ADS bibstem, GCMD UUID — scheme-agnostic). Same external ID across schemes is the strongest possible co-reference signal. Transitive: if A≡B by QID and B≡C by GCMD UUID, all three cluster.

**Phase 2 — Name exact (`match_method='name_exact'`, confidence 0.9).**
For entities not yet clustered, group by `(lower(canonical_name), entity_type)` across differing `source`. Uses existing index `idx_entities_canonical_lower`. Type must match exactly here (the cheap, safe signal).

**Phase 3 — Alias overlap (`match_method='alias_overlap'`, confidence 0.85).**
For still-unclustered entities, join `entity_aliases` to find a shared alias (or alias↔canonical_name) with a *compatible* entity_type. Type compatibility is an explicit allow-list (e.g. `mission`↔`instrument` for spacecraft-that-are-also-telescopes; see Wikidata PRD Open Question 3), **not** a fuzzy judgment.

**Hard gate against over-clustering:** entities with `ambiguity_class='homograph'` (migration `028`) are **never** auto-clustered by Phase 2/3 — only by Phase 1 identifier match. "Mercury" the planet and "Mercury" the element will not merge on name alone. This directly mitigates the highest-severity risk in the Wikidata PRD (§9.1 over-clustering).

**Canonical representative election** (within each cluster), by source-authority tiebreaker — transparent, deterministic ranking, the other ZFC-allowed exception:

1. Domain-authoritative curated source: `gcmd`, `spase`, `aas`, `ssodnet`, `vizier` (expert-reviewed)
2. `ads_data` / corpus-derived
3. `wikidata` (broad but lower per-entry curation)
4. Tiebreak within a tier: most `entity_aliases` + `entity_identifiers` (most complete record), then lowest `entities.id` (stable).

`label` = canonical entity's `canonical_name`.

### 3.4 Read path — where dedup is *consumed*

This is the half the Wikidata PRD omits. Two integration points, both fold members into the canonical representative:

**(a) `agent_entity_context` matview (migration `070_*`, rewrites `055`).**
Add a `LEFT JOIN entity_cluster_members` so each entity carries its `cluster_id`. Provide a **new** matview `agent_entity_context_clustered` keyed on `cluster_id` that emits **one row per cluster** (canonical name/type/discipline, union of aliases, summed `doc_count`, array of member sources). The `entity_context` MCP tool reads the clustered view by default; an `expand_sources=true` arg falls back to the per-entity view for provenance drill-down. Column set of the existing view is preserved (no breaking change to current callers).

**(b) `EntityResolver` and `resolve_entities()`.**
Add an opt-in `collapse_clusters: bool` to `EntityResolver.resolve()` (default `False` — preserves the documented "never limited to one result" contract for callers that need raw candidates). When `True`, candidates sharing a `cluster_id` collapse to the canonical entity, keeping the **max** confidence and recording collapsed member ids in `evidence`. `resolve_entities()` (M13) passes `collapse_clusters=True` so `document_entities` links land on canonical entities, eliminating the 4×-link inflation. **Constraint:** all writes to `document_entities` still flow solely through `resolve_entities()` — the AST lint (`scripts/ast_lint_resolver.py`) invariant is untouched.

---

## 4. Backfill

One-time over the existing ~90K-row graph (independent of any Wikidata harvest):

1. `scix-batch python scripts/cluster_entities.py --dry-run` — report projected cluster count and the top-50 largest clusters for eyeballing.
2. Manual spot-check of the dry-run report (see §6 quality audit).
3. `scix-batch python scripts/cluster_entities.py` — commit clusters.
4. `REFRESH MATERIALIZED VIEW CONCURRENTLY agent_entity_context_clustered`.

Incremental: each harvester's post-run hook calls `cluster_entities.py --source <s>` so new rows join existing clusters without a full re-scan.

---

## 5. Schema changes summary

| Migration | Change | Tables |
|-----------|--------|--------|
| `069_entity_clusters.sql` | New cluster tables | `entity_clusters`, `entity_cluster_members` |
| `070_agent_entity_context_clustered.sql` | Cluster-keyed matview + `cluster_id` on existing view | `agent_entity_context*` |

No structural change to `entities`, `entity_identifiers`, `entity_aliases`, `entity_relationships`, or `document_entities`. Clustering is strictly additive.

---

## 6. Evaluation

### 6.1 Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Cross-ontology clusters (size ≥ 2) | 0 | 4K–8K | `SELECT count(*) FROM entity_clusters WHERE member_count >= 2` |
| `entity_context`("Mars") rows | 4 | 1 (with N source provenances) | MCP tool call, default args |
| `document_entities` link inflation factor | ~1.0–1.4× per clustered mention | 1.0× | links before vs. after `collapse_clusters` on a 10K-bibcode sample |
| Largest cluster size | n/a | ≤ ~12 (sanity ceiling) | `SELECT max(member_count) FROM entity_clusters` — a 200-member cluster is a Phase-3 false-merge smell |
| Homograph false merges | n/a | 0 | audit of `ambiguity_class='homograph'` entities' cluster assignments |

### 6.2 Quality audit

- **False-merge precision:** manually inspect 50 random clusters (weighted toward size ≥ 3) for incorrectly merged distinct entities. Target ≥ 95% precision.
- **Missed-merge recall:** sample 50 known cross-ontology duplicates (Mars, Hubble/HST, Chandra, JWST, Gaia) and verify each forms exactly one cluster.
- **Canonical election sanity:** verify the elected representative is the most-complete record for 20 sampled clusters.

### 6.3 Test plan (ships in the same commits — see agent-collaboration §Tests Ship With Fixes)

- Unit: each of the three phases against a fixture graph with hand-built duplicates and one homograph trap; canonical-election tiebreaker ordering; single-membership invariant violation raises.
- Integration (needs `SCIX_TEST_DSN`): apply `069`, run `cluster_entities.py` on seeded `entities`, assert cluster shape; assert `resolve(collapse_clusters=True)` collapses; assert AST-lint still passes (no new `document_entities` writer).
- Idempotency: run job twice, assert zero membership delta.

---

## 7. Risks & open questions

### 7.1 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Phase-3 over-clustering (name collisions merge distinct entities) | High | Homograph hard-gate; confidence floor; size-ceiling alert; manual audit of size ≥ 3 |
| Cluster goes stale after a harvest re-run renames an entity | Medium | Incremental `--source` re-cluster in each harvester's post-run hook; idempotent rebuild |
| Canonical entity deleted out from under a cluster | Medium | `ON DELETE RESTRICT` on `canonical_id` forces re-election first |
| Read-path regression for callers relying on per-source rows | Medium | `collapse_clusters` defaults `False`; clustered matview is additive; `expand_sources` escape hatch |
| Double-counting if both clustered and unclustered views are summed by a downstream consumer | Low | Document that the two views are mutually exclusive aggregation domains |

### 7.2 Open questions

1. **Should `same_as` relationships in `entity_relationships` seed Phase 1?** They are weaker than identifier matches but capture human-asserted equivalences. Proposed: yes, as a Phase 1.5 at confidence 0.95, gated on the relationship's own `confidence ≥ 0.9`.
2. **Re-cluster cadence for the full job?** Proposed: monthly, aligned with harvester schedules; incremental per-source after each harvest.
3. **Do clusters participate in citation/community analytics?** Out of scope here; flagged for the entity-graph analytics epic.

### 7.3 Explicitly out of scope

- Link-time homograph disambiguation (`src/scix/jit/disambiguator.py`) — unchanged.
- Contributing equivalences back to upstream ontologies.
- Fuzzy/embedding-based clustering — deferred until the deterministic three-phase precision is measured; adding it now is premature (YAGNI).

---

## 8. Sequencing

| Phase | Work | Depends on |
|-------|------|-----------|
| 1 | Migration `069` (cluster tables) + unit-test scaffold | — |
| 2 | `scripts/cluster_entities.py` three-phase algorithm + canonical election | 1 |
| 3 | Migration `070` clustered matview + `entity_context` MCP read path | 1 |
| 4 | `collapse_clusters` in `EntityResolver` / `resolve_entities()` | 1, 2 |
| 5 | Full-table backfill + quality audit | 2, 3, 4 |

Phase 1–4 land with their tests in the same commits. Phase 5 is an operational run, gated on the §6.2 audit passing.
