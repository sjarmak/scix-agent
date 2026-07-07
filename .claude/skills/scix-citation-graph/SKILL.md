---
name: scix-citation-graph
description: >
  The SciX citation-graph layer: graph_metrics.py (PageRank/HITS/Leiden),
  the paper_metrics and communities tables, the (signal, resolution,
  community_id) model, the 299M-row citation_edges table, and the MCP
  citation/provenance tools (citation_traverse, forward_citations incl. the
  find_replications/cited_by_intent aliases, claim_blame, graph_context) with
  citation-intent classification. Load when computing or debugging PageRank,
  HITS, Leiden communities, community labels, citation traversal, claim
  provenance, replication finding, or citation intent. NOT for the RRF search
  stack (use scix-retrieval-architecture), NOT for the Qdrant dense lane or
  embeddings ingest (use scix-vector-serving-qdrant / scix-embedding-pipeline),
  NOT for MCP tool-cap/contract mechanics (use scix-mcp-tool-surface), NOT for
  the entity graph (use scix-entity-ner-system), NOT for query-side agent
  usage of the tools (use the existing scix-mcp skill).
---

# SciX Citation Graph — structural topology layer

The differentiated surface of this project beyond ranked-list search: the
citation graph (299M edges over 32.4M papers), precomputed graph metrics,
three community partitions, and the provenance tools built on them. This
skill is the maintainer-side runbook: schema, pipeline anatomy, what is
actually populated on prod, the committed-code drift you must not trip over,
and how to run or diagnose any of it safely.

Verified against branch `bd/0yp5-external-copy-accuracy-audit` @ `452ab86`
(2026-07-07; NOT main — 5 commits behind `origin/main`, but every file this
skill cites is byte-identical to `origin/main`, checked with
`git diff HEAD..origin/main -- <files>`).

## When NOT to use this skill

| You are working on                       | Use instead                        |
| ---------------------------------------- | ---------------------------------- |
| RRF fusion, BM25 lanes, search quality   | `scix-retrieval-architecture`      |
| Qdrant collections, dense serving        | `scix-vector-serving-qdrant`       |
| INDUS embed/ingest, the s7cy fire        | `scix-embedding-pipeline`          |
| Tool cap, contract regen, alias plumbing | `scix-mcp-tool-surface`            |
| GLiNER entities, document_entities       | `scix-entity-ner-system`           |
| Gold sets, nDCG, evidence bar            | `scix-eval-and-evidence`           |
| Driving the tools as a research agent    | `scix-mcp` (query-side skill)      |
| DSN guards, prod protection generally    | `scix-db-safety-and-telemetry`     |
| scix-batch / OOM discipline              | `scix-memory-and-batch-discipline` |

## Jargon (defined once)

- **bibcode** — ADS's 19-char paper ID (e.g. `2006Sci...312.1780P`); the
  primary key everywhere in this schema.
- **PageRank / HITS** — link-analysis scores. PageRank = global importance;
  HITS yields a _hub_ score (cites many authorities) and an _authority_
  score (cited by many hubs). Both computed with igraph, directed.
- **Leiden** — community-detection algorithm (successor to Louvain), via
  `leidenalg`. A _community_ is a cluster of densely inter-citing papers.
- **resolution** — Leiden granularity knob. Here it is discretized to three
  named levels: `coarse` / `medium` / `fine` (~20 / ~200 / ~2000 target
  communities).
- **signal** — WHICH community partition: `citation` (Leiden on the citation
  graph), `semantic` (minibatch k-means over INDUS embeddings), `taxonomic`
  (arXiv class string). Formalized by migration 052.
- **giant component** — the largest weakly-connected subgraph. Leiden runs
  only there; isolated/small-component papers are handled separately.
- **NMI / conductance / coverage** — partition-quality metrics implemented
  in `graph_metrics.py` (`compute_nmi`, `compute_conductance`,
  `compute_coverage`).
- **citation intent** — per-citation-context label in
  `citation_contexts.intent`: `background` | `method` | `result_comparison`
  (SciCite label set).

## 1. Data model

### citation_edges (migration 001)

```sql
CREATE TABLE citation_edges (
    source_bibcode TEXT NOT NULL,   -- the citing paper
    target_bibcode TEXT NOT NULL,   -- the cited paper
    PRIMARY KEY (source_bibcode, target_bibcode)
);
CREATE INDEX idx_cite_target ON citation_edges(target_bibcode);
```

Forward traversal (who cites X) hits `idx_cite_target`; backward (what X
cites) hits the PK. Scale numbers (from `results/graph_quality_report.md`,
generated 2026-04-06 — re-measure before quoting in anything new):

| Metric                                 | Value                                               |
| -------------------------------------- | --------------------------------------------------- |
| Nodes (papers)                         | 32,390,237                                          |
| Edges in DB                            | 299,253,213                                         |
| Resolved edges (both ends in `papers`) | 298,058,210 (99.6%)                                 |
| Dangling edges                         | 1,195,003 (0.4%)                                    |
| Isolated nodes (degree 0)              | 12,274,690                                          |
| Connected components                   | 12,330,419                                          |
| Giant component                        | 19,981,157 nodes (61.7% of all; 99.3% of connected) |
| Small-component papers                 | 134,390                                             |
| Out-degree mean / median / p99         | 18.04 / 12 / 97                                     |

### paper_metrics (migrations 006, 008, 051)

One row per paper: `pagerank`, `hub_score`, `authority_score`,
`community_id_{coarse,medium,fine}` (citation signal),
`community_semantic_{coarse,medium,fine}` (k=20/200/2000 k-means, migration
051), `community_taxonomic` (TEXT arXiv class, migration 008), `updated_at`.
Btree indexes on every community column and on `pagerank DESC`.

### communities (migrations 006 + 052)

Label metadata per community. Migration 052 (PRD "community-labels M4")
added `signal TEXT NOT NULL CHECK (signal IN
('citation','semantic','taxonomic'))` and replaced the PK:

```
PRIMARY KEY (signal, resolution, community_id)
```

Columns: `community_id`, `resolution` (`coarse|medium|fine` CHECK), `signal`,
`label`, `paper_count`, `top_keywords TEXT[]`, `updated_at`. The old 2-col
PK `(community_id, resolution)` collided across signals. Any INSERT into
`communities` MUST supply `signal` and conflict-target the 3-col key —
see the drift trap in §3.

### What is actually populated on prod (as of 2026-07-07, source-read only)

Could not connect to the DB to verify (prod-DB access is gated); the
following is what committed code and artifacts assert:

- **Citation Leiden has NEVER completed on prod.** Per the
  `_handle_find_gaps` docstring (`src/scix/mcp_handlers/entity.py`) and
  `_fetch_query_mode_buckets` (`src/scix/search.py:2205`): Phase A marked
  non-giant papers with sentinel `-1`; Phase B (Leiden on the ~20M-node
  giant component) repeatedly OOM'd (state recorded as of 2026-04-24), so
  `community_id_{coarse,medium,fine}` holds only `-1` or NULL.
  `search.py:2179` defines `_NO_COMMUNITY_SENTINEL = -1`. The M3 recompute
  script exists (§4) but `logs/leiden_recompute/run_meta.json` records only
  a test-scale run (giant=0, 2026-04-18). Treat "Leiden landed on prod" as
  FALSE until `verify_communities_populated.py` says otherwise.
- **Semantic is the populated, default partition.** `explore_community`
  and `find_gaps` default `signal='semantic'`.
  `results/semantic_communities.json`: medium k=200 and fine k=2000 over
  32,383,535 rows (silhouette 0.017 / 0.283 respectively).
- **Taxonomic is populated** from `papers.arxiv_class`; single real
  resolution (`coarse`); `medium`/`fine` requests alias to coarse
  (`explore_community` docstring).
- **PageRank/HITS**: computed by `graph_full.sh` Phase 1 era; per-paper
  values served by `graph_context`. Freshness unverified read-only.

## 2. graph_metrics.py anatomy (src/scix/graph_metrics.py, ~1700 lines)

| Function                                                                            | What it does                                                                                                                       | Notes                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_graph(conn, keep_bibcode_to_id=True)`                                         | Streams `papers` + `citation_edges` into a directed igraph via server-side cursors                                                 | Preallocated int32 numpy arrays (~2.4 GB vs ~20 GB of tuples); 5M-edge chunked `add_edges` (~21 GB peak vs ~50 GB naive); `keep_bibcode_to_id=False` drops the ~8 GB dict + `malloc_trim` before igraph allocates; commits at end to release the idle-in-transaction xmin pin |
| `compute_pagerank(graph)` / `compute_hits(graph)`                                   | igraph `pagerank(directed=True)` / `hub_score()`+`authority_score()`                                                               |                                                                                                                                                                                                                                                                               |
| `filter_isolated_nodes(...)`                                                        | Drops degree-0 vertices before Leiden                                                                                              | ~12.3M isolated on full corpus                                                                                                                                                                                                                                                |
| `extract_giant_component(...)`                                                      | Weak connectivity; O(V) sizes/membership passes (the O(C·V) `components[i]` lookup was a real perf bug, fixed in commit `47d7a0c`) |                                                                                                                                                                                                                                                                               |
| `compute_leiden(graph, resolution, seed=42, partition_type='modularity')`           | Undirected collapse then `leidenalg.find_partition`                                                                                | `partition_type`: `'modularity'` (RBConfiguration) or `'CPM'` (CWTS-recommended, resolution-limit-free)                                                                                                                                                                       |
| `calibrate_resolution(...)`                                                         | Log-scale binary search to hit a target community count (targets 20/200/2000)                                                      | Slow on large graphs — `graph_full.sh` uses fixed resolutions instead                                                                                                                                                                                                         |
| `sweep_resolutions(...)` / `compare_partitions(...)`                                | Quality sweep (size stats, conductance, coverage, optional NMI) / pairwise NMI matrix                                              |                                                                                                                                                                                                                                                                               |
| `compute_nmi` / `compute_conductance` / `compute_coverage` / `community_size_stats` | Partition-quality metrics, pure Python/numpy                                                                                       |                                                                                                                                                                                                                                                                               |
| `assign_small_component_communities(...)`                                           | Nearest-community-centroid by embedding cosine for small-component papers                                                          | Depends on embeddings — see trap 3.1                                                                                                                                                                                                                                          |
| `store_metrics(...)`                                                                | COPY into temp staging + upsert into `paper_metrics` in 100k chunks                                                                | Schema-current, safe                                                                                                                                                                                                                                                          |
| `store_community_metadata(...)`                                                     | Upsert into `communities`                                                                                                          | BROKEN vs current schema — trap 3.2                                                                                                                                                                                                                                           |
| `generate_community_labels(conn, resolution_name)`                                  | TF-IDF over `papers.keywords` per community                                                                                        | Reads `community_id_*` (citation cols) only                                                                                                                                                                                                                                   |
| `populate_taxonomic_communities(conn)`                                              | Bulk UPDATE `community_taxonomic` from `arxiv_class` (prefers `astro-ph.*`)                                                        |                                                                                                                                                                                                                                                                               |
| `run_pipeline(...)`                                                                 | Orchestrates all of the above                                                                                                      | Do NOT use as-is — traps 3.1 + 3.2                                                                                                                                                                                                                                            |

Memory discipline in this file is load-bearing, not decoration: RSS logging
(`_rss_gb`), `gc.collect()` + `malloc_trim(0)` via ctypes (glibc does not
return freed arenas otherwise), int32 edge arrays, chunked igraph build.
Preserve these patterns when extending; the giant-component Leiden OOM
history is why they exist.

Install: the `graph` extra — `pip install -e '.[graph]'` gives
`python-igraph>=0.11` + `leidenalg>=0.10` (pyproject.toml). CI installs it;
without it, graph test modules self-skip via `tests/conftest.py`.

## 3. Committed-code drift traps (read before running ANYTHING here)

### 3.1 `_load_embeddings` targets a dropped table

`graph_metrics.py:1403` (`_load_embeddings`) and therefore
`run_pipeline`'s small-component assignment step SELECT from
`paper_embeddings` — a table that was dropped on prod out-of-process
~2026-06-29/30 (bead s7cy). At committed HEAD this step fails on prod.
The direct-to-Qdrant remediation exists but is UNCOMMITTED working-tree
material — PROVISIONAL pending Stephanie (discovery Q2): teach committed
reality, do not canonize the in-flight fix. Until s7cy lands, any
graph run needing embeddings (small-component assignment, NMI vs semantic
seeds) must be scoped to skip that step or run against a test DB that
still has the table (CI schema `ci/scix_test_schema.sql` does). Details of
the fire: `scix-embedding-pipeline`.

### 3.2 `store_community_metadata` predates migration 052

It INSERTs into `communities` WITHOUT `signal` and uses
`ON CONFLICT (community_id, resolution)`. Current schema (052; confirmed
in `ci/scix_test_schema.sql:1188-1198,3194`): `signal` is NOT NULL with no
default, PK is `(signal, resolution, community_id)`. The INSERT fails
(NOT NULL violation; stale conflict target). Consequence:

- **Do not run `run_pipeline` / `scripts/graph_metrics.py` end-to-end** —
  it will die at the label-storage step even if 3.1 is dodged.
- The schema-current label writer is `scripts/generate_community_labels.py`
  (writes `(signal, resolution, community_id)` correctly, line 556-559).
- Fixing `store_community_metadata` is a real, unclaimed cleanup; it
  requires deciding the `signal` parameter (citation) and shipping the
  test in the same commit. Route through change control
  (`scix-change-control`).

### 3.3 The rest of the trap list

1. Never trust `README`/`CHANGELOG` numbers for the graph — re-measure
   (they carry the corpus totals but not partition state).
2. `graph_context`'s `communities` block returns per-signal memberships;
   citation entries are sentinel/-1-or-NULL today. An agent-facing "no
   citation community" is expected, not a bug.
3. `find_gaps` / `explore_community` reject `signal='taxonomic'` vs accept
   it respectively — `find_gaps` supports only `citation|semantic`
   (`_SIGNAL_COLUMN_PREFIX` in `mcp_handlers/entity.py`).
4. `citation_contexts` coverage is ~0.27% of edges (~825K contexts, ~30K
   source papers, ~250K cited papers — handler docstrings, bead 79n).
   Intent annotation on traversal results is sparse BY DESIGN; absence of
   `intent` on an edge means "not covered", not "background".

## 4. Runbooks — computing graph metrics

All of these are heavy, prod-DB-writing batch jobs. On THIS installation
(host co-runs the gascity supervisor) every one of them must be wrapped in
`scix-batch` and passed `--allow-prod` where supported — the
`--allow-prod` guard self-checks `INVOCATION_ID` (set by `systemd-run`)
and refuses plain shells (`recompute_citation_communities.py:446`). That
host discipline is an operational requirement of this installation, not of
the code. See `scix-memory-and-batch-discipline`. **Do not run any command
in this section casually** — hours-long, tens of GB RSS, writes to prod.

| Task                                       | Command (guarded form)                                                                                                                                                                                                                    | Writes                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| PageRank + HITS, full corpus               | `scix-batch --mem-high 40G --mem-max 60G bash scripts/graph_full.sh` (Phase 1) — **do not run casually**; script hardcodes `dbname=scix` in its final stats block and has NO `--allow-prod` guard of its own                              | `paper_metrics`                                                       |
| Citation Leiden (giant component, M3 path) | `scix-batch python scripts/recompute_citation_communities.py --resolution-coarse 1.0 --resolution-medium 2.5 --resolution-fine 10.0 --seed 42 --allow-prod` — **do not run casually**; this is the never-yet-succeeded-on-prod step       | `paper_metrics.community_id_*`, `logs/leiden_recompute/run_meta.json` |
| Semantic k-means communities               | `scix-batch python scripts/compute_semantic_communities.py ... --allow-prod` — **do not run casually**                                                                                                                                    | `paper_metrics.community_semantic_*`                                  |
| Community labels (all 3 signals)           | `scix-batch python scripts/generate_community_labels.py --signal all --seed 42 --allow-prod` — **do not run casually**                                                                                                                    | `communities` + spot-check MD                                         |
| Intent backfill                            | `scix-batch python scripts/backfill_citation_intent.py ...` (SciBERT-SciCite; resumable via `WHERE intent IS NULL`; progress in `ingest_log` under `intent_backfill:citation_contexts`, migration 066) — **do not run casually**; GPU job | `citation_contexts.intent`                                            |

Test-DB forms: `export SCIX_TEST_DSN="dbname=scix_test"` and drop
`--allow-prod`. Scripts resolve `SCIX_TEST_DSN` first by design
(`generate_community_labels.py` docstring). Default DSN is PROD
(`scix-db-safety-and-telemetry`).

Smoke-verifiable without a DB: `pytest --collect-only -q
tests/test_graph_metrics.py tests/test_claim_blame.py
tests/test_find_replications.py tests/test_citation_intent.py` — 176 tests
collected at the pinned SHA. Full unit runs need `SCIX_TEST_DSN`.

Sequencing rule (from the M3/Phase-B history): PageRank/HITS fit in memory;
Leiden on the giant component is THE memory cliff. If you attempt Phase B,
budget from `measure_graph_quality.py`'s two-phase notes (~4 GB scipy CC +
~11 GB igraph Leiden with PG-side temp-table ID mapping) and keep the
`keep_bibcode_to_id=False` + del/gc/malloc_trim choreography of
`recompute_citation_communities.py`. Expected invariant recorded in
`run_meta.json`: coarse largest community ≤ 10% of giant.

## 5. Read-only diagnostics (safe to run; still respect the DSN)

| Question                                          | Command                                                                                                                                                                           |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Did community detection actually populate the DB? | `python scripts/verify_communities_populated.py` — read-only, checks migrations 051/052 DDL + per-column row counts; no `--allow-prod` needed                                     |
| What fraction of papers has each signal?          | `python scripts/report_community_coverage.py` — read-only SELECTs (`results/community_coverage.json`; the checked-in copy is all zeros = ran against an empty DB, do not cite it) |
| Graph shape numbers for the paper                 | `scix-batch python scripts/measure_graph_quality.py` — read-heavy (loads the graph, ~17 GB envelope), so batch-wrap it; **do not run casually**                                   |
| Partition quality sweep                           | `scripts/measure_graph_leiden.py`, `scripts/calibrate_leiden.py` — heavy, batch-wrap                                                                                              |

## 6. MCP citation/provenance tools

15 visible tools total (cap is ADR-pinned; import-time `RuntimeError` past
15 — mechanics in `scix-mcp-tool-surface`). The citation-graph subset, all
verified in `contract/scix_mcp_v1.json` (CONTRACT_VERSION "1"):

### citation_traverse (`mcp_handlers/citation.py`)

- `mode='graph'` (default): neighborhood walk. Single `bibcode`, explicit
  `bibcodes=[...]`, or fall-through to the session's focused papers
  (working-set mode, results keyed `by_bibcode`). `direction`:
  `forward` (papers citing X) | `backward` (X's references) | `both`.
  `limit` default 20. Working-set mode uses batched queries
  (`get_citations_batch`/`get_references_batch`) — constant DB round-trips
  regardless of set size (the per-bibcode loop timed out; bead sd71).
- `mode='chain'`: shortest citation path `source_bibcode` →
  `target_bibcode`, Python-driven BFS, `max_depth` clamped 1..5, visited
  cap 100,000 nodes (`search.citation_chain`).
- Validation runs BEFORE any DB access; structured errors carry
  `error_code` (`missing_required_params`, `INVALID_MODE`,
  `INVALID_DIRECTION`) — never raises across the MCP boundary.
- Each returned neighbor gains an `intent` field IFF the edge is covered
  in `citation_contexts` (sparse — trap 3.3.4).
- Deprecated aliases routed here via `_ALIAS_TRANSFORMS` (single source of
  truth, `mcp_server.py`): `citation_graph`, `citation_chain`,
  `get_citations`, `get_references`, `get_citation_context`.

### forward_citations (bead 9afa merge)

One anchor param `bibcode` (`target_bibcode` accepted as synonym);
`annotate` enum selects the leg, responses byte-compatible with the
pre-merge tools:

- `annotate='intent'` (default) → the old `cited_by_intent`: papers citing
  the target for a given reason. Optional `intent` filter
  (`method|background|result_comparison`). Window-function dedup to one
  row per citing paper with `n_contexts` density count; ranked
  `citation_count DESC`. Every response carries a `coverage` block so
  agents can tell "no events" from "no coverage".
- `annotate='relation'` → the old `find_replications`
  (`src/scix/find_replications.py`): forward citations annotated with an
  inferred replication relation. `relation` filter ∈
  `replicates|refutes|qualifies|partial`. Relation inference is a
  DOCUMENTED HEURISTIC substitute for NegBERT (not wired up): hedge-cue
  lexicon + comparison-verb lists, inference order partial → refutes →
  qualifies → replicates, hedged agreement downgraded to `qualifies`, no
  match → `unknown`. Ranked by intent weight desc, year asc. Do not
  present relations as model-classified — they are lexical-pattern
  inferences.
- `limit` convention: default 20 (`DEFAULT_RESULT_LIMIT`), cap 200
  (`MAX_WORKING_SET_BIBCODES`).

### claim_blame (`src/scix/claim_blame.py`, PRD MH-4 in docs/prd/scix_deep_search_v1.md)

Traces `claim_text` back to its chronologically earliest non-retracted
origin over citation contexts (no hard intent filter). Load-bearing
semantics:

- Ranking: `(chronological_priority, intent_weight, semantic_match)`.
  Chronological-earliest is the citation-laundering guard (laundering is a
  temporal problem).
- Intent weights: `result_comparison 1.0, method 0.6, background 0.3`;
  NULL intent → floor 0.3 (`DEFAULT_INTENT_WEIGHT` — unlabeled hops never
  get inflated credit).
- Confidence: `clamp(0.5·intent_weight + 0.3·chronology_score +
0.2·semantic_match)`; chronology_score 1.0 iff origin year strictly
  precedes all other candidates, else 0.5; semantic_match is INDUS cosine
  mapped `(s+1)/2`, substituting 0.5 when no embedding is available (which
  today is common — the query-side embedding loads INDUS locally via
  `scix.embed`, and candidate embeddings are seeded as None).
- Retractions (`papers.correction_events` type='retraction') are excluded
  from origin selection and surfaced in `retraction_warnings`, never
  silently dropped. Errata/EoC do NOT disqualify in v1.
- Optional `scope` (`research_scope.py`) filters candidates and hops
  (year_window, community, venue, ...). Args: `candidate_limit` (20),
  `lineage_limit` (10). Timeout 15s (`SCIX_TIMEOUT_CLAIM_BLAME`).
- Startup self-test invokes `claim_blame` when `SCIX_TEST_DSN` is set
  (`mcp_server.py:812`).

### graph_context (`mcp_handlers/entity.py:_handle_graph_context`)

The serving surface for `paper_metrics`: PageRank/HITS/community
memberships for one bibcode, with a per-signal/per-resolution
`communities` label block. `include_community=true` adds sibling papers
via `explore_community` (default `signal='semantic'`,
`resolution='medium'` for find_gaps parity, bead unmm). Aliases:
`get_paper_metrics`, `explore_community`.

### Timeouts (env-tunable, seconds)

`SCIX_TIMEOUT_TRAVERSE=20`, `SCIX_TIMEOUT_CLAIM_BLAME=15`,
`SCIX_TIMEOUT_FIND_REPLICATIONS=15`, `SCIX_TIMEOUT_FORWARD_CITATIONS=15`,
`SCIX_TIMEOUT_GRAPH_CONTEXT=10`, plus legacy alias entries
(`TOOL_TIMEOUTS`, `mcp_server.py:435`).

## 7. Intent classification (`src/scix/citation_intent.py`)

- Label set: `background | method | result_comparison` (SciCite mapping;
  `result` → `result_comparison`). `VALID_INTENTS` is the contract; both
  classifiers validate every output label.
- Two `IntentClassifier` implementations behind a Protocol:
  `SciBertClassifier` (fine-tuned SciBERT via transformers pipeline,
  batched, module-level pipeline cache, the production path) and
  `LLMClassifier` (Anthropic API fallback, sequential — note the pinned
  default `claude-sonnet-4-20250514` is dated; treat as config, not
  doctrine).
- Population: `update_intents()` / `scripts/backfill_citation_intent.py`
  — batched, resumable (`WHERE intent IS NULL`), transactional per batch,
  audited in `ingest_log` (migration 066 marker row; that migration also
  documents the 056→066 renumbering collision). Hand-validation sample
  export: `--validate-sample N` → `docs/eval/mh1_intent_validation.md`.
- Downstream consumers: traversal annotation (§6), intent weights in
  `claim_blame`/`find_replications`, the `intent` filter in
  `forward_citations`.

## 8. Adjacent experimental limb (do not confuse with prod)

`src/scix/mcp_graph_experiment.py` + `src/scix/graph_experiment/` (bead
vdtd spike): a SEPARATE experimental MCP server that loads a pre-built
igraph snapshot (`SCIX_GRAPH_EXP_SNAPSHOT`) for benchmarking. NOT in the
production tool surface, not under the 15-cap. Parked, not dead —
PROVISIONAL pending Stephanie (discovery Q5): check bead/branch state
before re-landing or deleting anything here.

## 9. Open/candidate items (stated plainly, no oversell)

- Citation Leiden on prod remains UNDONE (the Phase-B OOM). The M3 script
  and the memory work (`47d7a0c`, `732dec0` lineage) are the prepared
  path; success criterion is `verify_communities_populated.py` showing
  non-sentinel `community_id_*` counts plus the ≤10% largest-coarse
  invariant. Any attempt is prod-DB-writing + ADR-adjacent: HALT
  branch-ready, operator sign-off — PROVISIONAL pending Stephanie
  (discovery Q5, conservative gating).
- `store_community_metadata` schema drift (§3.2) — unclaimed fix.
- NegBERT for relation inference — designed future drop-in, not started.
- `claim_blame` has no real-corpus accuracy number yet (the gold set work
  is bead 6ajy; see `scix-eval-and-evidence`). Do not claim provenance
  accuracy in any external artifact until that lands.
- Checked-in `results/community_coverage.json` is an empty-DB artifact;
  regenerate before citing coverage anywhere.

## Provenance and maintenance

Written 2026-07-07 against `bd/0yp5-external-copy-accuracy-audit` @
`452ab86` (scope files verified identical to `origin/main` @ `e59d89d`).
All claims source-read; no DB or runtime verification was performed (this
host's prod-DB and heavy-work gates). Re-verify before trusting:

```bash
git rev-parse --short HEAD && git branch --show-current
git diff --stat HEAD..origin/main -- src/scix/graph_metrics.py src/scix/mcp_handlers/citation.py   # drift since pin
grep -n "ON CONFLICT (community_id, resolution)" src/scix/graph_metrics.py   # §3.2 trap still present?
grep -n "paper_embeddings" src/scix/graph_metrics.py                          # §3.1 trap still present?
grep -n "PRIMARY KEY (signal, resolution, community_id)" ci/scix_test_schema.sql
grep -rn "_NO_COMMUNITY_SENTINEL" src/scix/search.py                          # sentinel model still live?
python3 -c "import json; d=json.load(open('contract/scix_mcp_v1.json')); print(len(d['tools']), sorted(t['name'] for t in d['tools']))"
grep -n "annotate" src/scix/mcp_handlers/citation.py | head -3               # forward_citations enum intact?
ls logs/leiden_recompute/                                                     # has a prod-scale M3 run landed?
pytest --collect-only -q tests/test_graph_metrics.py | tail -1
```

If any of these disagree with this skill, the repo wins — update the skill
under an explicit bead, not silently.
