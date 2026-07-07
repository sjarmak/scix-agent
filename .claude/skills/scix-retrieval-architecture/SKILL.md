---
name: scix-retrieval-architecture
description: >
  How a query becomes ranked results in SciX: the RRF hybrid stack in
  src/scix/search.py — title/abstract BM25 + body BM25 + INDUS dense (Qdrant)
  fused at RRF k=60, optional alias/ontology lanes and a cross-encoder reranker
  (off by default), and the honest eval state (dense 0.38 < bm25 0.43 nDCG@10
  on the 1200q gold set). Load when reading or changing hybrid_search /
  lexical_search / vector_search / rrf_fuse, tuning SCIX_LEXICAL_* /
  SCIX_RERANK_* flags, or checking a change against the ADR-pinned invariants
  (768d, halfvec, no paid-API lane, Qdrant-not-pgvector serving). NOT for
  Qdrant ops (scix-vector-serving-qdrant), the embed/ingest pipeline or s7cy
  fire (scix-embedding-pipeline), running evals (scix-eval-and-evidence), or
  index builds (scix-index-and-storage-discipline).
---

# SciX Retrieval Architecture

The fragile core: how one query string turns into a ranked paper list.
Everything here was verified by source-reading on 2026-07-07 against commit
`452ab86` (branch `bd/0yp5-external-copy-accuracy-audit`; the retrieval files
described here are byte-identical to `origin/main` @ `e59d89d`). Line numbers
drift — re-verify with the commands at the bottom before trusting them.

**Jargon, defined once:**

- **BM25 / lexical lane** — here, PostgreSQL full-text search: a `tsvector`
  match ranked by `ts_rank_cd`. Not literal Okapi BM25; the codebase calls it
  "BM25" throughout.
- **Dense lane** — k-nearest-neighbor search over 768-dimensional INDUS
  embeddings (`nasa-impact/nasa-smd-ibm-st-v2`), served by Qdrant.
- **RRF (Reciprocal Rank Fusion)** — score = Σ 1/(k + rank_in_lane) across
  lanes; papers appearing in several lanes get boosted. `k` damps how much
  top ranks dominate.
- **Reranker** — a cross-encoder model that re-scores the fused top-N
  (query, title+abstract) pairs. Currently disabled by default (see §5).
- **halfvec** — pgvector's float16 vector type; the only approved
  quantization for this corpus.

## 1. When to use this skill / when not

Use it to understand or modify the retrieval path, review a retrieval PR, or
sanity-check a proposed lane/fusion change against the pinned invariants.

| If your question is about                                | Go to                                 |
| -------------------------------------------------------- | ------------------------------------- |
| Qdrant collection config, payloads, canary, RAM          | `scix-vector-serving-qdrant`          |
| Embedding generation, PG→Qdrant sync, the s7cy live fire | `scix-embedding-pipeline`             |
| Gold sets, nDCG/MRR/recall, running a sweep safely       | `scix-eval-and-evidence`              |
| Index DDL, "don't trust a new index" rules, disk crisis  | `scix-index-and-storage-discipline`   |
| MCP tool schemas, 15-tool cap, guards, telemetry         | `scix-mcp-tool-surface`               |
| Using search tools as a research agent                   | `scix-mcp` (query-side, partly stale) |
| Whether a change needs an ADR / sign-off                 | `scix-change-control`                 |

## 2. The stack, end to end

All live retrieval logic is in `src/scix/search.py` (~4850 lines). The MCP
entry point is `_handle_search` in `src/scix/mcp_handlers/search.py`.

```
query text
  │
  ├─ mode="keyword"  → lexical_search() only
  ├─ mode="semantic" → vector_search() only (errors cleanly if no dense lane)
  └─ mode="hybrid" (default) → hybrid_search():
        1. lexical_search()        — title/abstract BM25        (lane 1)
        2. [alias lanes]           — OFF by default, ≤3 extra lexical lanes
        3. lexical_search_body()   — body BM25, include_body=True (lane 2)
        4. vector_search()         — INDUS dense via Qdrant     (lane 3)
        5. rrf_fuse(k=60, top_n)   — fuse all present lanes
        6. [reranker]              — OFF by default (see §5)
```

Every lane returns a `SearchResult` (frozen dataclass: `papers`, `total`,
`timing_ms`, `metadata`). Timing metadata is a hard contract — keep it when
extending.

Key constants (all in `src/scix/search.py` unless noted):

| Constant                         | Value   | Where                     | Meaning                                                     |
| -------------------------------- | ------- | ------------------------- | ----------------------------------------------------------- |
| `RRF_K`                          | 60      | search.py:34              | default RRF k, also `hybrid_search(rrf_k=)` default         |
| `vector_limit` / `lexical_limit` | 60 / 60 | `hybrid_search` signature | per-lane candidate depth fed to fusion                      |
| `top_n`                          | 20      | `hybrid_search` signature | fused results returned (MCP passes its `limit`, default 10) |
| `_LEXICAL_POOL_DEFAULT`          | 30000   | search.py:267             | candidate-pool cap for the title/abstract lane              |
| `_LEXICAL_RANK_FLAG_DEFAULT`     | 32      | search.py:317             | `ts_rank_cd` normalization (rank/(rank+1), no length norm)  |
| `_MAX_ALIAS_LEXICAL_LANES`       | 3       | search.py:898             | cap on alias-expansion extra lanes                          |
| `_RERANK_TOP_K_CAP`              | 20      | mcp_runtime.py:674        | reranker bypassed when limit > 20                           |
| `SELECTIVITY_THRESHOLD`          | 0.01    | search.py:768             | filter-first routing cutoff (pg path only, legacy)          |

## 3. The three live lanes

### Lane 1 — title/abstract BM25: `lexical_search()` (search.py:353)

- Matches `papers.tsv` (title+abstract tsvector) with
  `plainto_tsquery('scix_english', query)` — AND semantics; `tsquery_mode=
"plain_or"` exists for eval attribution arms only (`_TSQUERY_MODES` =
  `{plain_and, plain_or}`; `_TS_CONFIG_WHITELIST` = `{scix_english, english}`).
- **Candidate-pool cap** (bead 3t37): a CTE caps matched candidates at
  `SCIX_LEXICAL_POOL` rows (default 30000) before `ts_rank_cd` runs,
  otherwise common terms time out. Trade-off: above the cap, only the first
  ~POOL rows in bitmap-heap-scan order get ranked (biased toward
  earlier-ingested papers). Eval harnesses set `SCIX_LEXICAL_POOL=INF` (or
  `all`/`none`) for the full match set — never do that on the live server.
  The 30000 default is evidence-backed (comment at search.py:260): 5000
  recovered only ~15% of the uncapped top-20; 30000 is the knee.
- **Rank flag** `SCIX_LEXICAL_RANK_FLAG` (default 32): does NOT
  length-normalize; a short dense doc outranks a long one on the same term
  (verified, bead q9k5). Length-aware bits (1, 2, 16) OR with 32; only bits
  0..63 accepted. Both env vars are re-read on every call — tune a running
  server without restart.

### Lane 2 — body BM25: `lexical_search_body()` (search.py:463)

- Matches full-text bodies via the GIN **expression** index from migration
  039: `to_tsvector('english', p.body)`, restricted to
  `body IS NOT NULL AND length(body) <= 1_048_575` (must mirror the partial
  index predicate or the planner seq-scans).
- **Coverage: 46% of the corpus** (14.9M of 32.4M papers have body text —
  README, dated 2026-07-07). A paper without a body simply can't appear in
  this lane.
- **Ranking quirk (intentional):** the match runs against the body, but
  `ts_rank_cd` scores against `papers.tsv` (title+abstract) — ranking 65KB
  bodies costs ~400s/query. Ordering inside this lane is approximate; RRF
  absorbs it.
- Enabled by `hybrid_search(include_body=True)` — the default, and the MCP
  handler does not override it. (The docstring calls it "a 4th RRF signal" —
  stale phrasing from when the removed OpenAI dense lane still existed; with
  today's three lanes it is the 2nd or 3rd. Committed comment drift, not a
  behavior bug.)

### Lane 3 — INDUS dense: `vector_search()` (search.py:619)

- Routing: `_qdrant_dense_gated(model_name)` (search.py:59) returns True when
  `model_name` has a Qdrant collection (`_QDRANT_DENSE_COLLECTIONS =
{"indus": "scix_indus_v2_papers_s1"}`) AND `QDRANT_URL` is set → all kNN
  goes to `_vector_search_qdrant()` (search.py:527). This is the production
  path per ADR-013 (`docs/ADR/013_dense_lane_qdrant.md`).
- Qdrant client: REST only (`prefer_grpc=False`) — gRPC fails deserializing
  qdrant 1.18.2 responses (verified 2026-06-11, bead pkcd).
- Filters: the v1 collection carries only a `bibcode` payload, so SQL filters
  are applied as a **post-filter on the PG metadata join** with over-fetch
  (10× limit, capped at 500). Points missing a bibcode payload are skipped
  with a warning, not fatal.
- `SCIX_QDRANT_EXACT=1` forces exact (non-indexed) kNN — eval-only control
  for isolating HNSW approximation loss; far too slow for serving.
- Query embedding is computed by the caller: the MCP handler loads INDUS
  locally (`load_model`/`embed_batch` from `scix.embed`,
  `SCIX_EMBED_DEVICE`, default cpu) and passes the 768-d vector in. If
  transformers/torch are missing, hybrid degrades to lexical-only with a
  logged warning.

**Legacy pgvector path (below the gate in `vector_search` and
`_filter_first_vector_search`): treat as dead code at committed HEAD.** It
queries the `paper_embeddings` table, which was dropped from prod
out-of-process ~2026-06-29/30 (bead s7cy). PROVISIONAL pending Stephanie
(discovery Q2): a direct-to-Qdrant remediation exists but is uncommitted
in-flight work — do not teach or extend the pg dense path, and do not assume
the working tree's `embed.py`/`qdrant_dense.py` are canon. Details:
`scix-embedding-pipeline`. Related: `SCIX_USE_HALFVEC` (search.py:44) only
affects this dead pg path for indus; it is not a live serving knob.

**Degradation behavior worth knowing:** if `QDRANT_URL` is unset,
`_hnsw_index_exists()` (`src/scix/mcp_runtime.py:73`) falls through to a
`pg_indexes` check that now finds nothing → `mode="semantic"` returns a
structured `vector_index_unavailable` error, and `mode="hybrid"` silently
runs BM25-only. A "working" hybrid search is not proof the dense lane ran —
check `timing_ms["vector_ms"] > 0`.

## 4. Fusion: `rrf_fuse()` (search.py:731) and the optional lanes

```python
score[bibcode] = sum(1.0 / (k + rank) for each lane containing it)  # rank is 1-based
```

k=60 everywhere by default. The 1200q sweep found nDCG@10 nearly flat across
k ∈ {10, 30, 60, 100} (0.4767–0.4786), so k is not a sensitive knob; 60 won
and matches the literature default.

Lanes that are **off by default** (and NOT reachable through the MCP `search`
tool at all — the handler never passes these flags; they are library/eval-only):

| Flag                         | What it adds                                                                                                                             | Guards                                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_alias_expansion`     | ≤3 extra `lexical_search` lanes, one per matched entity canonical name (`scix.alias_expansion.expand_query`)                             | Each lane runs in a savepoint; a per-lane `statement_timeout` drops just that lane and records it in `metadata["dropped_lanes"]` (beads wzqz, uq28) |
| `enable_ontology_parser`     | Lifts parsed `entity_types` / `properties_filters` into `SearchFilters` before all lanes (`scix.ontology_query_parser.parse_query`)      | Entity-id lookups capped at 200                                                                                                                     |
| (MCP arg) `community_expand` | **Replaces** hybrid retrieval with `community_expand_search()` — it does not RRF-fuse (PRD `docs/prd/prd_community_expand_search.md` §4) | Super-hub seeds >50k papers get a structured `seed_too_broad` refusal                                                                               |

The body lane is likewise savepoint-wrapped: a body-lane timeout drops the
lane and reports `dropped_lanes=["body_bm25"]` instead of poisoning the outer
transaction. **When reading eval numbers, always check `dropped_lanes`** — a
degraded fusion is not a full-signal fusion.

The MCP-side unscoped-broad-query guard (no filters AND ≥3 tokens or ≥30
chars → structured `unscoped_broad_query` error, bead uerc) fires before any
lane runs; that surface belongs to `scix-mcp-tool-surface`.

## 5. The reranker is OFF, and the numbers say why

`hybrid_search(reranker=...)` accepts any callable; the class is
`CrossEncoderReranker` (search.py:1569), scoring (query, title+abstract)
pairs, 512-token cap (the INDUS ranker's RoBERTa asserts on 514).

Production gating (`src/scix/mcp_runtime.py:653-674`,
`mcp_handlers/search.py:267-278`):

- `SCIX_RERANK_DEFAULT_MODEL` defaults to `'off'` → the factory returns
  `None` and no model is ever instantiated. Allowed values: `off`, `minilm`,
  `bge-large`, `indus-ranker`.
- Even when a model is configured, reranking only applies when the request
  `limit ≤ 20` (`_RERANK_TOP_K_CAP`, PRD `prd_cross_encoder_reranker_local.md` M3).

Every evaluated candidate **regressed** retrieval quality (nDCG@10, from the
committed comments and `results/retrieval_eval_50q_rerank_indus.md`):

| Model                              | Baseline → reranked | Δ       | p     | Verdict                                     |
| ---------------------------------- | ------------------- | ------- | ----- | ------------------------------------------- |
| ms-marco-MiniLM-L-12-v2            | 0.3255 → 0.2802     | −0.0453 | 0.042 | fail (M1 ablation, commit 06a6cc3)          |
| BAAI/bge-reranker-large            | 0.3255 → 0.2699     | −0.0556 | 0.026 | fail (M1 ablation)                          |
| nasa-smd-ibm-ranker (domain-tuned) | 0.2242 → 0.1843     | −0.0400 | 0.074 | NO-GO (bead 4skc, re-baselined Qdrant pool) |

No reranker has cleared the rollout gate. Do not flip one on in production
without a fresh eval that clears it; flipping it on for experimentation is
what the env var is for.

## 6. The honest retrieval state (dated 2026-07-07)

Source: `results/fusion_sweep_1200q.md` (gold set `eval/recall_gold_v1.jsonl`,
1200 queries; lanes: INDUS dense via Qdrant vs combined PG BM25; READ-ONLY
sweep). Headline rows:

| Config                                 | nDCG@10    | MRR@10 | Recall@50 | Δ vs dense_only |
| -------------------------------------- | ---------- | ------ | --------- | --------------- |
| **naive_rrf(k=60)** (production shape) | **0.4786** | 0.8926 | 0.4572    | +0.0983         |
| bm25_only                              | 0.4291     | 0.8152 | 0.4258    | +0.0488         |
| dense_only                             | 0.3803     | 0.7207 | 0.3765    | —               |

**Read it plainly: the INDUS dense lane alone underperforms BM25 alone
(0.3803 < 0.4291).** Hybrid RRF beats both, but the sweep's own verdict is
that the +0.098 lift over dense-alone "partly reflects outrunning a
non-dominant dense lane rather than a clean top-rank fusion gain." The
earlier premise of a dominant dense lane (bead dfba, ~0.864 on the 50q set)
did **not** reproduce on the 1200q set; the 50q sweep
(`results/fusion_sweep_v1.md`) also puts bm25_only on top at much lower
absolute numbers — the two gold sets measure very different regimes (see
`scix-eval-and-evidence`). Canonical home for the publication-facing
interpretation of these numbers (what may/may not be claimed) is
`scix-research-frontier` §1; this table is the retrieval stack's own eval
state, restated here for that purpose only.

PROVISIONAL pending Stephanie (discovery Q3): report the hybrid number with
that caveat and with lane provenance; never state INDUS/dense superiority as
fact. A dense lane that beats BM25 is an **open frontier**
(`scix-research-frontier` owns the campaign), not a settled gate or a
precondition.

Reproducing the sweep is heavy (full INDUS query encoding + 1200 live
queries). It is read-only against the corpus but **do not run casually** —
operator RAM window + batch wrapper required (see
`scix-memory-and-batch-discipline`):

```bash
# HEAVY — do not run casually; needs scix-batch + operator window
scix-batch python scripts/fusion_sweep.py --queries eval/recall_gold_v1.jsonl

# Cheap wiring check (no DB, no model) — safe anywhere:
python scripts/fusion_sweep.py --dry-run
```

## 7. Load-bearing invariants (change only via ADR — see scix-change-control)

Each is a decision with an incident or measurement behind it, not a
preference. All are restated in the repo `CLAUDE.md` "Don't" section.

| Invariant                                                                                                                                                                                                               | Rationale / incident                                                                                                                                                                                    | Source                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Dense ANN serves from **Qdrant**, never pgvector/pgvectorscale                                                                                                                                                          | The DiskANN catastrophe: dimensionless `vector` column → expression index pgvectorscale cannot scan (`assertion failed: attnum > 0`); 56h build lost, ~2 weeks lexical-only (bead 12rp, closed/settled) | `docs/ADR/013_dense_lane_qdrant.md`                                                                             |
| **768d locked**                                                                                                                                                                                                         | pg block-size/TOAST limits; matches INDUS native output                                                                                                                                                 | CLAUDE.md pin                                                                                                   |
| **halfvec (float16) is the only safe quantization; binary quantization banned**                                                                                                                                         | >40% nDCG@10 loss on scientific retrieval; BQ acceptable only as a first-pass filter, never storage                                                                                                     | CLAUDE.md pin; `docs/prd/prd_body_chunk_embeddings.md`, `docs/prd/qdrant_nas_migration.md`                      |
| **No paid-API embedding lane** (e.g. text-embedding-3-large)                                                                                                                                                            | Any second dense lane must be local-weight + ADR-approved; the OpenAI lane was removed from live code                                                                                                   | CLAUDE.md pin (`feedback_no_paid_apis` — project memory, not an in-repo file); `scripts/fusion_sweep.py` header |
| No new index trusted until one query returns from it: ≤50k scratch build + forced-index-scan smoke test; benchmark DDL byte-identical to prod; never drop a serving index before its replacement is validated scannable | The four rules ADR-013 §"Validation rules" bought with the 12rp failure                                                                                                                                 | `docs/ADR/013_dense_lane_qdrant.md:89-100`; runbook detail in `scix-index-and-storage-discipline`               |
| Qdrant/PG data dirs never on NAS                                                                                                                                                                                        | NFS unsafe for live-write workloads                                                                                                                                                                     | CLAUDE.md pin; `docs/prd/qdrant_nas_migration.md`                                                               |

Related but not law: ADR-014 (Qdrant sparse BM25 lexical lane) is
**Proposed/parked-on-disk** — pilot showed +0.06–0.10 fused nDCG@10 but with
an AND-vs-OR confound; it unblocks after storage relief. Check bead state
before relanding; do not treat the PG lexical lanes as permanent or the
sparse lane as dead.

## 8. Change checklist for this area

Before touching `search.py` retrieval paths:

1. Does the change touch an invariant in §7? → ADR required; HALT for
   sign-off (PROVISIONAL pending Stephanie, discovery Q5: treat all pinned
   axes as HALT-branch-ready).
2. Does it alter lane behavior or fusion? → re-run the relevant gold-set eval
   (via `scix-eval-and-evidence`) and report per-bucket numbers with lane
   provenance; a green test suite is not retrieval evidence.
3. Preserve the `SearchResult.timing_ms` contract and `dropped_lanes`
   surfacing.
4. New lane SQL touching prod-size tables: respect the candidate-pool-cap
   pattern (§3, lane 1) and savepoint-wrap recoverable lanes.
5. Tests ship in the same commit. Existing anchors: `tests/test_search.py`,
   `tests/test_fusion_sweep.py`, `tests/test_mcp_search.py` (write tests need
   `SCIX_TEST_DSN` — see `scix-db-safety-and-telemetry`; without it they
   skip silently).

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`,
HEAD `452ab86` (not main; `src/scix/search.py`, `src/scix/mcp_handlers/search.py`,
and `src/scix/mcp_runtime.py` verified identical to `origin/main` @ `e59d89d`).
All numbers in §5–§6 are quoted from committed results/comments, not re-run.

Re-verify (all read-only):

```bash
git rev-parse --short HEAD && git branch --show-current
grep -n "RRF_K = " src/scix/search.py                          # expect 60
grep -n "_QDRANT_DENSE_COLLECTIONS" src/scix/search.py          # expect scix_indus_v2_papers_s1
grep -n "_LEXICAL_POOL_DEFAULT\|_LEXICAL_RANK_FLAG_DEFAULT" src/scix/search.py   # 30000 / 32
grep -n "_MAX_ALIAS_LEXICAL_LANES" src/scix/search.py           # expect 3
grep -n "_RERANK_TOP_K_CAP\|SCIX_RERANK_DEFAULT_MODEL" src/scix/mcp_runtime.py   # 20 / default 'off'
grep -n "include_body: bool = True\|enable_alias_expansion: bool = False" src/scix/search.py
grep -n "dense_only\|bm25_only\|naive_rrf(k=60)" results/fusion_sweep_1200q.md   # 0.3803 / 0.4291 / 0.4786
grep -n "46%" README.md                                         # body coverage still 46%?
grep -rn "paper_embeddings" src/scix/search.py | head -5        # pg dense path still dead-gated?
head -3 docs/ADR/014_qdrant_sparse_lexical_lane.md              # sparse lane still Proposed/parked?
```

If `RRF_K`, the collection name, the reranker default, or the fusion-sweep
verdict changed, this skill is stale — fix it in the same change, and check
whether the s7cy remediation (uncommitted at authoring time) has landed,
which would rewrite the "legacy pgvector path" paragraph in §3.
