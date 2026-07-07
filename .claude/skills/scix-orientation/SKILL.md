---
name: scix-orientation
description: >
  First-load orientation map for the SciX Experiments repo: what SciX is
  (agent-navigable knowledge layer over the NASA ADS/SciX corpus, NOT a
  benchmark), corpus scale, the PostgreSQL + Qdrant dual store, the four
  surfaces (hybrid retrieval, citation graph, entity graph, provenance
  tools), source layout, the fastest reading route into the codebase, the
  viz dashboard layer, and the doc-drift warning (which docs to trust and
  which are stale). Load this when you are new to the repo, when you need
  to locate where a subsystem lives, or before deciding which sibling
  skill to load. NOT for running heavy jobs (use
  scix-memory-and-batch-discipline), NOT for touching any database (use
  scix-db-safety-and-telemetry FIRST), NOT for retrieval internals (use
  scix-retrieval-architecture), NOT for querying the corpus as a research
  user (use the existing scix-mcp skill), NOT for making gated changes
  (use scix-change-control).
---

# SciX Orientation — the map before you touch anything

Date-stamped 2026-07-07. All paths and line numbers verified against
committed HEAD `452ab86` on branch `bd/0yp5-external-copy-accuracy-audit`
(NOT `main`; see Provenance at the bottom).

Read this skill fully before your first edit. It answers three questions:
what is this system, where does everything live, and which documents can
you trust.

## 1. What SciX is (and is not)

SciX Experiments turns the full NASA ADS / SciX scholarly corpus into an
**agent-navigable knowledge layer**: instead of returning ranked lists, it
exposes the structural topology of science — hybrid retrieval, the
citation graph, research communities, and a cross-discipline entity graph
— through a **15-tool MCP server** so AI agents can navigate the
literature programmatically.

Jargon, defined once:

| Term                | Meaning here                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| ADS / SciX          | NASA's Astrophysics Data System and its multi-discipline successor SciX; the upstream corpus              |
| MCP                 | Model Context Protocol — the tool interface agents call (`src/scix/mcp_server.py`)                        |
| INDUS               | NASA's domain-specific embedding model (`nasa-impact` family, 768-dim) powering the dense lane            |
| RRF                 | Reciprocal Rank Fusion — how lexical and dense result lists merge (`RRF_K = 60`, `src/scix/search.py:34`) |
| BM25 / lexical lane | PostgreSQL tsvector + `ts_rank_cd` keyword ranking                                                        |
| Dense lane          | k-nearest-neighbor search over INDUS embeddings, served by Qdrant                                         |
| bead                | A work item in the Gas City queue (`.beads/`); the project's real backlog and change log                  |
| ADR                 | Architecture Decision Record, `docs/ADR/` — the binding design documents                                  |

What it is NOT:

- **Not a benchmark.** It is a live retrieval system with a publication
  track (ADASS paper, `docs/paper_outline.md`).
- **Not a toy corpus.** The default database is the production 32M-paper
  store. `SCIX_DSN` unset resolves to `dbname=scix` = PRODUCTION
  (`src/scix/db.py:16`). Read scix-db-safety-and-telemetry before ANY
  database interaction.
- **Not safe for casual heavy jobs.** This installation co-hosts a
  process supervisor; unwrapped multi-GB work gets OOM-killed and takes
  the fleet down with it. Read scix-memory-and-batch-discipline before
  running anything that loads models or scans tables.

## 2. Corpus scale

Documented values (README.md stats table; these are claims of record, not
re-measurable read-only — verifying them requires a production DB query,
which you must not run casually):

| Dimension               | Value (documented 2026-07-07)                        |
| ----------------------- | ---------------------------------------------------- |
| Papers                  | 32.4M (1800–2026)                                    |
| With abstracts          | 23.3M (72%)                                          |
| With full text ingested | 14.9M (46%)                                          |
| Citation edges          | 299.3M                                               |
| INDUS embeddings        | 768-dim, full corpus (but see the s7cy caveat in §6) |
| Entities                | ~9M across 13 vocabularies (README.md:187)           |
| Paper–entity links      | 57.7M (`document_entities`)                          |

## 3. The dual store: PostgreSQL + Qdrant

**Architecture headline: two stateful systems, not one.**

| Store                                          | Holds                                                                                              | Notes                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| PostgreSQL 16 + pgvector 0.8.2 (`dbname=scix`) | papers, metadata, citations, entities, communities, sessions, BOTH BM25 lexical lanes, `query_log` | Authoritative for everything except dense kNN                              |
| Qdrant 1.17/1.18, local NVMe                   | The INDUS dense lane — collection `scix_indus_v2_papers_s1` (`src/scix/search.py:51`)              | REST-only client; gated by `QDRANT_URL` (`src/scix/search.py:46`); ADR-013 |

The dense lane moved from pgvector to Qdrant on 2026-06-11 (ADR-013,
Accepted) after a pgvectorscale/DiskANN failure cost ~2 weeks of dense
serving. The follow-on storage line: ADR-014 (Qdrant sparse lexical lane,
Proposed/parked), ADR-015 (drop the PG INDUS embedding footprint,
Proposed), ADR-016 (cold-text tier on NAS, Accepted 2026-06-29). Details
and the failure history live in scix-vector-serving-qdrant and
scix-index-and-storage-discipline.

## 4. The four surfaces

1. **Hybrid retrieval** (`src/scix/search.py`, 178K — the largest module):
   RRF fusion (`k=60`) of a title/abstract tsvector BM25 lane, a body BM25
   lane (~46% coverage), and the INDUS dense lane via Qdrant. Optional
   entity-alias lexical lanes are off by default; an optional post-fusion
   reranker exists. The retired OpenAI `text-embedding-3-large` lane is
   REMOVED from live code — any doc listing it is stale. Measured state as
   of the 1200q gold set: `dense_only` nDCG@10 (0.3803) is BELOW
   `bm25_only` (0.4291); RRF still lifts over dense-alone. Internals →
   scix-retrieval-architecture; the credibility question →
   scix-research-frontier.
2. **Citation graph** (`src/scix/graph_metrics.py`): PageRank, HITS,
   Leiden communities over 299.3M edges; `citation_traverse`,
   `graph_context`, `citation_similarity` tools. → scix-citation-graph.
3. **Entity graph** (`src/scix/extract.py`, `src/scix/jit/`): 13 entity
   vocabularies, GLiNER NER, 57.7M paper–entity links, per-bucket
   precision bands (aggregate ≥80% precision is NOT met — use the
   lower bound when the bucket is unknown). → scix-entity-ner-system.
4. **Provenance tools**: `claim_blame` (trace a claim to its earliest
   non-retracted origin, `src/scix/claim_blame.py`) and replication
   finding (`src/scix/find_replications.py`, surfaced through the
   `forward_citations` tool rather than as its own tool). These are the
   paper's differentiation claim vs Elicit/SciSpace/paper-qa.

The 15 visible MCP tools (from `contract/scix_mcp_v1.json`, verified):
`search`, `concept_search`, `get_paper`, `read_paper`, `lit_review`,
`citation_traverse`, `citation_similarity`, `forward_citations`,
`graph_context`, `entity`, `facet_counts`, `find_gaps`,
`temporal_evolution`, `claim_blame`, `synthesize_findings`. Five more are
hidden by default (`chunk_search`, `section_retrieval`,
`read_paper_claims`, `find_claims`, `claim_search` —
`src/scix/mcp_server.py:495`). The visible cap is 15, enforced at import
time — exceeding it raises `RuntimeError` before the server boots
(`src/scix/mcp_server.py:625-638`). Tool-surface changes →
scix-mcp-tool-surface.

## 5. Source layout

| Path                           | What lives there                                                                                                                                                                                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/scix/`                    | The package. Key modules: `search.py` (retrieval core), `mcp_server.py` + `mcp_tool_specs.py` + `mcp_runtime.py` + `mcp_handlers/` (tool surface), `db.py` (DSN + production guard), `embed.py` (embedding pipeline — see §6 caveat), `graph_metrics.py`, `extract.py`, `claim_blame.py`, `session.py` |
| `src/scix/viz/`                | Viz FastAPI layer: `server.py`, `api.py`, `trace_stream.py` (see §8)                                                                                                                                                                                                                                   |
| `src/scix/eval/`               | Retrieval evaluation framework                                                                                                                                                                                                                                                                         |
| `scripts/`                     | ~201 CLI scripts (`.py`+`.sh`). Many are heavy ingest/embed jobs — DO NOT run casually; they target production by default and must be wrapped (→ scix-memory-and-batch-discipline)                                                                                                                     |
| `migrations/`                  | 71 committed numbered SQL files (`001`–`071`), append-only, NO auto-runner — applied by hand. `072` exists only uncommitted in the working tree (§6)                                                                                                                                                   |
| `contract/scix_mcp_v1.json`    | The frozen MCP contract; CI fails if it drifts from `build_contract()`                                                                                                                                                                                                                                 |
| `tests/`                       | ~264 `test_*.py` files, pytest. Env-dependent suites skip silently — → scix-build-test-ci                                                                                                                                                                                                              |
| `ci/scix_test_schema.sql`      | The consolidated schema snapshot CI loads                                                                                                                                                                                                                                                              |
| `docs/ADR/`                    | ADR-006–016. The binding decisions                                                                                                                                                                                                                                                                     |
| `docs/prd/`, `docs/premortem/` | ~30 PRDs and premortems                                                                                                                                                                                                                                                                                |
| `docs/paper_outline.md`        | The ADASS paper — the thesis document                                                                                                                                                                                                                                                                  |
| `eval/`                        | Gold sets: `retrieval_50q.jsonl`, `recall_gold_v1.jsonl` (1200q), `claim_extraction_gold_standard.jsonl`                                                                                                                                                                                               |
| `results/`, `reports/`         | Campaign outputs (fusion sweeps, eval reports, audits)                                                                                                                                                                                                                                                 |
| `web/viz/`                     | Static dashboard bundles (§8)                                                                                                                                                                                                                                                                          |
| `deploy/`                      | Docker/k8s manifests for the DECOMMISSIONED public deployment (intentionally retired 2026-06-12, per CLAUDE.local.md — not an outage; do not resurrect)                                                                                                                                                |
| `.beads/`                      | Live work queue (Gas City). Do not run `bd dolt start/stop/status`                                                                                                                                                                                                                                     |
| `.gc-reports/`                 | Weekly deep-audits — the best archaeology source                                                                                                                                                                                                                                                       |

Ignore as noise: stale worktree/cache dirs at the root (`.codegraph/`,
`scix-sankey-work/`, `.qdrant_*_storage/`, `*.log`, `out.txt`).

## 6. Working-tree caveat: the uncommitted s7cy fix

PROVISIONAL pending Stephanie (discovery Q2). Teach committed reality;
treat the fix as in-flight, canonize nothing.

- The PG `paper_embeddings` table was dropped out-of-process
  (~2026-06-29/30, bead s7cy, open). Committed HEAD `src/scix/embed.py`
  still inserts into that dropped table (verify: `git show
HEAD:src/scix/embed.py | grep paper_embeddings`) — **the committed
  embedding pipeline cannot run**, and `scripts/embed_fast.py` is in the
  same state.
- A direct-to-Qdrant remediation sits UNCOMMITTED in the working tree:
  `src/scix/qdrant_dense.py`, `migrations/072_indus_qdrant_synced.sql`,
  modified `src/scix/embed.py`/`scripts/daily_sync.sh`, plus new tests.
  It is proposed-not-landed; do not build on it or describe it as the
  standard path until it lands.
- Consequences (from bead s7cy / Phase 1 discovery, 2026-07-07; not
  re-verifiable read-only): `daily_sync.sh` aborts partway, and ~83K new
  papers have no dense vector in Qdrant.

Full story and the payload-preservation hazard → scix-embedding-pipeline.

## 7. Doc-drift: what to trust

**Trust order: ADRs → code → beads → everything else.** Several documents
of record describe a pre-Qdrant, pre-consolidation world (as of
2026-07-07):

| Document                           | Stale claim (verified in place)                                                            | Reality                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `README.md:19`                     | "Single PostgreSQL 16 instance … No separate search engine"                                | Qdrant is a second stateful store (ADR-013)                                         |
| `README.md` layout section         | "migrations 001..054", "114 scripts", "153 test files"                                     | 71 committed migrations, ~201 scripts, ~264 test files                              |
| `CHANGELOG.md`                     | Latest entry v0.1.0 (2026-04-20): "13 tools", "dense (pgvector HNSW)"                      | 15 tools; dense lane on Qdrant; ~2.5 months of unreleased change                    |
| `AGENTS.md` (= `CLAUDE.md`) header | "over PostgreSQL 16 + pgvector"                                                            | Dual store                                                                          |
| `AGENTS.md:71,86` compass table    | Points at `docs/conventions/*.md`                                                          | `docs/conventions/` DOES NOT EXIST                                                  |
| `.claude/skills/scix-mcp/SKILL.md` | Lists `text-embedding-3-large` as an RRF signal; gives a `trycloudflare` connection config | OpenAI lane removed; public deployment decommissioned 2026-06-12 (local stdio only) |

Rules that follow:

1. When README/CHANGELOG and an ADR disagree, the ADR wins. When an ADR
   and the code disagree, read the ADR status line (Proposed vs Accepted)
   and check the bead before concluding anything.
2. Do NOT silently fix stale docs while doing other work — doc
   corrections go through an explicit bead (PROVISIONAL pending
   Stephanie, discovery Q5). The `scix-mcp` skill is read-only context
   for this library: it is the query-side guide; this library is
   maintainer-side.
3. Git history is shallow: the earliest reachable commit is 2026-06-11
   (history re-init); 706 commits to 2026-06-30. Pre-June decisions
   survive only in ADR/PRD/bead prose, never in commits. Archaeology =
   ADRs + PRDs + `.gc-reports/` audits + the bead store, not `git log`.

## 8. Viz layer (secondary surface)

A FastAPI app (`src/scix/viz/server.py`) serves static dashboards from
`web/viz/` (CDN-loaded d3/deck.gl, no npm build) plus a small JSON API
(`src/scix/viz/api.py`, ~1,275 lines) and a live agent-trace SSE stream
(`src/scix/viz/trace_stream.py`).

Shipped dashboards (tags from `web/viz/index.html`): V2 temporal
community Sankey, V3 UMAP embedding browser (100k INDUS points), V4 live
agent-trace overlay, V5 citation heatmap, V6 citation ego network, V11
full-text section coverage, V12 entity source provenance. Further
dashboard work is tracked under the viz epic (bead `xoas`, "V8–V12", per
the 2026-07-07 discovery pass — bead state not re-verifiable read-only).
Operator/presenter guide: `docs/viz/DEMO.md` (covers V2–V4).

Test-coverage note: the 2026-06-22 deep-audit flagged `viz/api.py` as
having no test file. Since then, tests importing it exist
(`tests/test_umap_frontend.py`, `test_viz_ego.py`,
`test_viz_demo_search.py`, `test_viz_demo_composite.py`) using FastAPI
dependency overrides — coverage is partial, they self-skip when `fastapi`
is not installed, and none exercise the live-DB paths. Treat `api.py` as
lightly tested, not untested.

The viz layer reads production data through the same DSN machinery as
everything else — the DB-safety rules apply unchanged.

## 9. Fastest reading route (in order)

1. `README.md` — the intent and the stats table (with §7's grain of salt).
2. `docs/ADR/013_dense_lane_qdrant.md` → `014` → `015` → `016` — the
   storage/serving story of the last month, including why Qdrant exists
   and what is parked vs accepted.
3. `src/scix/search.py` — skim the module docstring and the lane
   functions (`lexical_search`, the body lane, `vector_search`, the RRF
   fusion). It is 178K; do not read linearly.
4. `src/scix/mcp_server.py` — the tool registry, the hidden-tool set
   (`:495`), the 15-cap guard (`:625`), and `startup_self_test`.
5. `docs/paper_outline.md` — the ADASS thesis; what the system claims to
   be for.

After that route, load the sibling skill for whatever you are about to
touch.

## 10. When NOT to use this skill (sibling routing)

| You are about to…                                        | Load instead                                             |
| -------------------------------------------------------- | -------------------------------------------------------- |
| Run anything heavy (embed, ingest, graph job, GPU work)  | scix-memory-and-batch-discipline                         |
| Touch ANY database, run tests that write, read telemetry | scix-db-safety-and-telemetry                             |
| Change retrieval/vector/storage/tool-surface behavior    | scix-change-control (gates first), then the domain skill |
| Understand/modify RRF, lanes, ranking                    | scix-retrieval-architecture                              |
| Work on Qdrant serving or collections                    | scix-vector-serving-qdrant                               |
| Work on embeddings/ingest sync                           | scix-embedding-pipeline                                  |
| Add/change indexes, move data between DS and NAS         | scix-index-and-storage-discipline                        |
| Graph analytics, communities, citation tools             | scix-citation-graph                                      |
| NER, entity linking, precision bands                     | scix-entity-ner-system                                   |
| Add/rename/hide an MCP tool, contract regen              | scix-mcp-tool-surface                                    |
| Set up the env, get CI green, understand test skips      | scix-build-test-ci                                       |
| Gold sets, metrics, what counts as evidence              | scix-eval-and-evidence                                   |
| The paper's open questions, dense-lane integrity         | scix-research-frontier                                   |
| Query the corpus as a research user via MCP tools        | scix-mcp (existing, query-side)                          |

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`
@ `452ab86` (NOT main — `main` was at `56cdab9`, 2 commits not on this
branch; this branch 28 ahead). All file/line claims verified by
source-reading only; no scripts, containers, or databases were run.
Corpus-scale numbers and bead states are documented values, not
re-measured.

Re-verify (all read-only):

```bash
git branch --show-current && git rev-parse --short HEAD   # provenance pin
grep -n "RRF_K = " src/scix/search.py                     # RRF constant (expect :34, 60)
grep -n "VISIBLE_TOOL_CAP" src/scix/mcp_server.py          # 15-cap guard (expect :625)
grep -c '"name"' contract/scix_mcp_v1.json                 # visible tools (expect 15)
grep -n "_QDRANT_DENSE_COLLECTIONS" src/scix/search.py     # collection name (expect :51)
grep -n 'SCIX_DSN' src/scix/db.py                          # prod default DSN (expect :16)
git ls-tree HEAD migrations/ | wc -l                       # committed migrations (expect 71)
ls docs/conventions 2>&1                                   # expect: No such file or directory (drift still live)
git show HEAD:src/scix/embed.py | grep -c paper_embeddings # >0 = committed pipeline still targets dropped table (s7cy unlanded)
git status --porcelain | grep -c qdrant_dense              # >0 = s7cy fix still uncommitted
grep -o 'V[0-9]*</span>' web/viz/index.html                # shipped dashboard tags
```
