# PRD: INDUS Integration & MCP Tool Consolidation

## Problem Statement

SciX exposes 28-30 MCP tools to LLM agents navigating 32M scientific papers. Research (A-RAG, ToolLLM) shows agent tool selection accuracy degrades beyond ~15 tools. The current tool surface has confusion pairs (e.g., `semantic_search` vs `concept_search` vs `entity_search`), implementation-leaking descriptions, and session management tools that belong in server state rather than the agent's action space.

Simultaneously, the INDUS HNSW index (32M papers, 768d) is built and operational, but `semantic_search` only calls single-model `vector_search`. The existing `hybrid_search()` function in `search.py` already implements 3-signal RRF fusion (dense + dense + BM25) but is unreachable from any MCP tool. INDUS needs to be wired into the RRF pipeline as the domain-specific signal, and the tool surface needs consolidation from ~30 to 13 tools.

## Goals & Non-Goals

### Goals

- Wire INDUS into RRF fusion (INDUS dense + optionally text-embedding-3-large dense + BM25) via a single `search` MCP tool backed by `hybrid_search()`
- Consolidate 28-30 tools to 13 without losing any retrieval capability
- Fix latent bugs: `hybrid_search` defaults to `specter2` instead of `indus`; `entity_search` description/code mismatch on valid `entity_type` values
- Rewrite all tool descriptions using intent-action-contrast pattern (when to use, when NOT to use, under 80 words, zero implementation details)
- Convert session management from 4 explicit tools to implicit server-side state
- Validate consolidation with the 50-query eval harness

### Non-Goals

- Adding SPECTER2 as a fourth RRF signal (only 20K pilot rows; insufficient corpus coverage)
- Implementing Tool RAG / dynamic tool retrieval (13 tools is below the threshold where this adds value)
- Changing RRF constant k=60 (no evidence to change)
- Building StreamingDiskANN indexes (premature at current query volume)
- Full-text chunking pipeline (future work)

## Consolidated Tool Surface (13 tools)

| #   | Tool                  | Replaces                                                                 | Key Design Decision                                                                                                                                                                       |
| --- | --------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `search`              | `semantic_search`, `keyword_search`                                      | `mode`: hybrid (default) / semantic / keyword. Backed by `hybrid_search()` with INDUS + optionally text-embedding-3-large + BM25 via RRF                                                  |
| 2   | `concept_search`      | —                                                                        | Stays separate. UAT hierarchy traversal has categorically different input type (controlled vocabulary, not free text). Litmus test: agent cannot ignore mode param and get useful results |
| 3   | `get_paper`           | `get_paper`, `document_context`, `get_openalex_topics`, `entity_profile` | `include_entities` flag (default true) routes to `agent_document_context` materialized view. Absorbs document_context without extra tool slot                                             |
| 4   | `read_paper`          | `read_paper_section`, `search_within_paper`                              | `section` + optional `search_query` params. If search_query provided, runs ts_headline within paper                                                                                       |
| 5   | `citation_graph`      | `get_citations`, `get_references`, `get_citation_context`                | `direction` param (forward/backward/both) + `include_context` flag                                                                                                                        |
| 6   | `citation_similarity` | `co_citation_analysis`, `bibliographic_coupling`                         | `method` param (co_citation/coupling)                                                                                                                                                     |
| 7   | `citation_chain`      | —                                                                        | Stays (unique BFS/shortest-path semantics)                                                                                                                                                |
| 8   | `entity`              | `entity_search`, `resolve_entity`                                        | Both accept text input and return entity candidates. Compatible signatures → valid merge. Fix `entity_type` validation to match code                                                      |
| 9   | `entity_context`      | —                                                                        | Stays separate. Takes integer `entity_id` — fundamentally different input type from text-based entity tools                                                                               |
| 10  | `graph_context`       | `get_paper_metrics`, `explore_community`                                 | `include_siblings` flag for community exploration                                                                                                                                         |
| 11  | `find_gaps`           | `find_gaps`                                                              | Reads implicit session state. Optional `clear_first` param for session reset                                                                                                              |
| 12  | `temporal_evolution`  | —                                                                        | Stays (unique time-series output)                                                                                                                                                         |
| 13  | `facet_counts`        | —                                                                        | Stays (corpus distribution exploration)                                                                                                                                                   |

**Eliminated entirely**: `entity_profile` (subsumed by `get_paper(include_entities=true)`), `get_author_papers` (→ `search` with author filter), `add_to_working_set`, `get_working_set`, `get_session_summary`, `clear_working_set` (→ implicit tracking), `health_check` (internal, not agent-facing), `get_openalex_topics` (→ `get_paper`)

### Composite Tool Litmus Test

A tool merge via mode/action parameter is valid **if and only if** the agent can ignore the parameter and still get useful results (the default mode handles the common case). Applied:

- `search(mode=hybrid)`: Agent ignores mode → gets hybrid results. **Valid merge.**
- `citation_graph(direction=forward)`: Agent must specify direction → but forward is a sensible default. **Valid merge.**
- `entity(action=search|resolve)`: Both take text, return entity candidates. **Valid merge.**
- `concept_search` as `search(mode=concept)`: Agent ignores mode → gets hybrid results, NOT concept results. Agent must know UAT to use concept mode. **Invalid merge — keep separate.**

## Requirements

### Must-Have

- **M1: Unified `search` tool with hybrid RRF**
  - Replaces `semantic_search`, `keyword_search`
  - Parameters: `query` (required), `mode` (enum: `hybrid`|`semantic`|`keyword`, default `hybrid`), `filters` (optional)
  - In `hybrid` mode: embeds query with INDUS (preloaded) + BM25, fuses via `rrf_fuse()`. text-embedding-3-large has 0 rows — excluded from RRF until corpus coverage exists
  - Falls back gracefully when indexes are missing (existing circuit-breaker pattern)
  - Acceptance: `search(query="dark energy equation of state", mode="hybrid")` returns results with `fusion_method: "rrf"` in response metadata; `EXPLAIN ANALYZE` shows HNSW index scan for INDUS

- **M2: Fix `hybrid_search` default model + wire to MCP**
  - Change `model_name: str = "specter2"` to `model_name: str = "indus"` in `search.py` `hybrid_search()`
  - text-embedding-3-large has 0 rows (checked 2026-04-11) — skip OpenAI signal entirely. 2-signal RRF (INDUS + BM25) only
  - Acceptance: `grep -n "specter2" src/scix/search.py` returns zero hits in default parameter values

- **M3: Consolidate citation tools**
  - `citation_graph` tool: merges `get_citations` + `get_references` via `direction` param (forward/backward/both) + `include_context` flag for citation context
  - `citation_similarity` tool: merges `co_citation_analysis` + `bibliographic_coupling` via `method` param
  - `citation_chain` stays separate (unique BFS/shortest-path semantics)
  - Acceptance: old tool names handled via deprecated redirect; new tools return identical result shapes for same inputs

- **M4: Consolidate entity tools to 2**
  - `entity` tool: merges `entity_search` + `resolve_entity` (both text-input, compatible signatures)
  - `entity_context` stays separate (integer entity_id input — different type)
  - Fix `entity_type` validation: align description with code (`methods`, `datasets`, `instruments`, `materials`)
  - Acceptance: `entity(action="search", entity_type="methods", query="JWST")` returns results; `entity(action="search", entity_type="entities")` returns clear validation error

- **M5: Consolidate session tools into implicit state**
  - Auto-track papers returned by any tool into a "seen" set (server-side SessionState)
  - Auto-track papers fetched via `get_paper(include_entities=true)` into a "focused" set
  - `find_gaps` reads from implicit "focused" set + accepts optional `clear_first=true` param
  - Eliminate `add_to_working_set`, `get_working_set`, `get_session_summary`, `clear_working_set`
  - Acceptance: after calling `search(query="exoplanets")` then `get_paper(bibcode=...)` then `find_gaps()`, gaps tool discovers communities without explicit working set calls

- **M6: Merge `get_paper` + `document_context`**
  - `get_paper(bibcode, include_entities=true)` routes to `agent_document_context` materialized view when include_entities=true (default)
  - `include_entities=false` routes to simpler papers query
  - Absorb `get_openalex_topics` and `entity_profile` into this tool
  - Acceptance: `get_paper(bibcode="2024ApJ...X", include_entities=true)` returns paper metadata + all linked entities in one response

- **M7: Rewrite all 13 tool descriptions**
  - Intent-action-contrast pattern: what does the agent get, when to use vs alternatives, what NOT to use it for
  - No implementation details (no "pgvector", "HNSW", "tsvector", "JSONB", "GIN", model identifiers)
  - Under 80 words per description
  - Include one "Use X instead when..." contrast sentence
  - Acceptance: no tool description contains implementation terms; every description includes a contrast sentence

- **M8: Deprecated tool redirects**
  - Old tool names removed from `list_tools()` (agent sees only 13 tools)
  - Old names still handled in `_dispatch_tool` → return real results + `"deprecated": true` + `"use_instead": "new_name"` metadata
  - Log every redirect hit for removal planning
  - Acceptance: calling `semantic_search("test")` returns search results + deprecation metadata; `list_tools()` returns exactly 13 tools

### Should-Have

- **S1: Merge `read_paper_section` + `search_within_paper` into `read_paper`**
  - Parameters: `bibcode`, `section` (optional), `search_query` (optional)
  - If `search_query` provided, runs ts_headline search within paper
  - Acceptance: `read_paper(bibcode="2024ApJ...X", search_query="methodology")` returns highlighted matches

- **S2: Merge `get_paper_metrics` + `explore_community` into `graph_context`**
  - Returns PageRank, HITS, Leiden community assignments + optional sibling papers
  - Parameters: `bibcode`, `include_siblings` (bool, default false)
  - Acceptance: `graph_context(bibcode="...", include_siblings=true)` returns metrics + community papers

- **S3: RRF weight tuning**
  - Run 50-query eval comparing equal-weight RRF vs INDUS-weighted (1.5x) RRF
  - If INDUS-weighted improves nDCG@10 by >2%, make it the default
  - Acceptance: eval results saved to `results/rrf_weight_eval.json` with per-query scores

- **S4: Fold `get_author_papers` into `search` with author filter**
  - `search(query="", filters={"author": "Einstein, A."})` replaces `get_author_papers`
  - Acceptance: author search returns same results via both old and new tool paths

### Nice-to-Have

- **N1: Working set as MCP resource**
  - Expose `scix://session/working_set` as a subscribable MCP resource
  - Clients that support resources get live updates; tool fallback for others
  - Acceptance: MCP resource read returns current session papers

- **N2: Query latency benchmarks**
  - Benchmark hybrid search at 32M scale: target <100ms p95 (cache hit)
  - Run `EXPLAIN ANALYZE` on representative queries
  - Acceptance: p95 latency < 100ms across 50 benchmark queries

- **N3: Relevance confidence signal**
  - Return max cosine similarity score in search results metadata
  - Acceptance: search results include `max_similarity` field; value is 0.0-1.0

## Design Considerations

### Key Tensions (Resolved via Convergence Debate)

1. **Composite tools vs agent confusion**: Resolved via litmus test — merge only when agent can ignore the mode/action param and get useful results. Applied consistently: search modes (valid), citation direction (valid), entity search+resolve (valid), concept_search as mode (invalid — kept separate).

2. **INDUS replaces vs complements SPECTER2**: Resolved — INDUS replaces SPECTER2 as the domain-specific signal. Both are 768d models trained on scientific title-abstract pairs. SPECTER2 embeddings (20K pilot) remain in database but are not queried in production RRF.

3. **Implicit vs explicit session state**: Resolved — eliminate all 4 session tools. Two-tier implicit tracking: "seen" (auto from search) vs "focused" (auto from `get_paper(include_entities=true)`). `find_gaps` uses "focused" set. `clear_first` param on `find_gaps` handles session reset.

4. **Description quality vs tool count**: Both matter. Rewriting descriptions (M7) is high-leverage and parallelizable with structural consolidation. Target: 40-80 words, intent-action-contrast pattern, zero implementation details.

5. **Deprecation strategy**: Resolved — old names removed from `list_tools()` but redirected in `_dispatch_tool` with deprecation metadata. Log hits. Remove zero-hit aliases in Phase 2.

### Latency Budget (hybrid search)

| Component                        | Estimated     | Notes                                     |
| -------------------------------- | ------------- | ----------------------------------------- |
| INDUS HNSW scan (32M, 768d)      | 5-15ms        | m=16, ef_search=100                       |
| text-embedding-3-large HNSW scan | 5-15ms        | Only if coverage >5M rows                 |
| BM25 lexical scan (GIN)          | 10-20ms       | Already exists                            |
| INDUS query embedding (GPU)      | 10-50ms       | Model preloaded                           |
| OpenAI API call (cached)         | 0-300ms       | LRU cache 512; skip if coverage gated out |
| RRF fusion                       | <1ms          | In-memory                                 |
| **Total (2-signal, INDUS+BM25)** | **25-85ms**   | If text-embedding-3-large gated out       |
| **Total (3-signal, cache hit)**  | **30-100ms**  |                                           |
| **Total (3-signal, cache miss)** | **130-400ms** | OpenAI API dominates                      |

### Migration Path (2 Phases)

**Phase 1: Consolidate + Validate**

- Fix `hybrid_search` default from `specter2` to `indus`
- Check text-embedding-3-large coverage → decide 2-signal vs 3-signal
- Wire `hybrid_search` to MCP as `search` tool
- Consolidate all tool groups (M1-M8)
- Rewrite all descriptions (M7)
- Add deprecated redirects for old names (M8)
- Run 50-query eval: before/after nDCG@10 comparison
- Gate: if eval regresses >2%, investigate before proceeding

**Phase 2: Tune + Clean Up**

- Run RRF weight tuning eval (S3)
- Remove deprecated aliases with zero query_log hits
- Implement should-have items (S1-S4)
- Benchmark query latency (N2)

## Risk Annotations (from Premortem)

Full premortem analysis: [premortem_indus_integration_mcp_consolidation.md](premortem_indus_integration_mcp_consolidation.md)

### Top Risks (4 scored Critical/High = 12)

| Risk                                                                                                                                                                                                             | Score              | Required Mitigation                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Filtered HNSW iterative scan blowup** — selective filters (<1% of 32M) cause 2-8s queries, exhausting connection pool                                                                                          | 12 (Critical/High) | Add cardinality estimation in `hybrid_search()`; filter-first fallback for <1% selectivity                                   |
| **Matview refresh as operational hazard** — no automated refresh, 45-min lock at scale, cascading pool exhaustion                                                                                                | 12 (Critical/High) | Replace matview default path with live query + 60s app cache; or deploy automated off-peak refresh with staleness monitoring |
| **MCP SDK + UAT dependency breakage** — unpinned mcp SDK pulls breaking 2.0; UAT SKOS URL hardcoded to `master` branch (→ 404 after rename to `main`). Both cause silent total tool failures undetected for days | 12 (Critical/High) | Pin `mcp>=1.2,<2.0`; vendor UAT SKOS or pin to release tag; add startup self-test + all-tool smoke test on deploy            |
| **Scale ceiling at chunk-level vectors** — full-text chunking → 350M vectors in same table; HNSW rebuild takes 9+ days, runs out of disk                                                                         | 12 (Critical/High) | Separate `chunk_embeddings` table; vector count threshold before rebuild                                                     |
| **Eval gap: single-turn only** — 50-query eval catches nDCG@10 regression but not multi-step workflow degradation                                                                                                | 6 (High/Medium)    | Add 10 multi-tool workflow scenarios to eval before Phase 1 gate                                                             |

### Design Changes Required by Premortem

1. **M6 amendment**: Default `include_entities=false` on `get_paper` (not true). Agents opt-in to entity payloads. Prevents context bloat cascade identified in Scope failure narrative.
2. **M2 amendment**: Add cardinality-aware query routing to `hybrid_search()`. If filter selectivity <1%, use filter-first CTE + brute-force cosine instead of HNSW iterative scan.
3. **New M9: Operational readiness** — automated matview refresh after daily_sync, staleness probe, connection pool increase to 20 with fast/slow isolation, startup readiness gate (model loaded before accepting requests).
4. **New M10: Dependency hardening** — pin `mcp` SDK to `>=1.2,<2.0`; vendor UAT SKOS file or pin to release tag; add startup self-test that calls `list_tools()` and validates 13 tools with valid schemas; add all-tool smoke test (1 golden-path query per tool) to CI.
5. **Phase 1 gate amendment**: Eval must include multi-step workflow scenarios, not just single-turn nDCG@10. Also must include smoke test of all 13 tools returning valid responses.

## Open Questions

1. **text-embedding-3-large corpus coverage**: `SELECT count(*) FROM paper_embeddings WHERE model_name = 'text-embedding-3-large'` — answer determines 2-signal vs 3-signal RRF. If >5M, include. If <1M, gate out.
2. **RRF weighting**: Equal weights are the default. S3 eval will determine if INDUS-weighted improves nDCG@10 by >2%.
3. **`temporal_evolution` and `facet_counts` usage**: If query_log shows <1% usage, consider folding into other tools in Phase 2.
4. **HNSW index for text-embedding-3-large**: If coverage justifies inclusion, need migration for `idx_embed_hnsw_openai` on 1024d vectors.
5. **Matview vs live query**: Should `get_paper(include_entities=true)` use the matview or a live JOIN + app cache? Premortem strongly recommends live query to eliminate refresh-as-operational-risk.
6. **Chunk embedding table separation**: When full-text chunking begins, use dedicated `chunk_embeddings` table or extend `paper_embeddings`? Premortem says separate to preserve 32M-row HNSW build times.

## Research Provenance

### Convergence Debate Summary

Three independent research agents (Prior Art, Technical Design, Agent UX) produced divergent findings, then three debate agents (Minimalist, Pragmatist, Purist) refined tensions through 2 rounds of structured debate.

**Resolved by consensus:**

- 13-tool target (converged from 10/13/14)
- Session tools → implicit state (all 3)
- `hybrid_search` wiring as priority (all 3)
- Equal RRF weights, eval later (all 3)
- Gate text-embedding-3-large on coverage data (all 3)
- Old names out of `list_tools()` (all 3)

**Resolved by 2-1 vote + decisive argument:**

- `concept_search` stays separate (purist + pragmatist; decisive: "can agent ignore mode param and get useful results?" → no for concept)
- 2 entity tools not 1 or 3 (all 3 converged; decisive: incompatible input signatures make action-param merge harmful)

**Resolved by synthesis:**

- `document_context` → `get_paper(include_entities=true)` (purist conceded, pragmatist proposed, minimalist already wanted)
- Deprecation via redirect with metadata (pragmatist proposed, purist and minimalist converged)
- 2-phase migration (compromise between 1-phase and 3-phase proposals)

### Key Principles Emerged

1. **Composite tool litmus test**: Merge via mode/action param only if agent can ignore the param and get useful results
2. **Confusion pairs come from descriptions, not names**: Fix descriptions before merging tools
3. **Input signature compatibility**: Tools with the same input types can merge; tools with different input types (text vs integer ID) should stay separate
4. **Design around data, not assumptions**: Check text-embedding-3-large coverage before committing to 3-signal RRF
