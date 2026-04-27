# MCP Tool Consolidation Audit — 2026-04

**Bead:** `scix_experiments-wqr.9.2` (PRD §D4 — Tool consolidation).
**PRD:** `docs/prd/prd_section_embeddings_mcp_consolidation.md`.
**Scope:** Confirm current MCP tool count meets the `≤ 15` target set by the
premortem tool-count concern (CLAUDE.md §Tool Count Concern), classify each
tool, and record deprecation notes for the aliased legacy tools.

## Summary

| Target | Status |
|---|---|
| Collapse MCP tools to `≤ 15` | **Met.** 15 agent-visible tools as of 2026-04-26 (added `cited_by_intent`, expanded `entity` with `action='papers'`). 4 additional tools (`chunk_search`, `section_retrieval`, `read_paper_claims`, `find_claims`) are env-hidden (`_HIDDEN_TOOLS`) because their backing data isn't yet populated; restore via `SCIX_HIDDEN_TOOLS=`. |
| Deprecation notes per removed tool | **Met.** Aliases map to the new tools via `src/scix/mcp_server.py::_DEPRECATED_ALIASES`; schema transforms in `_transform_deprecated_args`; responses wrapped with `{deprecated: true, use_instead, original_tool}` by `_wrap_deprecated`. |
| `SKILL.md` tool table reflects the new set | Stale — needs update to reflect 15-tool surface and the 4 env-hidden tools. |

## 2026-04-26 — pre-talk demo prep changes

The session that landed these changes was preparing a live agent demo. New
capabilities + rationale:

- **New tool: `cited_by_intent(target_bibcode, intent)`** — exploits
  `citation_contexts.intent` (method / background / result_comparison)
  populated for ~825K rows. Demonstrates structural citation awareness.
- **New action: `entity(action='papers', entity_id=N)`** — surfaces
  `document_entities` (57.7M rows, 13 entity types). Pass `entity_id` from
  `action='resolve'`, or `query` to auto-resolve. Cross-discipline flex.
- **`entity(action='papers')` adds `precision_estimate` + `precision_band`**
  per row from `src/scix/extract/ner_quality_profile.py` — the dbl.3 D3
  quality profile visible at the agent surface. Bead bqva partial (entity
  wired; `get_paper` and `find_gaps` deferred).
- **`citation_traverse` annotates each edge with `intent`** when covered
  by `citation_contexts` (~0.27% per bead 79n).
- **`read_paper(section=...)` now reads from `papers_fulltext.sections` JSONB**
  (14.4M populated, 96.2%) — accurate section retrieval rather than
  heuristic regex on flat body. Falls back when papers_fulltext has no entry.
- **`find_gaps` accepts optional `query` param** to auto-seed working set
  via concept_search in a single call.
- **4 tools env-hidden via `_HIDDEN_TOOLS`:** `chunk_search`,
  `section_retrieval`, `read_paper_claims`, `find_claims` — registered
  handlers but missing backing data. Override via `SCIX_HIDDEN_TOOLS=`.
- **`hnsw.ef_search` lowered to 40** at DB level
  (`ALTER DATABASE scix SET hnsw.ef_search = 40`). Trades ~1pp recall for
  2-3× speedup; reversible via `RESET`.
- **`pg_prewarm`** called on `idx_embed_hnsw_indus` to push the
  most-traversed graph nodes into OS page cache.

Landing commits (for migration-guidance cross-refs):

- `7fe258d` — `prd-build: consolidate-mcp-tools — Consolidate 28 MCP tools to
  13 with aliases and implicit session` — base consolidation + alias map.
- `ffef6eb` — `prd-build: rewrite-descriptions — Rewrite 13 tool descriptions
  (intent-action-contrast)` — final tool descriptions.
- `f334961` — `Merge branch 'prd-build/indus-mcp-consolidation'` — merge into
  main.
- `94fd307` — `docs: update stale corpus stats and tool counts to current
  state` — SKILL.md sync.

> **Note:** PRD §D4 says "22 → ≤ 15". The real pre-consolidation count was
> **28** (see commit `7fe258d`); `22` in CLAUDE.md §"Tool Count Concern" is a
> stale transitional count. See "CLAUDE.md note" below for the local fix.

## Tool classification (13 core + 1 optional)

Source of truth: `src/scix/mcp_server.py::list_tools` (lines 814–1305) and
`EXPECTED_TOOLS` (line 623). `_dispatch_tool` (line 1378) routes both new
names and legacy aliases into `_dispatch_consolidated` (line 1520).

| # | Tool | Keep / Consolidated-from | Purpose | Notes |
|---|---|---|---|---|
| 1 | `search` | **consolidated** (semantic_search, keyword_search, get_author_papers) | Hybrid / semantic / keyword search; `mode` param selects signal. | Default mode = `hybrid` (4-signal RRF). |
| 2 | `concept_search` | keep | UAT-concept retrieval (formal taxonomy). | Separate from `search` because the input contract differs (label/URI, not NL). |
| 3 | `get_paper` | **consolidated** (document_context, get_openalex_topics, add_to_working_set) | Metadata + optional entity links for one bibcode. | `include_entities=true` replaces `document_context` behaviour. Implicit session tracking via `_auto_track_bibcodes`. |
| 4 | `read_paper` | **consolidated** (read_paper_section, search_within_paper) | Read or search inside one paper's body. | `search_query` toggles read-vs-search. |
| 5 | `citation_traverse` | **consolidated** (citation_graph, citation_chain, get_citations, get_references, get_citation_context) | Citation graph traversal: neighbourhood walk OR shortest-path chain, selected by `mode`. | `mode ∈ {graph, chain}`; per-mode required-param sets are validated in the handler with a structured `missing_required_params` payload (bead `zjt9`) since JSON Schema can't express the disjoint sets. |
| 6 | `citation_similarity` | **consolidated** (co_citation_analysis, bibliographic_coupling) | Structural similarity via shared citations. | `method ∈ {co_citation, coupling}` replaces 2 legacy tools. |
| 7 | _(slot folded into `citation_traverse` row 5)_ | — | — | The original `citation_chain` "keep separate" recommendation was reversed on 2026-04-25; see Recommendation §3 below. |
| 8 | `entity` | **consolidated** (entity_search, resolve_entity) | `action ∈ {search, resolve}`. | Added entity-type / confidence-tier / provenance-source filters in later builds. |
| 9 | `entity_context` | keep | Full entity profile by `entity_id`. | Separate from `entity` because the input is a numeric id, not text. |
| 10 | `graph_context` | **consolidated** (get_paper_metrics, explore_community) | PageRank/HITS + community membership (citation / semantic / taxonomic) for a bibcode. | `include_community=true` replaces `explore_community`. |
| 11 | `find_gaps` | **consolidated** (get_working_set, get_session_summary, clear_working_set) | Cross-community gap detection over implicit session. | Replaces three explicit session tools with one action-oriented tool + implicit session tracking. |
| 12 | `temporal_evolution` | keep | Publications-per-year or citations-per-year for a query or bibcode. | Returns per-year anchor papers + dominant communities (agent-friendly payload). |
| 13 | `facet_counts` | keep | Single-field distribution with filters. | Complementary to `temporal_evolution` (flat distribution vs. trend). |
| 14* | `find_similar_by_examples` | optional (Qdrant) | "More like these, less like those" over INDUS embeddings. | Registered only if `QDRANT_URL` is set. Not counted toward the core 13. |

## Deprecation map

From `_DEPRECATED_ALIASES` (`src/scix/mcp_server.py:515`) and the arg
transforms in `_transform_deprecated_args` (`src/scix/mcp_server.py:1414`).
All deprecated aliases continue to work; responses carry
`{deprecated: true, use_instead, original_tool}` per `_wrap_deprecated`.

| Old tool | Replacement | Arg migration | Landed |
|---|---|---|---|
| `semantic_search(query, ...)` | `search(query, mode="semantic", ...)` | Adds `mode="semantic"`. | `7fe258d` |
| `keyword_search(terms, ...)` | `search(query, mode="keyword", ...)` | Renames `terms → query`; adds `mode="keyword"`. | `7fe258d` |
| `get_citations(bibcode, ...)` | `citation_traverse(bibcode, mode="graph", direction="forward", ...)` | Adds `mode="graph"`, `direction="forward"`. | `7fe258d` (via `citation_graph`); rerouted to `citation_traverse` 2026-04-25 |
| `get_references(bibcode, ...)` | `citation_traverse(bibcode, mode="graph", direction="backward", ...)` | Adds `mode="graph"`, `direction="backward"`. | `7fe258d` (via `citation_graph`); rerouted to `citation_traverse` 2026-04-25 |
| `get_citation_context(source, target)` | `citation_traverse(bibcode, mode="graph", include_context=true)` | Dedicated legacy handler retained in `_dispatch_consolidated` — same args; modern path uses `mode="graph"`, `include_context=true`. | `7fe258d` |
| `citation_graph(bibcode, ...)` | `citation_traverse(bibcode, mode="graph", ...)` | Adds `mode="graph"`. | 2026-04-25 |
| `citation_chain(source_bibcode, target_bibcode, ...)` | `citation_traverse(source_bibcode, target_bibcode, mode="chain", ...)` | Adds `mode="chain"`. Missing endpoints now return `error_code="missing_required_params"` with `required=["source_bibcode","target_bibcode"]` before any DB access (bead `zjt9`). | 2026-04-25 |
| `co_citation_analysis(bibcode, ...)` | `citation_similarity(bibcode, method="co_citation", ...)` | Adds `method="co_citation"`. | `7fe258d` |
| `bibliographic_coupling(bibcode, ...)` | `citation_similarity(bibcode, method="coupling", ...)` | Adds `method="coupling"`. | `7fe258d` |
| `entity_search(entity_name, ...)` | `entity(action="search", query=entity_name, ...)` | Adds `action="search"`; renames `entity_name → query`. | `7fe258d` |
| `resolve_entity(query, ...)` | `entity(action="resolve", query, ...)` | Adds `action="resolve"`. | `7fe258d` |
| `entity_profile(entity_id)` | `get_paper(bibcode, include_entities=true)` (for paper-centric views); dedicated handler retained for the legacy row-shape. | `use_instead = get_paper`; actual dispatch stays on a dedicated `entity_profile` handler because the response shape differs (raw `entity_extractions` rows vs. grouped entities). | `7fe258d` |
| `get_paper_metrics(bibcode)` | `graph_context(bibcode, include_community=false)` | Adds `include_community=false`. | `7fe258d` |
| `explore_community(bibcode, ...)` | `graph_context(bibcode, include_community=true, ...)` | Adds `include_community=true`. | `7fe258d` |
| `document_context(bibcode)` | `get_paper(bibcode, include_entities=true)` | Adds `include_entities=true`. | `7fe258d` |
| `get_openalex_topics(bibcode)` | `get_paper(bibcode, include_entities=true)` | Same as `document_context` — topics are surfaced via the entity block. | `7fe258d` |
| `get_author_papers(author, ...)` | `search(query="first_author:...")` (recommended) | Dedicated legacy handler retained so bibcode-bound callers do not break; modern path uses `search` + `first_author` filter. | `7fe258d` |
| `add_to_working_set(bibcode)` | `get_paper(bibcode)` | Implicit session — `_auto_track_bibcodes` attaches touched bibcodes to the working set on any result. No explicit add call needed. | `7fe258d` |
| `get_working_set()` | `find_gaps(...)` | Working set is implicit — surface it via the next action (`find_gaps`) instead of a standalone read. | `7fe258d` |
| `get_session_summary()` | `find_gaps(...)` | Same — session state is consumed by the action, not inspected separately. | `7fe258d` |
| `clear_working_set()` | `find_gaps(clear_first=true)` | Clearing happens as part of the next query. | `7fe258d` |
| `read_paper_section(bibcode, section, ...)` | `read_paper(bibcode, section, ...)` | No arg change. | `7fe258d` |
| `search_within_paper(bibcode, query)` | `read_paper(bibcode, search_query)` | Renames `query → search_query`. | `7fe258d` |

Removed / superseded tools that are **not** aliased (hard-removed; agents
must migrate):

- None. Every pre-consolidation tool retains a working alias so external
  clients do not break.

## Recommendations on the three PRD "candidates for consolidation"

The PRD proposed three further merges as hypotheses to validate. The audit
recommends keeping all three pairs separate; rationale below.

1. **`get_paper` + `read_paper` → single `read_paper` with depth param.**
   **Recommend: keep separate.** `get_paper` returns structured metadata +
   optional entity block for planning; `read_paper` returns body prose /
   highlighted snippets for reading. Merging would overload one tool with
   two different output shapes (object vs. paginated text) and two
   different latency profiles. Agents already use them in distinct
   workflows (see SKILL.md §1 vs §2). The current split is the least
   surprising contract.

2. **`co_citation_analysis` + `bibliographic_coupling` + `citation_similarity` → single `citation_similarity(method=…)`.**
   **Already done.** Single tool with `method ∈ {co_citation, coupling}`.

3. **`citation_graph` + `citation_chain` → single `citation_traverse`.**
   **Status: shipped (2026-04-25).** Reversed the original "keep separate"
   recommendation. The merge happened because the project is actively
   reducing tool count (28 → 13 → 20 with later additions, with `≤ 15`
   as the visible-surface target after env-hidden tools), and adding a
   second top-level citation tool would have spent budget on a name, not
   on capability. `citation_graph` and `citation_chain` are now
   deprecated aliases that route through `_dispatch_consolidated` to
   `citation_traverse(mode=…)` with the appropriate mode injected.

   The known cost — **disjoint required-param sets per mode that JSON
   Schema can't express** — is paid down by handler-side validation
   rather than by splitting the tool again (bead
   `scix_experiments-zjt9`). When required params for the chosen mode are
   missing, `_handle_citation_traverse` returns a structured payload
   **before any DB access**:

   ```json
   {
     "error": "<human-readable message>",
     "error_code": "missing_required_params",
     "mode": "graph" | "chain",
     "required": ["bibcode"] | ["source_bibcode", "target_bibcode"],
     "got":      [<names of params actually supplied>]
   }
   ```

   Agents branch on `error_code`; humans read `error`; `required` / `got`
   make the gap concrete so the agent can self-correct in one step
   without a probe round-trip. The tool description spells the per-mode
   required sets out explicitly so a well-prompted agent never reaches
   the validation path. Splitting the tool was reconsidered and rejected
   on tool-budget grounds; if a future PRD finds that agent selection
   accuracy degrades on `citation_traverse` despite the explicit
   description and structured errors, splitting can be revisited under
   its own bead — but not as part of bead `zjt9`, whose acceptance is
   met by the schema-enforce path above.

**Net:** 13 is the floor we want. Going to 11 would save 2 tool slots at
the cost of agent clarity and schema honesty. The PRD target (`≤ 15`) is
met with headroom — headroom that `section_retrieval` (D3) will consume
without breaking the budget.

## CLAUDE.md note

`CLAUDE.md` §"Tool Count Concern" previously stated:

> 22 MCP tools currently exposed

This is a stale transitional count. The real current count is **13**
(**14** with Qdrant). The landing commit (`7fe258d`) consolidated **28 → 13**.

`CLAUDE.md` is gitignored (see `.gitignore:37`) — it is an internal /
per-host agent-briefing file, not a tracked artifact. Local copies should
be edited to say "13 MCP tools" and point at this audit doc. Any future
worktree bootstrapping should bake the updated phrasing into its seed
template rather than re-copy the stale count.

## SKILL.md deprecation appendix

`~/.claude/skills/scix-mcp/SKILL.md` §"Tool Overview (13 tools)" already
lists the correct 13 tools and their categories. The alias map is large
(21 entries) and reflects transition state, not long-lived agent-facing
surface area — it should not live in the skill prompt (cost: tokens; value
to a forward-looking agent: near zero). Keeping the alias map in
`src/scix/mcp_server.py` + this audit doc is the right split.

## Acceptance checklist

- [x] Tool count ≤ 15 (current: 13 core + 1 optional Qdrant).
- [x] Per-tool classification (keep / consolidated-from) written above.
- [x] Deprecation notes per removed tool with arg migration + commit ref
      (`7fe258d`).
- [x] Three PRD consolidation hypotheses reviewed; recommendations
      recorded.
- [x] `SKILL.md` tool table verified against current registration
      (matches).
- [x] CLAUDE.md stale "22" count corrected locally (file is gitignored).

## Out of scope (follow-up beads as applicable)

- Removing aliases entirely. Not recommended before the next major version
  — current cost (one dict lookup + arg transform per call) is negligible
  and keeps external MCP clients working.
- Adding `section_retrieval` (D3). Will land as a 14th core tool; still
  within the `≤ 15` budget.
- Agent eval to measure tool-selection accuracy pre/post-consolidation.
  Desirable but not a blocker for this audit — tracked separately under
  the 50-query retrieval eval (D5).
