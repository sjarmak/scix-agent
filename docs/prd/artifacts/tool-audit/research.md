# Research — MCP Tool Audit

> Scratch notes for the tool-audit work unit. Final deliverable: `../mcp_tool_audit.md`.
>
> NOTE on path: the work-unit instructions specified `.claude/prd-build-artifacts/`,
> but that path is permission-blocked by the agent policy. These scratch notes
> are kept under `docs/prd/artifacts/tool-audit/` so they remain version-controlled
> alongside the deliverable.

## Source of truth
- File: `src/scix/mcp_server.py` (2032 lines)
- Authoritative tool list lives in the `list_tools()` async handler at line 699.
- Authoritative tool name list: `EXPECTED_TOOLS` tuple at line 522. Startup self-test asserts `tool_count == 13`.

## True count

13 `Tool(...)` registrations inside `list_tools()` (lines 700-1152). Verified by
`awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py | grep -cE '^[[:space:]]+name="' = 13`.

The PRD's "current 22 MCP tools" claim is wrong. The discrepancy comes from counting
deprecated aliases (~21 entries in `_DEPRECATED_ALIASES`) plus a few legacy in-process
handlers that have dispatch arms but are NOT advertised by `list_tools()`. Total
dispatchable surface (advertised + alias + legacy) is ~40 paths, but the
agent-visible tool count is 13.

## Enumerated active tools (visible via list_tools)

| # | name | required args | optional args (defaults) | Description gist |
|---|---|---|---|---|
| 1 | search | query | mode=hybrid, filters, limit=10, disambiguate=true | Hybrid/semantic/keyword search. |
| 2 | concept_search | query | include_subtopics=true, limit=20 | UAT taxonomy concept lookup. |
| 3 | get_paper | bibcode | include_entities=false | Single-paper metadata. |
| 4 | read_paper | bibcode | section=full, search_query, char_offset=0, limit=5000 | Read body section OR search inside body. |
| 5 | citation_graph | bibcode | direction=forward, include_context=false, limit=20 | Direct citation/reference walk. |
| 6 | citation_similarity | bibcode | method=co_citation, min_overlap=2, limit=20 | Structural similarity via shared citations. |
| 7 | citation_chain | source_bibcode, target_bibcode | max_depth=5 | Shortest path between two papers. |
| 8 | entity | action, query | entity_type, discipline, fuzzy=false, limit=20, min_confidence_tier, sources | Unified entity search/resolve. |
| 9 | entity_context | entity_id | — | Full profile of a known entity_id. |
| 10 | graph_context | bibcode | include_community=false, resolution=coarse, signal=semantic, limit=20 | Influence + community membership. |
| 11 | find_gaps | (none) | resolution=coarse, limit=20, clear_first=false | Cross-community recommender. |
| 12 | temporal_evolution | bibcode_or_query | year_start, year_end | Year-by-year citations or pub volume. |
| 13 | facet_counts | field | filters, limit=50 | Distribution by metadata field. |

## Deprecated aliases

`_DEPRECATED_ALIASES` at line 414 maps old names to new tools. Each alias
rewrites args via `_transform_deprecated_args` and dispatches; results are
wrapped with `deprecated: true, use_instead: <new_tool>`.

| Old name | Routes to | Arg transform |
|---|---|---|
| semantic_search | search | mode=semantic |
| keyword_search | search | mode=keyword, terms->query |
| get_citations | citation_graph | direction=forward |
| get_references | citation_graph | direction=backward |
| co_citation_analysis | citation_similarity | method=co_citation |
| bibliographic_coupling | citation_similarity | method=coupling |
| entity_search | entity | action=search; entity_name->query |
| resolve_entity | entity | action=resolve |
| entity_profile | get_paper (use_instead); dedicated handler | own schema |
| document_context | get_paper | include_entities=true |
| get_openalex_topics | get_paper | include_entities=true |
| get_paper_metrics | graph_context | include_community=false |
| explore_community | graph_context | include_community=true |
| get_author_papers | search (use_instead); legacy handler | passthrough |
| add_to_working_set | get_paper (use_instead); legacy handler | passthrough |
| get_working_set | find_gaps (use_instead); legacy handler | passthrough |
| get_session_summary | find_gaps (use_instead); legacy handler | passthrough |
| clear_working_set | find_gaps (use_instead); legacy handler | passthrough |
| get_citation_context | citation_graph (use_instead); legacy handler | passthrough |
| read_paper_section | read_paper | passthrough |
| search_within_paper | read_paper | query->search_query |

Plus `health_check` — present in `_dispatch_consolidated` and `TOOL_TIMEOUTS`
but neither in `list_tools()` nor `_DEPRECATED_ALIASES`. Reachable only by
direct call.

## Verifying PRD consolidation hypotheses against code

1. **get_paper + read_paper -> single read_paper with depth param.** Currently
   distinct. `get_paper` returns metadata; `read_paper` reads/searches body.
   Different timeouts (5s vs 10s), different output shapes.
2. **citation_similarity already covers co-citation + bibliographic_coupling?**
   Verified at `_handle_citation_similarity` (line 1684): `method` enum dispatches
   to `search.co_citation_analysis` or `search.bibliographic_coupling`. Already
   consolidated.
3. **citation_graph + citation_chain -> single citation_traverse.** Different
   required args (one bibcode vs two) and output shapes (neighbor list vs ordered
   path).
4. **entity + entity_context — merge?** `entity` takes text, returns candidates;
   `entity_context` takes numeric `entity_id`, returns full profile. Could fold
   via `action=context` with type-overloaded `query`, at cost of clarity.
5. **find_similar_by_examples — keep / drop?** Does not exist in current code.
   Drop from plan.

## SKILL.md cross-reference

`/home/ds/.claude/skills/scix-mcp/SKILL.md` documents the same 13 tools in 5
categories (Search 2, Paper access 2, Citation graph 3, Entity system 2,
Structure 4). Total = 13 — matches code.

## Observed parameter overlaps

- `bibcode` required for: get_paper, read_paper, citation_graph,
  citation_similarity, graph_context. `bibcode_or_query` for temporal_evolution.
  5+ tools dispatch off the same identifier.
- `filters` shared by `search` and `facet_counts`.
- `limit` on 11/13 tools.
- No two tools have identical input schemas.

## Conclusion

13 active tools is already at the consolidation target. Remaining decisions are
mostly ratify-current-state with one viable further trim (`citation_chain` into
`citation_graph` via a `path` mode), one weak candidate (`entity_context` into
`entity`), and explicit rejections for merging `get_paper`+`read_paper` and
introducing `find_similar_by_examples`.
