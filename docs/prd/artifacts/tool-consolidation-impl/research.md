# Research — tool-consolidation-impl

## Metadata
| Field | Value |
|---|---|
| Date | 2026-04-25 |
| Author | tool-consolidation-impl agent |
| Source branches | `prd-build/section-embeddings-mcp-consolidation` |
| Source files | `src/scix/mcp_server.py`, `docs/prd/artifacts/mcp_tool_audit.md`, `/home/ds/.claude/skills/scix-mcp/SKILL.md`, `tests/test_mcp_smoke.py`, `tests/test_scix_deep_search.py`, `.claude/agents/deep_search_investigator.md` |

## Current state of `list_tools()` (verified)

`awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py | grep -oE 'name="[a-z_]+"' | sort -u` returns 17 unique names:

```
citation_chain, citation_graph, citation_similarity, claim_blame,
concept_search, entity, entity_context, facet_counts, find_gaps,
find_replications, find_similar_by_examples, get_paper, graph_context,
read_paper, search, section_retrieval, temporal_evolution
```

`find_similar_by_examples` is registered conditionally inside an `if _qdrant_enabled():` block (line 1839), so the *runtime* count drops to 16 when the Qdrant backend is not configured (the production default per the deployment-state memory). `EXPECTED_TOOLS` covers 16; the 17th is tracked in `_OPTIONAL_TOOLS`.

## Audit document gap

`docs/prd/artifacts/mcp_tool_audit.md` was written against an older base of 13 tools. It does NOT cover:

| Tool | First registered in | Current home |
|---|---|---|
| `claim_blame` | PRD MH-4 | `src/scix/mcp_server.py:1707` |
| `find_replications` | PRD MH-4 | `src/scix/mcp_server.py:1756` |
| `find_similar_by_examples` | qdrant_tools probe (gated) | `src/scix/mcp_server.py:1841` |
| `section_retrieval` | sibling unit (this PRD) | `src/scix/mcp_server.py:1801` |

Per the work-unit prompt, the default verdict for each is:

* **`claim_blame`** — KEEP. Provenance-specific, no obvious consolidation target.
* **`find_replications`** — KEEP. Semantic-specific (replication/refutation inference), no consolidation target.
* **`find_similar_by_examples`** — REMOVE. Qdrant backend is local-only and not in active use (see `mcp_deployment_state` memory). Removing it brings the active tool count to 15, exactly meeting the acceptance criterion.
* **`section_retrieval`** — KEEP (already implemented by the sibling unit; was the "new tool" delta on this PRD).

## Audit recommended consolidations

Per the migration table in `mcp_tool_audit.md` (section "New migration introduced by this audit"):

* `citation_graph` + `citation_chain` -> `citation_traverse(mode="graph"|"chain")`
  * The audit's H3 specifies `mode: neighbors|path` — but the work-unit prompt explicitly mandates `mode: graph|chain` (default `graph`). The work-unit prompt wins.

No other audited tool is flagged for consolidation. All other audited tools are KEEP.

## `citation_graph` and `citation_chain` implementations

Located in `src/scix/mcp_server.py`:

* `_handle_citation_graph(conn, args)` at line 2448 — uses `args["bibcode"]`, `args.get("direction", "forward")`, `args.get("limit", 20)`. Calls `search.get_citations` for forward and `search.get_references` for backward. Direction `both` returns a combined envelope.
* `citation_chain` is dispatched inline at line 2137 — `max_depth = max(1, min(args.get("max_depth", 5), 5))` then calls `search.citation_chain(conn, source_bibcode, target_bibcode, max_depth=max_depth)`.

No additional helper for `include_context` is wired beyond what the underlying `search.get_citations` and `search.get_references` accept (the audit's mention of `include_context` corresponds to the Tool input schema only; the dispatch layer does not currently forward this — but that's outside this work unit's scope).

## `find_similar_by_examples` removal surface

* `Tool(name="find_similar_by_examples", ...)` registration: lines 1840-1882, conditionally inside `if _qdrant_enabled():`.
* Dispatch entry: lines 2102-2103 (`if name == "find_similar_by_examples": return _handle_find_similar_by_examples(args)`).
* Handler: `_handle_find_similar_by_examples` at line 2951.
* Tracking: `_OPTIONAL_TOOLS = ("find_similar_by_examples",)` at line 921 and `_expected_tool_set()` at line 924.

After retirement, the cleanest path is: drop the `Tool(...)` registration entirely (so it never appears in `list_tools()` regardless of Qdrant being enabled), drop the `_OPTIONAL_TOOLS` entry (or set it to an empty tuple), and replace the dispatch with a deprecation error. The handler can stay (private function) but is now unreachable from the MCP surface — fine to keep for now to minimise diff churn; can be deleted in a follow-up.

## SKILL.md current state

`/home/ds/.claude/skills/scix-mcp/SKILL.md` (line 20) currently advertises "Tool Overview (13 tools)". Already stale — the embedded table lists the 13 baseline tools (search, concept_search, get_paper, read_paper, citation_graph, citation_chain, citation_similarity, entity, entity_context, graph_context, find_gaps, temporal_evolution, facet_counts) and does NOT mention `claim_blame`, `find_replications`, `section_retrieval`, or `find_similar_by_examples`.

It needs a full rewrite of the tool table to reflect the post-consolidation roster of 15.

## Strategy decisions

Per the work-unit prompt's options and trade-offs:

1. **Consolidation strategy for `citation_graph`/`citation_chain`**: SHIM (forward via `_DEPRECATED_ALIASES`). Rationale: existing alias pattern is well-tested; agents already calling these tools by name continue to work; removal-with-error breaks the deep_search_investigator persona which lists `citation_chain`. The shim adds `mode` and forwards.
2. **Strategy for `find_similar_by_examples`**: REMOVE OUTRIGHT. Rationale: qdrant backend is down, no agent depends on it, removing it brings count to 15 (acceptance criterion strictly met). The dispatch layer raises a clear "removed in 2026-04-25" error if anyone calls the name.

## Test plan

* New file `tests/test_mcp_tool_consolidation.py`:
  * `test_citation_traverse_graph_mode_equivalent_to_old_citation_graph` — patch `search.get_citations` and verify `citation_traverse(mode="graph", ...)` returns the same result as `citation_graph(...)`.
  * `test_citation_traverse_chain_mode_equivalent_to_old_citation_chain` — patch `search.citation_chain` and verify `citation_traverse(mode="chain", ...)` returns the same result as `citation_chain(...)`.
  * `test_citation_traverse_default_mode_is_graph` — when `mode` is omitted, behaves as `mode="graph"`.
  * `test_citation_graph_alias_still_works` — calling old name `citation_graph` shims to `citation_traverse(mode="graph")` and returns the deprecation envelope.
  * `test_citation_chain_alias_still_works` — same for `citation_chain` -> `citation_traverse(mode="chain")`.
  * `test_find_similar_by_examples_returns_removed_error` — calling the name dispatches to a clear error response.
  * `test_list_tools_count_is_15` — `list_tools()` after consolidation returns exactly 15 tools.
  * `test_list_tools_contains_citation_traverse` and `test_list_tools_does_not_contain_old_citation_tools`.
* Update `tests/test_mcp_smoke.py`:
  * Switch the count assertions from 16 to 15.
  * Replace `test_citation_graph` and `test_citation_chain` with `test_citation_traverse_graph_mode` and `test_citation_traverse_chain_mode`.
  * Drop `test_every_expected_tool_has_a_smoke_test` reliance on names that no longer exist (or update it to ignore the renamed tools).
* `tests/test_scix_deep_search.py:test_persona_lists_15_tools` — the persona file lists `mcp__scix__citation_chain` and `mcp__scix__citation_graph`. We need to update the persona to use `mcp__scix__citation_traverse` (one tool replacing two), bringing the persona total from 15 to 14 — which would break that test unless we also add `mcp__scix__section_retrieval` (or add `mcp__scix__claim_blame`/`find_replications` is already there). Inspecting the persona front-matter shows: 13 listed tools today including `citation_chain` and `citation_graph` and the two MH-4 tools, totaling 15. After consolidation, we drop two and add `citation_traverse` => 14. To keep the test green AND reflect the new tool, add `mcp__scix__section_retrieval` => back to 15. This update is in scope (it's part of "update SKILL/persona to reflect the consolidated set").

## File layout for the implementation

* `src/scix/mcp_server.py`:
  * Add new `Tool(name="citation_traverse", ...)` registration with `mode` enum.
  * Remove `Tool(name="citation_graph", ...)` and `Tool(name="citation_chain", ...)` registrations.
  * Remove the conditional `Tool(name="find_similar_by_examples", ...)` registration block.
  * Add `_handle_citation_traverse(conn, args)` handler that branches on `mode`.
  * Wire `_dispatch_consolidated`: add `if name == "citation_traverse"`, remove the inline `citation_chain` block, route old `citation_graph` through the alias path.
  * Replace `if name == "find_similar_by_examples"` dispatch with a `removed_in` error.
  * Extend `_DEPRECATED_ALIASES` with `citation_graph -> citation_traverse` and `citation_chain -> citation_traverse`.
  * Extend `_transform_deprecated_args` with `mode` injection for both old names.
  * Update `EXPECTED_TOOLS` tuple: drop `citation_graph`, `citation_chain`; add `citation_traverse`. Keep `claim_blame`, `find_replications`, `section_retrieval`. Final = 15.
  * Empty `_OPTIONAL_TOOLS` (or remove it entirely).
  * Update `TOOL_TIMEOUTS`: keep `citation_graph` and `citation_chain` for legacy callers; add `citation_traverse` (defaulting to max of the two = 20s).
  * Update module docstring "15 consolidated tools" + "28 original tools -> 13 agent-facing tools" -> "33 historical tools -> 15 agent-facing tools" (or keep wording deliberately vague — the precise counts have shifted multiple times).
* `tests/test_mcp_tool_consolidation.py`: NEW — see test plan above.
* `tests/test_mcp_smoke.py`: UPDATE counts and citation-tool smoke tests.
* `tests/test_scix_deep_search.py`: NO change to the test file. The persona file (the data the test reads) is the thing to update.
* `.claude/agents/deep_search_investigator.md` (in repo): UPDATE the `tools:` list — drop `mcp__scix__citation_graph` and `mcp__scix__citation_chain`, add `mcp__scix__citation_traverse` and `mcp__scix__section_retrieval`. Total stays at 15.
* `/home/ds/.claude/skills/scix-mcp/SKILL.md`: REWRITE the "Tool Overview" table for the 15-tool roster + a "Deprecated" section.
* `docs/prd/artifacts/mcp_tool_audit.md`: APPEND an "Audit follow-up (2026-04-25)" section noting the 3 missed tools and the find_similar_by_examples retirement.

## Risks / non-goals

* The audit's H3 specifies `mode: neighbors|path` but the work unit's enum is `graph|chain`. Following the work-unit verdict.
* Not changing `_handle_citation_graph` / inline citation_chain dispatch beyond what's required to wire the new `citation_traverse` entry — this keeps blast radius small.
* No changes to underlying `search.py` functions — `citation_traverse` is a thin router.
