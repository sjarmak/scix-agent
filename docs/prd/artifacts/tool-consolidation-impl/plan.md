# Plan — tool-consolidation-impl

## Per-tool action

| Tool | Action | Detail |
|---|---|---|
| search | KEEP | unchanged |
| concept_search | KEEP | unchanged |
| get_paper | KEEP | unchanged |
| read_paper | KEEP | unchanged |
| **citation_graph** | MERGE -> citation_traverse | shim via `_DEPRECATED_ALIASES` and `_transform_deprecated_args` (inject `mode="graph"`). Drop `Tool(...)` registration from `list_tools()`. |
| **citation_chain** | MERGE -> citation_traverse | shim via `_DEPRECATED_ALIASES` and `_transform_deprecated_args` (inject `mode="chain"`). Drop `Tool(...)` registration from `list_tools()`. |
| citation_similarity | KEEP | unchanged |
| entity | KEEP | unchanged |
| entity_context | KEEP | unchanged |
| graph_context | KEEP | unchanged |
| find_gaps | KEEP | unchanged |
| temporal_evolution | KEEP | unchanged |
| facet_counts | KEEP | unchanged |
| claim_blame | KEEP | out-of-audit-scope KEEP per work-unit prompt |
| find_replications | KEEP | out-of-audit-scope KEEP per work-unit prompt |
| **find_similar_by_examples** | REMOVE | Drop registration, dispatch raises clear "removed in 2026-04-25" error. Qdrant-gated and not in active use. |
| section_retrieval | KEEP | already added by sibling unit |
| **citation_traverse** | NEW | mode enum: graph (default) | chain |

Final active tool count = 15.

## `citation_traverse` signature

```jsonschema
{
  "type": "object",
  "properties": {
    "mode": {
      "type": "string",
      "enum": ["graph", "chain"],
      "default": "graph",
      "description": "graph: walk neighbors of a single bibcode (citing or cited papers). chain: trace shortest path between source and target bibcodes."
    },
    "bibcode": {"type": "string", "description": "ADS bibcode (required when mode='graph')"},
    "direction": {
      "type": "string",
      "enum": ["forward", "backward", "both"],
      "default": "forward",
      "description": "forward=citing papers, backward=references, both=all (mode='graph' only)"
    },
    "include_context": {
      "type": "boolean",
      "default": false,
      "description": "Include citation context text (mode='graph' only, slower)"
    },
    "source_bibcode": {"type": "string", "description": "Path start (required when mode='chain')"},
    "target_bibcode": {"type": "string", "description": "Path end (required when mode='chain')"},
    "max_depth": {
      "type": "integer",
      "default": 5,
      "description": "Maximum path length 1..5 (mode='chain' only)"
    },
    "limit": {"type": "integer", "default": 20, "description": "Max neighbors (mode='graph' only)"}
  }
}
```

Note: `required` is intentionally empty at the top level. The handler validates per-mode and returns a structured `error` JSON for missing args. JSON-Schema `oneOf` per the audit's H3 would be cleaner but is not strictly required — runtime validation is sufficient and keeps the schema readable in MCP listings.

## `_handle_citation_traverse` dispatch logic

```
def _handle_citation_traverse(conn, args):
    mode = args.get("mode", "graph")
    if mode == "graph":
        if "bibcode" not in args:
            return error("bibcode is required when mode='graph'")
        return _handle_citation_graph(conn, args)  # existing handler, unchanged
    if mode == "chain":
        if "source_bibcode" not in args or "target_bibcode" not in args:
            return error("source_bibcode and target_bibcode are required when mode='chain'")
        max_depth = clamp(args.get("max_depth", 5), 1, 5)
        result = search.citation_chain(conn, args["source_bibcode"], args["target_bibcode"], max_depth=max_depth)
        return _result_to_json(result)
    return error(f"Invalid mode: {mode}. Use 'graph' or 'chain'.")
```

## `_DEPRECATED_ALIASES` additions

```
"citation_graph": "citation_traverse",
"citation_chain": "citation_traverse",
```

Note: the existing `_DEPRECATED_ALIASES` already maps several old names to `citation_graph` (`get_citations`, `get_references`, `get_citation_context`). Those entries must be remapped to `citation_traverse` to avoid double-hop deprecation. Update those values too.

## `_transform_deprecated_args` additions

```
if old_name == "citation_graph":
    new_args["mode"] = "graph"
    return "citation_traverse", new_args
if old_name == "citation_chain":
    new_args["mode"] = "chain"
    return "citation_traverse", new_args
```

For the existing aliases that previously routed to `citation_graph`:

```
if old_name == "get_citations":
    new_args["direction"] = "forward"
    new_args["mode"] = "graph"
    return "citation_traverse", new_args
# similar for get_references and get_citation_context
```

## `find_similar_by_examples` removal

* Delete the `if _qdrant_enabled():` block in `list_tools()` (lines 1839-1882).
* Delete the dispatch entry in `_dispatch_consolidated`:
  * Replace `if name == "find_similar_by_examples": return _handle_find_similar_by_examples(args)` with:
    ```
    if name == "find_similar_by_examples":
        return json.dumps({
            "error": "tool_removed",
            "removed_in": "2026-04-25",
            "message": "find_similar_by_examples was retired in 2026-04-25 (Qdrant backend not in active use).",
        })
    ```
* Empty `_OPTIONAL_TOOLS` to `()` so `_expected_tool_set()` matches `EXPECTED_TOOLS` exactly.
* Keep the `_handle_find_similar_by_examples` private function for now — unreachable but cheap to retain in case Qdrant comes back.

## `EXPECTED_TOOLS` final

```python
EXPECTED_TOOLS: tuple[str, ...] = (
    "search",
    "concept_search",
    "get_paper",
    "read_paper",
    "citation_traverse",
    "citation_similarity",
    "entity",
    "entity_context",
    "graph_context",
    "find_gaps",
    "temporal_evolution",
    "facet_counts",
    "claim_blame",
    "find_replications",
    "section_retrieval",
)
```

15 entries.

## TOOL_TIMEOUTS

Add `"citation_traverse": float(os.environ.get("SCIX_TIMEOUT_TRAVERSE", "20"))`. Keep `citation_graph` and `citation_chain` entries for the deprecated-alias path (they're the names `_set_timeout` is called with for old callers that haven't migrated).

## Test plan (matches research doc)

New file `tests/test_mcp_tool_consolidation.py`:
1. `test_citation_traverse_graph_mode_calls_get_citations` — mock `search.get_citations`, dispatch `citation_traverse(mode="graph", bibcode=..., direction="forward")`, assert mock was called with the right args.
2. `test_citation_traverse_default_mode_is_graph` — same as (1) but without explicit `mode`.
3. `test_citation_traverse_chain_mode_calls_citation_chain` — mock `search.citation_chain`, dispatch `citation_traverse(mode="chain", source_bibcode=..., target_bibcode=..., max_depth=3)`.
4. `test_citation_traverse_invalid_mode_returns_error` — `mode="bogus"` returns JSON `error`.
5. `test_citation_traverse_graph_missing_bibcode_returns_error`.
6. `test_citation_traverse_chain_missing_endpoints_returns_error`.
7. `test_citation_graph_alias_routes_to_traverse_graph_mode` — call old name `citation_graph`, mock `search.get_citations`, assert it gets called and the response carries `deprecated: true`, `use_instead: "citation_traverse"`.
8. `test_citation_chain_alias_routes_to_traverse_chain_mode` — symmetric.
9. `test_find_similar_by_examples_returns_removed_error` — `_dispatch_tool(conn, "find_similar_by_examples", {})` returns JSON with `error: tool_removed`.
10. `test_list_tools_count_is_15` — fresh server has exactly 15 tools.
11. `test_list_tools_contains_citation_traverse_not_old_citation_tools` — registered names include `citation_traverse` but not `citation_graph` or `citation_chain` or `find_similar_by_examples`.
12. `test_expected_tools_count_is_15` — `EXPECTED_TOOLS` has 15 entries.

Update `tests/test_mcp_smoke.py`:
* `test_expected_tools_has_16_entries` -> `test_expected_tools_has_15_entries` and assert `len(EXPECTED_TOOLS) == 15`.
* `test_self_test_passes_on_fresh_server` — change `tool_count == 16` -> 15.
* Change the bad-tools count fixture from 15-but-expected-16 to 14-but-expected-15.
* Replace `test_citation_graph` smoke test with `test_citation_traverse` (graph mode default).
* Drop `test_citation_chain` smoke test (now covered by consolidation test file). Rename to `test_citation_traverse_chain` to keep `test_every_expected_tool_has_a_smoke_test` happy or fold into one parametrised test.
* The `test_every_expected_tool_has_a_smoke_test` checks `set(EXPECTED_TOOLS) - smoke_test_methods`. Keeping a `test_citation_traverse` method covers `citation_traverse`. We need to drop the now-redundant `test_citation_graph` and `test_citation_chain` methods.

Update `.claude/agents/deep_search_investigator.md`:
* Drop `mcp__scix__citation_graph` and `mcp__scix__citation_chain` lines.
* Add `mcp__scix__citation_traverse` and `mcp__scix__section_retrieval` lines.
* Total stays 15. The `test_persona_lists_15_tools` test continues to pass.

Update `/home/ds/.claude/skills/scix-mcp/SKILL.md`:
* Replace "Tool Overview (13 tools)" with "Tool Overview (15 tools)".
* Rewrite the table:
  * Search: search, concept_search
  * Paper access: get_paper, read_paper, section_retrieval
  * Citation graph: citation_traverse, citation_similarity
  * Entity system: entity, entity_context
  * Structure: graph_context, find_gaps, temporal_evolution, facet_counts
  * Provenance / replication: claim_blame, find_replications
* Add a "Deprecated" subsection mapping `citation_graph` -> `citation_traverse(mode="graph")`, `citation_chain` -> `citation_traverse(mode="chain")`, `find_similar_by_examples` -> retired.
* Update workflow examples that referenced `citation_chain(direction=...)` (which wasn't actually that tool's signature anyway) to `citation_traverse(mode="graph", direction="forward")`.

Update `docs/prd/artifacts/mcp_tool_audit.md`:
* Append "## Audit follow-up (2026-04-25)" section listing the 3 missed tools (`claim_blame`, `find_replications`, `find_similar_by_examples`), the verdict for each, the chosen consolidation strategy (shim, not remove), the find_similar_by_examples retirement, and the final active tool count of 15.

## Acceptance verification

Run after implementation:
```
awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py | grep -oE 'name="[a-z_]+"' | sort -u | wc -l
# expect: 15

python -m pytest tests/test_mcp_tool_consolidation.py -v
python -m pytest tests/test_mcp_smoke.py -v
python -m pytest tests/test_scix_deep_search.py -v
# all expect: exit 0
```
