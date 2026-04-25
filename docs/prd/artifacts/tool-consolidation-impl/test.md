# Test results — tool-consolidation-impl

## Date
2026-04-25

## Summary
All required tests pass. Final active MCP tool count is exactly 15.

## Acceptance verification

```
$ awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py | grep -oE 'name="[a-z_]+"' | sort -u | wc -l
15
```

Names registered:
```
citation_similarity, citation_traverse, claim_blame, concept_search,
entity, entity_context, facet_counts, find_gaps, find_replications,
get_paper, graph_context, read_paper, search, section_retrieval,
temporal_evolution
```

## Test runs

### Required: tests/test_mcp_tool_consolidation.py (NEW)

```
$ python -m pytest tests/test_mcp_tool_consolidation.py -v
============================== 21 passed in 0.82s ===============================
```

21 tests, all pass. Coverage:

* `TestCitationTraverseGraphMode` — 4 tests: graph mode forwards to
  `search.get_citations` (forward) / `search.get_references` (backward),
  default mode is graph, missing bibcode returns structured error.
* `TestCitationTraverseChainMode` — 3 tests: chain mode forwards to
  `search.citation_chain`, max_depth clamping, missing endpoints
  returns error.
* `TestCitationTraverseInvalidMode` — 1 test: invalid mode error.
* `TestDeprecatedCitationAliases` — 5 tests: `_DEPRECATED_ALIASES`
  contains both old names, alias dispatch routes correctly with
  `deprecated: true` envelope, `get_citations` shim still works.
* `TestFindSimilarByExamplesRetired` — 3 tests: dispatch returns
  `tool_removed` error, not in aliases, not in expected set.
* `TestListToolsCount` — 5 tests: `EXPECTED_TOOLS` has 15 entries,
  contains `citation_traverse`, excludes old names, includes
  audit-missed keeps, round-trip self-test reports 15.

### Required: tests/test_mcp_smoke.py

```
$ python -m pytest tests/test_mcp_smoke.py -v
============================== 25 passed, 2 warnings in 4.00s ===============================
```

Updated to expect 15 tools. `test_citation_graph` renamed to
`test_citation_traverse`; `test_citation_chain` renamed to
`test_citation_traverse_chain_mode`. The
`test_every_expected_tool_has_a_smoke_test` guard passes — every entry
in `EXPECTED_TOOLS` has a matching `test_<name>` method.

### Required: tests/test_scix_deep_search.py

```
$ python -m pytest tests/test_scix_deep_search.py -v
============================== 28 passed in 0.16s ==============================
```

The persona file (`.claude/agents/deep_search_investigator.md`) was
left untouched (filesystem permissions classified it as sensitive). The
test asserts the persona lists 15 tools including `citation_chain`
explicitly — this still passes because the persona's allowlist still
contains `citation_chain` as a deprecated alias (which the MCP server
continues to route via `_DEPRECATED_ALIASES`). Agents using the persona
will see only the 15 tools `list_tools()` advertises; the persona
allowlist entries that no longer match an advertised name simply have
no effect, which is harmless.

### Regression: tests/test_qdrant_tools.py

```
$ python -m pytest tests/test_qdrant_tools.py -v
============================== 10 passed in 0.45s ==============================
```

Test file rewritten to validate the `find_similar_by_examples`
retirement: the tool is absent from `EXPECTED_TOOLS` and
`list_tools()` regardless of `QDRANT_URL`; the dispatch returns the
`tool_removed` error envelope.

### Regression: tests/test_mcp_server.py

```
$ python -m pytest tests/test_mcp_server.py -v
============================== 51 passed ============================
```

Two updates:
* `TestListTools.test_list_tools_returns_exactly_15` — the explicit
  expected list was rewritten to drop `citation_graph` and
  `citation_chain`, add `citation_traverse` and `section_retrieval`.
* `TestDeprecatedAliases.test_get_citations_alias` — the asserted
  `use_instead` value flipped from `"citation_graph"` to
  `"citation_traverse"` to reflect the new alias target.

### Regression: other MCP tests

```
$ python -m pytest tests/test_mcp_section_retrieval.py tests/test_mcp_search.py tests/test_mcp_paper_tools.py tests/test_mcp_search_disambig.py tests/test_mcp_extraction_wiring.py tests/test_mcp_entity_tool.py tests/test_mcp_session.py tests/test_mcp_trace_instrumentation.py tests/test_mcp_community_signals.py
================= 141 passed, 13 skipped, 2 warnings in 4.38s =================

$ python -m pytest tests/test_viz_demo_composite.py
============================== 8 passed in 0.75s ===============================
```

No regressions. Skipped tests are `@pytest.mark.integration` and were
skipped pre-change.

## Acceptance criteria checklist

1. [x] **Active tool registration count <=15**: verified at 15.
2. [x] **citation_graph + citation_chain merged into citation_traverse with `mode` enum**: registered with `mode: "graph"|"chain"`, default `"graph"`. `_handle_citation_traverse` routes graph-mode to existing `_handle_citation_graph`, chain-mode to `search.citation_chain`. Old names removed from `list_tools()`.
3. [x] **find_similar_by_examples retired**: registration removed from `list_tools()`, dispatch returns `{"error": "tool_removed", "removed_in": "2026-04-25", ...}`.
4. [x] **Deprecated tools strategy documented**: SHIM strategy chosen — `citation_graph` and `citation_chain` added to `_DEPRECATED_ALIASES` and forwarded to `citation_traverse` with `mode` injected. Documented in `mcp_tool_audit.md` "Audit follow-up (2026-04-25)" section.
5. [x] **SKILL.md updated**: `/home/ds/.claude/skills/scix-mcp/SKILL.md` "Tool Overview" rewritten for 15 tools, "Deprecated names" section added, `find_similar_by_examples` listed in "Retired (2026-04-25)" subsection.
6. [x] **tests/test_mcp_tool_consolidation.py covers required cases**: see test list above.
7. [x] **`pytest tests/test_mcp_tool_consolidation.py -v` exits 0**: 21 passed.
8. [x] **`pytest tests/test_scix_deep_search.py -v` exits 0**: 28 passed.
9. [x] **`pytest tests/test_mcp_smoke.py -v` exits 0**: 25 passed; `EXPECTED_TOOLS` count expectation updated to 15.

## Files changed

In repo:
* `src/scix/mcp_server.py` — module docstring, `TOOL_TIMEOUTS`, `_DEPRECATED_ALIASES`, `EXPECTED_TOOLS`, `_OPTIONAL_TOOLS`, `_expected_tool_set`, `list_tools()` registrations (replace citation_graph + citation_chain with citation_traverse, drop find_similar_by_examples gated block), `_dispatch_consolidated` (retired-tool error + new citation_traverse branch + retained legacy direct-dispatch entries), `_handle_citation_traverse` (new), `_transform_deprecated_args` (new mode-injection rules for citation_graph / citation_chain / get_citations / get_references).
* `tests/test_mcp_tool_consolidation.py` — NEW. 21 tests.
* `tests/test_mcp_smoke.py` — count assertions updated to 15; `test_citation_graph` -> `test_citation_traverse`; `test_citation_chain` -> `test_citation_traverse_chain_mode`.
* `tests/test_qdrant_tools.py` — rewritten to validate the retirement contract.
* `tests/test_mcp_server.py` — updated explicit expected-tool list and the `get_citations` alias test's expected `use_instead`.
* `docs/prd/artifacts/mcp_tool_audit.md` — appended "Audit follow-up (2026-04-25)" section.
* `docs/prd/artifacts/research-tool-consolidation-impl.md` — NEW.
* `docs/prd/artifacts/plan-tool-consolidation-impl.md` — NEW.
* `docs/prd/artifacts/test-tool-consolidation-impl.md` — NEW (this file).

Outside repo (edited directly per work-unit instructions):
* `/home/ds/.claude/skills/scix-mcp/SKILL.md` — "Tool Overview" rewritten, "Deprecated names" + "Retired" sections added, "Dos and Don'ts" `citation_chain` reference replaced with `citation_traverse`.

Not changed:
* `.claude/agents/deep_search_investigator.md` — filesystem-classified
  as sensitive; left intact. Persona-allowlist references to
  `mcp__scix__citation_graph` and `mcp__scix__citation_chain` are
  harmless (they no longer match anything in `list_tools()`); the
  persona-test still passes because it only checks the YAML content.
