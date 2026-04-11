# Test Results: consolidate-mcp-tools

## Test Run: 2026-04-11

### Primary test files (55 tests)

- `tests/test_mcp_server.py`: 42 passed (including list_tools AC1)
- `tests/test_mcp_session.py`: 13 passed

### Related test files (65 tests)

- `tests/test_mcp_integration.py`: all passed (get_author_papers via deprecated alias)
- `tests/test_mcp_citation_context.py`: all passed (get_citation_context via deprecated alias)
- `tests/test_mcp_paper_tools.py`: all passed (document_context, entity_context, resolve_entity via aliases)

### Total: 120 passed, 0 failed

### E2E tests (test_mcp_e2e.py): 15 errors (PRE-EXISTING)

- All errors are `AttributeError: 'Server' object has no attribute 'list_tools_handlers'`
- This is a MCP SDK version mismatch (old test uses removed API)
- Not caused by this refactor

## Acceptance Criteria Coverage

| AC  | Description                                                             | Status                                      |
| --- | ----------------------------------------------------------------------- | ------------------------------------------- |
| 1   | list_tools() returns exactly 13 tools                                   | PASS                                        |
| 2   | search(mode='hybrid') dispatches to hybrid_search                       | PASS                                        |
| 3   | search(mode='semantic') dispatches to vector_search                     | PASS                                        |
| 4   | search(mode='keyword') dispatches to lexical_search                     | PASS                                        |
| 5   | citation_graph(direction='forward') -> get_citations shape              | PASS                                        |
| 6   | citation_graph(direction='backward') -> get_references shape            | PASS                                        |
| 7   | citation_similarity(method='co_citation') -> co_citation_analysis shape | PASS                                        |
| 8   | entity(action='search', entity_type='methods') dispatches               | PASS                                        |
| 9   | entity(entity_type='entities') returns validation error                 | PASS                                        |
| 10  | entity(action='resolve') dispatches                                     | PASS                                        |
| 11  | entity_context(entity_id=1) dispatches                                  | PASS                                        |
| 12  | get_paper(include_entities=false) returns metadata only                 | PASS                                        |
| 13  | get_paper(include_entities=true) routes to document_context             | PASS                                        |
| 14  | find_gaps reads from implicit session state                             | PASS                                        |
| 15  | find_gaps(clear_first=true) resets focused set                          | PASS                                        |
| 16  | Session tools NOT in list_tools                                         | PASS                                        |
| 17  | 'semantic_search' returns deprecated:true + use_instead:'search'        | PASS                                        |
| 18  | 'get_citations' returns deprecated:true + use_instead:'citation_graph'  | PASS                                        |
| 19  | All deprecated aliases log original tool name                           | PASS                                        |
| 20  | New tests verify 3+ consolidated tools                                  | PASS (search, citation_graph, entity, etc.) |
| 21  | Existing passing tests still pass                                       | PASS (120/120)                              |
