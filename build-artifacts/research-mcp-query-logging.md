# Research: MCP Query Logging

## Key Findings

### MCP Server Architecture

- `src/scix/mcp_server.py` — single file, ~1167 lines
- 22 tools registered via `@server.list_tools()` and `@server.call_tool()`
- `_dispatch_tool(conn, name, args)` is the central routing function (line 793)
- Already tracks elapsed time with `time.monotonic()` (line 795, 1137)
- `_get_conn()` context manager provides DB connections (line 74)
- `call_tool()` async handler (line 779) wraps `_dispatch_tool` with connection + timeout

### call_tool handler (lines 778-783)

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    with _get_conn() as conn:
        _set_timeout(conn, name)
        result_json = _dispatch_tool(conn, name, arguments)
        return [TextContent(type="text", text=result_json)]
```

### Logging insertion point

Best place to add logging: inside `call_tool()`, wrapping the `_dispatch_tool()` call.
This captures ALL tool calls in one place. Use the same connection for logging.

### Existing migrations

- 001-014 exist (013_entity_dictionary.sql, 014_discipline_and_indexes.sql)
- New migration: `013_query_log.sql` (different name, no conflict per instructions)

### Test patterns

- `tests/test_mcp_server.py` — unit tests with `MagicMock` for conn, `@patch` for search functions
- Import `_dispatch_tool`, `_parse_filters`, `_result_to_json` directly
- E2E tests skip if DB or MCP SDK unavailable
- Tests use `json.loads()` to verify results

### Entity types for analysis

- `entity_search` tool has `entity_type` param: methods, datasets, instruments, materials
- Other tools query by bibcode, author_name, query terms, concept names
