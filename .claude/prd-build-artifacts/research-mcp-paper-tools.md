# Research: mcp-paper-tools

## Key Findings

### SearchResult pattern (search.py)

- `SearchResult(frozen=True)`: papers (list[dict]), total (int), timing_ms (dict[str,float]), metadata (dict)
- Functions take `conn: psycopg.Connection` as first arg, return SearchResult
- Use `_elapsed_ms(t0)` for timing, `time.perf_counter()` for start
- Use `dict_row` cursor for queries

### section_parser.py

- `parse_sections(body: str) -> list[tuple[str, int, int, str]]` - (name, start, end, text)
- Returns `[("full", 0, len(body), body)]` if no section headers found
- Section names: introduction, methods, observations, data, results, discussion, conclusions, summary, acknowledgments, references, preamble, full

### MCP server patterns

- Tools registered in `list_tools()` as `Tool(name, description, inputSchema)`
- Dispatch via `_dispatch_tool(conn, name, args)` with elif chain
- `TOOL_TIMEOUTS` dict for per-tool timeouts
- `_result_to_json()` serializes SearchResult to JSON
- `_get_conn()` context manager for DB connections

### DB schema

- `papers` table has `body TEXT` column (migration 010)
- `papers` table has `abstract`, `bibcode`, `title`, etc.

### Test patterns

- Mock DB with `MagicMock()`
- Patch `scix.search.*` functions
- Test dispatch returns correct JSON structure
