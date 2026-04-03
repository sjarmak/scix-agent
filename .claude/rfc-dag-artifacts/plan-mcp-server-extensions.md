# Plan: MCP Server Extensions — Entity + Session Tools

## Overview

Add 7 new MCP tools to `src/scix/mcp_server.py` and integrate `SessionState` from `session.py`.

## Changes

### 1. Module-level SessionState singleton

- Import `SessionState` and `WorkingSetEntry` from `scix.session`
- Create `_session_state = SessionState()` at module level

### 2. New Tools (7)

| Tool                  | Params                                                                        | SQL/Logic                                                                                 |
| --------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `entity_search`       | entity_type, entity_name, limit                                               | `SELECT bibcode, payload FROM extractions WHERE payload @> %s LIMIT %s` using GIN index   |
| `entity_profile`      | bibcode                                                                       | `SELECT extraction_type, extraction_version, payload FROM extractions WHERE bibcode = %s` |
| `add_to_working_set`  | bibcodes (list), metadata (source_tool, source_context, relevance_hint, tags) | Delegates to `_session_state.add_to_working_set()` for each bibcode                       |
| `get_working_set`     | (none)                                                                        | Returns `_session_state.get_working_set()`                                                |
| `get_session_summary` | (none)                                                                        | Returns `_session_state.get_session_summary()`                                            |
| `find_gaps`           | limit, resolution                                                             | SQL: papers in unexplored communities that cite working set papers                        |
| `clear_working_set`   | (none)                                                                        | Delegates to `_session_state.clear_working_set()`                                         |

### 3. in_working_set annotation

- Add helper `_annotate_working_set(papers)` that adds `in_working_set: bool` to each paper dict
- Apply to all paper-returning tools in `_result_to_json()` by checking for `papers` key

### 4. find_gaps SQL

```sql
SELECT DISTINCT p.bibcode, p.title, pm.pagerank, pm.community_id_{resolution}
FROM citation_edges ce
JOIN papers p ON p.bibcode = ce.citing
JOIN paper_metrics pm ON pm.bibcode = p.bibcode
WHERE ce.cited IN (working_set_bibcodes)
  AND pm.community_id_{resolution} NOT IN (
    SELECT DISTINCT pm2.community_id_{resolution}
    FROM paper_metrics pm2
    WHERE pm2.bibcode IN (working_set_bibcodes)
  )
  AND p.bibcode NOT IN (working_set_bibcodes)
ORDER BY pm.pagerank DESC
LIMIT %s
```

### 5. Tests (tests/test_mcp_session.py)

- Mock `psycopg.connect` and cursor
- Test each of the 7 tools via `_dispatch_tool`
- Test `in_working_set` annotation
- At least 10 tests total

## Files Modified

- `src/scix/mcp_server.py` — add imports, singleton, 7 tool registrations, 7 dispatch branches, annotation helper
- `tests/test_mcp_session.py` — new file with 10+ tests
