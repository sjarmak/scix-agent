# Plan: MCP Query Logging

## Step 1: Create migration `migrations/013_query_log.sql`

- CREATE TABLE `query_log` with columns: id (SERIAL PK), tool_name (TEXT NOT NULL), params_json (JSONB), latency_ms (REAL), success (BOOLEAN NOT NULL), error_msg (TEXT), created_at (TIMESTAMPTZ DEFAULT now())
- Add indexes on tool_name and created_at for analysis queries

## Step 2: Add logging function to `src/scix/mcp_server.py`

- Add `_log_query(conn, tool_name, params, latency_ms, success, error_msg)` function
- Uses a separate connection (not the tool's transactional conn) to ensure logging survives tool failures
- Or use the same conn with autocommit for the INSERT

### Decision: Use a separate one-off connection for logging

- The tool connection may have SET LOCAL statement_timeout, and if the tool fails, the connection state is uncertain
- A fire-and-forget INSERT on a fresh connection is safest
- Actually, simpler: just do the INSERT on the same conn after the tool completes, in a try/except. The conn from `_get_conn()` is autocommit by default in psycopg3.
- Best approach: wrap `_dispatch_tool` call in `call_tool()`, catch exceptions, log success/failure, then INSERT.

## Step 3: Modify `call_tool()` in `create_server()`

- Wrap `_dispatch_tool` in try/except
- Measure latency
- After dispatch (success or failure), INSERT into query_log
- On logging failure, log warning but don't fail the tool call

## Step 4: Create `scripts/analyze_query_log.py`

- Connect to DB, run 3 analysis queries:
  1. `top_queries`: Top 50 entity queries (by tool_name + params frequency)
  2. `failure_rate_by_tool`: failure count / total count per tool
  3. `entity_type_requests`: distribution of entity_type values from entity_search params
- Output as JSON to stdout

## Step 5: Write tests in `tests/test_query_logging.py`

- Test migration DDL (mock or check table creation SQL)
- Test that 3 different tool calls produce 3 rows in query_log
- Test that the analysis script produces correct JSON keys
- Mock DB connections following existing patterns

## Step 6: Run tests, fix failures
