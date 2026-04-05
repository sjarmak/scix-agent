-- 016_query_log.sql
-- Query logging for MCP tool calls
-- Note: This duplicates 013_query_log.sql with IF NOT EXISTS guards for safety.

BEGIN;

CREATE TABLE IF NOT EXISTS query_log (
    id SERIAL PRIMARY KEY,
    tool_name TEXT NOT NULL,
    params_json JSONB,
    latency_ms REAL,
    success BOOLEAN NOT NULL,
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_log_tool_name ON query_log(tool_name);
CREATE INDEX IF NOT EXISTS idx_query_log_created_at ON query_log(created_at);

COMMIT;
