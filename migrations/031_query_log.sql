-- 031_query_log.sql
-- M3.5.0: Extend the existing query_log table (created in 016_query_log.sql)
-- with the columns the new MCP instrumentation pass needs.
--
-- We ALTER the existing table rather than recreate it so historical rows and
-- the existing id sequence stay intact. The acceptance criterion for this
-- migration only requires that the new columns exist.
--
-- Idempotent: safe to re-run.

BEGIN;

-- New instrumentation columns (tool/query/result_count/session_id/is_test).
-- `ts` is an alias-style timestamp that defaults to now() so freshly inserted
-- rows are tagged without needing every call site to set it explicitly.
ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS ts TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS tool TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS query TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS result_count INT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS session_id TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_query_log_ts          ON query_log(ts);
CREATE INDEX IF NOT EXISTS idx_query_log_tool        ON query_log(tool);
CREATE INDEX IF NOT EXISTS idx_query_log_session_id  ON query_log(session_id);

COMMIT;
