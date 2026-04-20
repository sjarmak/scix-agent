-- 019_schema_migrations.sql
-- Schema migration tracking table

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    filename TEXT NOT NULL
);

COMMIT;
