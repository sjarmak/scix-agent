-- 030_staging_and_promote_harvest.sql
-- D3: Public-schema staging tables for entity-graph harvests and a
-- promote_harvest(run_id) skeleton function.
--
-- Distinct from the `staging.*` schema created in 022_staging_entities.sql:
-- those are used by the existing entity pipeline; these are for the new
-- harvest-run-based promote flow (u04 will fill in the function body).
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. entities_staging — mirrors public.entities + staging_run_id
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities_staging (
    id               BIGSERIAL PRIMARY KEY,
    staging_run_id   BIGINT  NOT NULL,
    canonical_name   TEXT    NOT NULL,
    entity_type      TEXT    NOT NULL,
    discipline       TEXT,
    source           TEXT    NOT NULL,
    source_version   TEXT,
    ambiguity_class  TEXT,
    link_policy      TEXT,
    properties       JSONB   DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entities_staging_run
    ON entities_staging(staging_run_id);
CREATE INDEX IF NOT EXISTS idx_entities_staging_natural_key
    ON entities_staging(canonical_name, entity_type, source);

-- ---------------------------------------------------------------------------
-- 2. entity_aliases_staging
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_aliases_staging (
    id              BIGSERIAL PRIMARY KEY,
    staging_run_id  BIGINT NOT NULL,
    staging_entity_id BIGINT,  -- local FK into entities_staging.id (unenforced for COPY speed)
    canonical_name  TEXT,      -- natural-key fallback when staging_entity_id is null
    entity_type     TEXT,
    source          TEXT,
    alias           TEXT NOT NULL,
    alias_source    TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_staging_run
    ON entity_aliases_staging(staging_run_id);

-- ---------------------------------------------------------------------------
-- 3. entity_identifiers_staging
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_identifiers_staging (
    id              BIGSERIAL PRIMARY KEY,
    staging_run_id  BIGINT NOT NULL,
    staging_entity_id BIGINT,
    canonical_name  TEXT,
    entity_type     TEXT,
    source          TEXT,
    id_scheme       TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    is_primary      BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_entity_identifiers_staging_run
    ON entity_identifiers_staging(staging_run_id);

-- ---------------------------------------------------------------------------
-- 4. promote_harvest(run_id BIGINT) — stub
--
-- u04 will replace the body with an atomic batch upsert from *_staging into
-- the public entity tables, with tier assignment based on match quality, and
-- clean-up of the staging rows on success. For now this stub exists purely so
-- u01 can guarantee the function signature is present; it returns 0 and does
-- NOT raise, so downstream scaffolding can call it safely.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION promote_harvest(run_id BIGINT)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    staged_count INTEGER;
BEGIN
    SELECT count(*) INTO staged_count
      FROM entities_staging
     WHERE staging_run_id = run_id;

    -- Stub: u04 will implement the full upsert + tier assignment here.
    -- For now just return the number of staged rows so callers can assert
    -- the function exists and is wired to the staging table.
    RETURN COALESCE(staged_count, 0);
END
$$;

COMMENT ON FUNCTION promote_harvest(BIGINT) IS
    'STUB — u01. u04 will implement atomic promote of *_staging rows for '
    'staging_run_id into public entity tables with tier assignment.';

COMMIT;
