-- 015_staging_schema.sql
-- Staging schema for extraction pipeline writes.
-- Isolates write-heavy extraction workloads from read-heavy MCP queries
-- by directing pipeline writes to staging.extractions, then batch-promoting
-- canonical results to public.extractions via staging.promote_extractions().

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Create staging schema
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS staging;

-- ---------------------------------------------------------------------------
-- 2. Create staging.extractions mirroring public.extractions structure
-- ---------------------------------------------------------------------------
-- No FK to papers: staging data may reference bibcodes not yet in public.

CREATE TABLE IF NOT EXISTS staging.extractions (
    id SERIAL PRIMARY KEY,
    bibcode TEXT NOT NULL,
    extraction_type TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_staging_extractions_bibcode_type_version
        UNIQUE (bibcode, extraction_type, extraction_version)
);

CREATE INDEX IF NOT EXISTS idx_staging_extractions_bibcode
    ON staging.extractions(bibcode);

CREATE INDEX IF NOT EXISTS idx_staging_extractions_type
    ON staging.extractions(extraction_type);

-- ---------------------------------------------------------------------------
-- 3. promote_extractions() — batch upsert from staging to public
-- ---------------------------------------------------------------------------
-- Uses the unique constraint (bibcode, extraction_type, extraction_version)
-- on public.extractions for ON CONFLICT upsert. After promotion, staging
-- is truncated.

CREATE OR REPLACE FUNCTION staging.promote_extractions()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    promoted_count INTEGER;
BEGIN
    -- Upsert from staging into public
    WITH upserted AS (
        INSERT INTO public.extractions (bibcode, extraction_type, extraction_version, payload, created_at)
        SELECT bibcode, extraction_type, extraction_version, payload, created_at
        FROM staging.extractions
        ON CONFLICT (bibcode, extraction_type, extraction_version)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            created_at = EXCLUDED.created_at
        RETURNING 1
    )
    SELECT count(*) INTO promoted_count FROM upserted;

    -- Clear staging after successful promotion
    TRUNCATE staging.extractions;

    RETURN promoted_count;
END;
$$;

COMMIT;
