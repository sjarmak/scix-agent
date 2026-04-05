-- 022_staging_entities.sql
-- Staging tables for entity graph: entities, entity_identifiers, entity_aliases.
-- Follows the staging pattern from 015_staging_schema.sql.
-- Pipeline writes land in staging.*, then staging.promote_entities() batch-upserts
-- into public.* and truncates staging.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Public entity tables (promote targets)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.entities (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    discipline TEXT,
    source TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (canonical_name, entity_type, source)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON public.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_source ON public.entities(source);

CREATE TABLE IF NOT EXISTS public.entity_identifiers (
    entity_id INT NOT NULL REFERENCES public.entities(id),
    id_scheme TEXT NOT NULL,
    external_id TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false,
    PRIMARY KEY (id_scheme, external_id)
);

CREATE TABLE IF NOT EXISTS public.entity_aliases (
    entity_id INT NOT NULL REFERENCES public.entities(id),
    alias TEXT NOT NULL,
    alias_source TEXT,
    PRIMARY KEY (entity_id, alias)
);

-- ---------------------------------------------------------------------------
-- 2. Staging entity tables (no FK enforcement)
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.entities (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    discipline TEXT,
    source TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (canonical_name, entity_type, source)
);

CREATE TABLE IF NOT EXISTS staging.entity_identifiers (
    entity_id INT NOT NULL,
    id_scheme TEXT NOT NULL,
    external_id TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false,
    PRIMARY KEY (id_scheme, external_id)
);

CREATE TABLE IF NOT EXISTS staging.entity_aliases (
    entity_id INT NOT NULL,
    alias TEXT NOT NULL,
    alias_source TEXT,
    PRIMARY KEY (entity_id, alias)
);

-- ---------------------------------------------------------------------------
-- 3. promote_entities() — atomic batch upsert from staging to public
-- ---------------------------------------------------------------------------
-- Promotes all 3 tables in a single call:
--   1. Upsert entities (ON CONFLICT updates properties + updated_at)
--   2. Upsert identifiers (remaps staging entity_id -> public entity_id
--      via canonical_name + entity_type + source natural key)
--   3. Upsert aliases (same remapping, ON CONFLICT DO NOTHING)
--   4. Truncate all staging tables
-- Returns the number of promoted entities.

CREATE OR REPLACE FUNCTION staging.promote_entities()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    promoted_count INTEGER;
BEGIN
    -- 1. Upsert entities
    WITH upserted AS (
        INSERT INTO public.entities
            (canonical_name, entity_type, discipline, source, properties, created_at, updated_at)
        SELECT canonical_name, entity_type, discipline, source, properties, created_at, updated_at
        FROM staging.entities
        ON CONFLICT (canonical_name, entity_type, source)
        DO UPDATE SET
            properties = EXCLUDED.properties,
            updated_at = NOW()
        RETURNING 1
    )
    SELECT count(*) INTO promoted_count FROM upserted;

    -- 2. Upsert identifiers (remap entity_id through natural key)
    INSERT INTO public.entity_identifiers (entity_id, id_scheme, external_id, is_primary)
    SELECT pe.id, si.id_scheme, si.external_id, si.is_primary
    FROM staging.entity_identifiers si
    JOIN staging.entities se ON se.id = si.entity_id
    JOIN public.entities pe ON pe.canonical_name = se.canonical_name
                            AND pe.entity_type = se.entity_type
                            AND pe.source = se.source
    ON CONFLICT (id_scheme, external_id)
    DO UPDATE SET
        entity_id = EXCLUDED.entity_id,
        is_primary = EXCLUDED.is_primary;

    -- 3. Upsert aliases (remap entity_id through natural key)
    INSERT INTO public.entity_aliases (entity_id, alias, alias_source)
    SELECT pe.id, sa.alias, sa.alias_source
    FROM staging.entity_aliases sa
    JOIN staging.entities se ON se.id = sa.entity_id
    JOIN public.entities pe ON pe.canonical_name = se.canonical_name
                            AND pe.entity_type = se.entity_type
                            AND pe.source = se.source
    ON CONFLICT (entity_id, alias)
    DO NOTHING;

    -- 4. Clear staging tables
    TRUNCATE staging.entity_aliases;
    TRUNCATE staging.entity_identifiers;
    TRUNCATE staging.entities;

    RETURN promoted_count;
END;
$$;

COMMIT;
