-- migration: 049_staging_ner_extractions — M4 NER staging schema
--
-- Extends the migration-015 staging schema with the tables required to
-- land NER extractions from the M4 pipeline before canonicalising them
-- into public.  Two new surfaces:
--
--   1. Provenance columns on staging.extractions (source, confidence_tier)
--      so the NER pipeline can record which producer emitted each row and
--      how confident we were about it.  extraction_version and created_at
--      already exist from migration 015; we add them defensively.
--
--   2. staging.extraction_entity_links — a LIST-partitioned table keyed by
--      entity_type with dedicated partitions for software, instrument,
--      dataset, method and a DEFAULT catch-all.  The public counterpart
--      (public.extraction_entity_links) is created here too so there is a
--      promotion target.
--
-- SAFETY: every table created or touched here MUST remain LOGGED.  The DO
-- block at the end re-asserts this against pg_class.relpersistence — see
-- migration 023 and the feedback_unlogged_tables memory for the 32M
-- embedding loss that prompted this invariant.
--
-- Idempotent: CREATE SCHEMA IF NOT EXISTS, CREATE TABLE IF NOT EXISTS,
-- ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS, and partition
-- CREATE TABLE IF NOT EXISTS make the migration safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Schema (idempotent — already created in migration 015)
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS staging;

-- ---------------------------------------------------------------------------
-- 2. staging.extractions — mirror of public.extractions + provenance
-- ---------------------------------------------------------------------------
-- Migration 015 created this table; this block is a defensive re-create for
-- fresh databases that somehow skip 015.  The provenance ALTERs below add
-- the new columns on existing deployments.

CREATE TABLE IF NOT EXISTS staging.extractions (
    id                  SERIAL PRIMARY KEY,
    bibcode             TEXT NOT NULL,
    extraction_type     TEXT NOT NULL,
    extraction_version  TEXT NOT NULL,
    payload             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_staging_extractions_bibcode_type_version
        UNIQUE (bibcode, extraction_type, extraction_version)
);

ALTER TABLE staging.extractions
    ADD COLUMN IF NOT EXISTS source TEXT;
ALTER TABLE staging.extractions
    ADD COLUMN IF NOT EXISTS confidence_tier SMALLINT;

CREATE INDEX IF NOT EXISTS idx_staging_extractions_bibcode
    ON staging.extractions (bibcode);
CREATE INDEX IF NOT EXISTS idx_staging_extractions_type
    ON staging.extractions (extraction_type);
CREATE INDEX IF NOT EXISTS idx_staging_extractions_source
    ON staging.extractions (source);

-- ---------------------------------------------------------------------------
-- 3. staging.extraction_entity_links — partitioned by entity_type
-- ---------------------------------------------------------------------------
-- Partition key must be part of the primary key for LIST partitioning in
-- PostgreSQL, so the PK is (id, entity_type).

CREATE TABLE IF NOT EXISTS staging.extraction_entity_links (
    id                  BIGSERIAL,
    extraction_id       BIGINT,
    bibcode             TEXT        NOT NULL,
    entity_type         TEXT        NOT NULL,
    entity_id           INT,
    entity_surface      TEXT        NOT NULL,
    entity_canonical    TEXT,
    span_start          INT,
    span_end            INT,
    source              TEXT        NOT NULL,
    confidence_tier     SMALLINT    NOT NULL,
    confidence          REAL,
    extraction_version  TEXT        NOT NULL,
    payload             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, entity_type)
) PARTITION BY LIST (entity_type);

-- Parent-level indexes (inherited by all partitions).
CREATE INDEX IF NOT EXISTS idx_staging_eel_bibcode
    ON staging.extraction_entity_links (bibcode);
CREATE INDEX IF NOT EXISTS idx_staging_eel_source_tier_version
    ON staging.extraction_entity_links (source, confidence_tier, extraction_version);
CREATE INDEX IF NOT EXISTS idx_staging_eel_created_at
    ON staging.extraction_entity_links (created_at);

-- 3a. Named partitions — one per expected entity_type.
CREATE TABLE IF NOT EXISTS staging.extraction_entity_links_software
    PARTITION OF staging.extraction_entity_links
    FOR VALUES IN ('software');

CREATE TABLE IF NOT EXISTS staging.extraction_entity_links_instrument
    PARTITION OF staging.extraction_entity_links
    FOR VALUES IN ('instrument');

CREATE TABLE IF NOT EXISTS staging.extraction_entity_links_dataset
    PARTITION OF staging.extraction_entity_links
    FOR VALUES IN ('dataset');

CREATE TABLE IF NOT EXISTS staging.extraction_entity_links_method
    PARTITION OF staging.extraction_entity_links
    FOR VALUES IN ('method');

-- 3b. DEFAULT partition — catches unexpected entity_types so INSERTs do
--     not fail while the taxonomy is evolving.  Operators can detach and
--     promote-to-named-partition later.
CREATE TABLE IF NOT EXISTS staging.extraction_entity_links_default
    PARTITION OF staging.extraction_entity_links DEFAULT;

-- ---------------------------------------------------------------------------
-- 4. public.extractions — ensure unique constraint for ON CONFLICT
-- ---------------------------------------------------------------------------
-- Migration 009 added uq_extractions_bibcode_type_version, but some
-- rebuilt environments lack it.  The promotion script relies on it, so we
-- re-assert it here via NOT VALID + VALIDATE to stay idempotent on tables
-- that already have duplicates from legacy loads.  If the constraint is
-- already present the DO block is a no-op.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint c
          JOIN pg_class cl ON cl.oid = c.conrelid
          JOIN pg_namespace n ON n.oid = cl.relnamespace
         WHERE n.nspname = 'public'
           AND cl.relname = 'extractions'
           AND c.conname  = 'uq_extractions_bibcode_type_version'
    ) THEN
        BEGIN
            ALTER TABLE public.extractions
                ADD CONSTRAINT uq_extractions_bibcode_type_version
                UNIQUE (bibcode, extraction_type, extraction_version);
        EXCEPTION WHEN unique_violation THEN
            RAISE NOTICE
                'public.extractions has duplicate (bibcode, extraction_type, '
                'extraction_version) rows; skipping unique constraint. '
                'Promotion script will fail until duplicates are resolved.';
        END;
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 5. public.extraction_entity_links — promotion target
-- ---------------------------------------------------------------------------
-- Not partitioned at the public tier; volumes there are bounded by
-- canonicalisation.  UNIQUE constraint supports ON CONFLICT DO NOTHING in
-- the promotion script.

CREATE TABLE IF NOT EXISTS public.extraction_entity_links (
    id                  BIGSERIAL PRIMARY KEY,
    extraction_id       BIGINT,
    bibcode             TEXT        NOT NULL REFERENCES public.papers(bibcode),
    entity_type         TEXT        NOT NULL,
    entity_id           INT,
    entity_surface      TEXT        NOT NULL,
    entity_canonical    TEXT,
    span_start          INT,
    span_end            INT,
    source              TEXT        NOT NULL,
    confidence_tier     SMALLINT    NOT NULL,
    confidence          REAL,
    extraction_version  TEXT        NOT NULL,
    payload             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_public_eel_bibcode_type_surface_version_source
        UNIQUE (bibcode, entity_type, entity_surface, extraction_version, source)
);

CREATE INDEX IF NOT EXISTS idx_public_eel_bibcode
    ON public.extraction_entity_links (bibcode);
CREATE INDEX IF NOT EXISTS idx_public_eel_entity_type
    ON public.extraction_entity_links (entity_type);
CREATE INDEX IF NOT EXISTS idx_public_eel_entity_id
    ON public.extraction_entity_links (entity_id)
    WHERE entity_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 6. LOGGED invariant — refuse to leave the migration with an UNLOGGED
--    table.  A single UNLOGGED surface here could silently lose NER
--    extractions on the next postgres restart.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    bad_rel TEXT;
BEGIN
    SELECT c.nspname || '.' || c.relname
      INTO bad_rel
      FROM (
          SELECT n.nspname, cl.relname, cl.relpersistence
            FROM pg_class cl
            JOIN pg_namespace n ON n.oid = cl.relnamespace
           WHERE cl.relkind IN ('r', 'p')
             AND (
                 (n.nspname = 'staging' AND cl.relname IN (
                     'extractions',
                     'extraction_entity_links',
                     'extraction_entity_links_software',
                     'extraction_entity_links_instrument',
                     'extraction_entity_links_dataset',
                     'extraction_entity_links_method',
                     'extraction_entity_links_default'
                 ))
                 OR (n.nspname = 'public' AND cl.relname = 'extraction_entity_links')
             )
      ) c
      WHERE c.relpersistence <> 'p'
      LIMIT 1;

    IF bad_rel IS NOT NULL THEN
        RAISE EXCEPTION
            'Table % must be LOGGED (relpersistence=''p'') to protect NER '
            'extractions from crash loss — see migration 023 and the '
            'feedback_unlogged_tables memory',
            bad_rel;
    END IF;
END
$$;

COMMIT;
