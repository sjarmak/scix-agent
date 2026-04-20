-- 028_entity_schema_hardening.sql
-- M1: Harden the entity graph for tiered linking.
--   - New ENUMs for ambiguity_class and link_policy on entities.
--   - Add tier / tier_version to document_entities.
--   - Replace document_entities PK with (bibcode, entity_id, link_type, tier)
--     so the same (bibcode, entity, link_type) can exist at multiple tiers.
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. ENUM: entity_ambiguity_class
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    CREATE TYPE entity_ambiguity_class AS ENUM (
        'unique',
        'domain_safe',
        'homograph',
        'banned'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- ---------------------------------------------------------------------------
-- 2. ENUM: entity_link_policy
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    CREATE TYPE entity_link_policy AS ENUM (
        'open',
        'context_required',
        'llm_only',
        'banned'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- ---------------------------------------------------------------------------
-- 3. entities: add ambiguity_class and link_policy columns (nullable — legacy
--    rows predate the classification pass)
-- ---------------------------------------------------------------------------

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS ambiguity_class entity_ambiguity_class;

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS link_policy entity_link_policy;

-- ---------------------------------------------------------------------------
-- 4. document_entities: add tier / tier_version
--    tier semantics (u04 will document in detail):
--       0 = legacy / default
--       1 = high-precision (exact ID match, unique canonical)
--       2 = medium (alias + context)
--       3 = low / LLM-adjudicated
-- ---------------------------------------------------------------------------

ALTER TABLE document_entities
    ADD COLUMN IF NOT EXISTS tier SMALLINT NOT NULL DEFAULT 0;

ALTER TABLE document_entities
    ADD COLUMN IF NOT EXISTS tier_version INT NOT NULL DEFAULT 1;

-- ---------------------------------------------------------------------------
-- 5. Replace primary key with one that includes tier
--    (DROP + ADD inside DO-block for idempotency: if the PK already matches
--    the new shape, leave it alone)
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    current_pk_cols text;
BEGIN
    SELECT string_agg(a.attname, ',' ORDER BY array_position(c.conkey, a.attnum))
      INTO current_pk_cols
      FROM pg_constraint c
      JOIN pg_attribute a
        ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
     WHERE c.conrelid = 'public.document_entities'::regclass
       AND c.contype  = 'p';

    IF current_pk_cols IS DISTINCT FROM 'bibcode,entity_id,link_type,tier' THEN
        IF current_pk_cols IS NOT NULL THEN
            EXECUTE 'ALTER TABLE document_entities DROP CONSTRAINT document_entities_pkey';
        END IF;
        EXECUTE 'ALTER TABLE document_entities
                 ADD CONSTRAINT document_entities_pkey
                 PRIMARY KEY (bibcode, entity_id, link_type, tier)';
    END IF;
END
$$;

-- Helpful index for tier-scoped deletes/queries
CREATE INDEX IF NOT EXISTS idx_document_entities_tier
    ON document_entities(tier);

COMMIT;
