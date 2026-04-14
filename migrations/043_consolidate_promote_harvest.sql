-- 043_consolidate_promote_harvest.sql
-- Consolidate promote_harvest: drop the migration-030 stub and install the
-- full v2 body as the canonical promote_harvest function.
--
-- Background: migration 030 shipped a promote_harvest(BIGINT) RETURNS INTEGER
-- stub.  The u04 work unit implemented the real logic as promote_harvest_v2()
-- (installed lazily by Python at runtime) to avoid colliding with the stub.
-- This migration makes promote_harvest the single canonical function and
-- removes promote_harvest_v2.
--
-- Idempotent: uses CREATE OR REPLACE and DROP IF EXISTS.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Drop the old stub (BIGINT -> INTEGER signature)
-- ---------------------------------------------------------------------------

DROP FUNCTION IF EXISTS promote_harvest(BIGINT);

-- ---------------------------------------------------------------------------
-- 2. Drop promote_harvest_v2 if it was previously installed by Python
-- ---------------------------------------------------------------------------

DROP FUNCTION IF EXISTS promote_harvest_v2(BIGINT, JSONB, NUMERIC, NUMERIC, INTEGER);

-- ---------------------------------------------------------------------------
-- 3. Install the canonical promote_harvest with full signature
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION promote_harvest(
    run_id            BIGINT,
    floors            JSONB   DEFAULT '{}'::jsonb,
    canonical_max     NUMERIC DEFAULT 0.02,
    alias_max         NUMERIC DEFAULT 0.05,
    orphan_threshold  INTEGER DEFAULT 1000
)
RETURNS JSONB
LANGUAGE plpgsql
AS $fn$
DECLARE
    result             JSONB := '{}'::jsonb;
    staging_total      BIGINT := 0;
    alias_staging_tot  BIGINT := 0;
    prod_entity_total  BIGINT := 0;
    prod_alias_total   BIGINT := 0;
    canonical_shrink   NUMERIC := 0;
    alias_shrink       NUMERIC := 0;
    floor_violations   JSONB := '[]'::jsonb;
    orphan_violations  JSONB := '[]'::jsonb;
    schema_errors      JSONB := '[]'::jsonb;
    per_source_json    JSONB := '{}'::jsonb;
    lock_acquired      BOOLEAN := FALSE;
    src_rec            RECORD;
    orphan_rec         RECORD;
    schema_rec         RECORD;
    n_promoted_ent     INTEGER := 0;
    n_promoted_ali     INTEGER := 0;
    n_promoted_ids     INTEGER := 0;
BEGIN
    -- 1. Advisory lock ------------------------------------------------------
    lock_acquired := pg_try_advisory_lock(hashtext('entities_promotion'));
    IF NOT lock_acquired THEN
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'lock_unavailable',
            'diff', '{}'::jsonb
        );
    END IF;

    -- 2. Schema compatibility check ----------------------------------------
    FOR schema_rec IN
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'entities_staging'
           AND column_name NOT IN ('id', 'staging_run_id', 'created_at')
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'entities'
               AND column_name = schema_rec.column_name
        ) THEN
            schema_errors := schema_errors || to_jsonb(schema_rec.column_name);
        END IF;
    END LOOP;

    IF jsonb_array_length(schema_errors) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'schema_mismatch',
            'diff', jsonb_build_object('schema_errors', schema_errors)
        );
    END IF;

    -- 3. Counts -------------------------------------------------------------
    SELECT COUNT(*) INTO staging_total
      FROM entities_staging WHERE staging_run_id = run_id;
    SELECT COUNT(*) INTO alias_staging_tot
      FROM entity_aliases_staging WHERE staging_run_id = run_id;

    WITH run_sources AS (
        SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
    )
    SELECT COUNT(*) INTO prod_entity_total
      FROM entities e
      JOIN run_sources rs ON rs.source = e.source;

    WITH run_sources AS (
        SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
    )
    SELECT COUNT(*) INTO prod_alias_total
      FROM entity_aliases ea
      JOIN entities e ON ea.entity_id = e.id
      JOIN run_sources rs ON rs.source = e.source;

    IF prod_entity_total > 0 THEN
        canonical_shrink := (prod_entity_total - staging_total)::NUMERIC
                            / prod_entity_total;
    END IF;
    IF prod_alias_total > 0 THEN
        alias_shrink := (prod_alias_total - alias_staging_tot)::NUMERIC
                        / prod_alias_total;
    END IF;

    FOR src_rec IN
        SELECT source, COUNT(*) AS n
          FROM entities_staging
         WHERE staging_run_id = run_id
         GROUP BY source
    LOOP
        per_source_json := per_source_json
            || jsonb_build_object(src_rec.source, src_rec.n);

        IF floors ? src_rec.source THEN
            IF src_rec.n < (floors ->> src_rec.source)::BIGINT THEN
                floor_violations := floor_violations
                    || jsonb_build_object(
                        'source', src_rec.source,
                        'observed', src_rec.n,
                        'floor', (floors ->> src_rec.source)::BIGINT
                    );
            END IF;
        END IF;
    END LOOP;

    -- 4. Orphan check -------------------------------------------------------
    FOR orphan_rec IN
        WITH run_sources AS (
            SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
        ),
        heavy AS (
            SELECT e.id, e.canonical_name, e.entity_type, e.source,
                   COUNT(de.*) AS doc_count
              FROM entities e
              JOIN run_sources rs ON rs.source = e.source
              JOIN document_entities de ON de.entity_id = e.id
             GROUP BY e.id, e.canonical_name, e.entity_type, e.source
            HAVING COUNT(de.*) >= orphan_threshold
        )
        SELECT h.*
          FROM heavy h
         WHERE NOT EXISTS (
            SELECT 1 FROM entities_staging s
             WHERE s.staging_run_id = run_id
               AND s.canonical_name = h.canonical_name
               AND s.entity_type    = h.entity_type
               AND s.source         = h.source
         )
    LOOP
        orphan_violations := orphan_violations
            || jsonb_build_object(
                'id', orphan_rec.id,
                'canonical_name', orphan_rec.canonical_name,
                'entity_type', orphan_rec.entity_type,
                'source', orphan_rec.source,
                'doc_count', orphan_rec.doc_count
            );
    END LOOP;

    -- 5. Build the diff object ----------------------------------------------
    result := jsonb_build_object(
        'staging_entity_count', staging_total,
        'staging_alias_count', alias_staging_tot,
        'prod_entity_count_for_sources', prod_entity_total,
        'prod_alias_count_for_sources', prod_alias_total,
        'canonical_shrinkage', canonical_shrink,
        'alias_shrinkage', alias_shrink,
        'per_source_counts', per_source_json,
        'floor_violations', floor_violations,
        'orphan_violations', orphan_violations
    );

    -- 6. Gate decisions -----------------------------------------------------
    IF canonical_shrink > canonical_max THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'canonical_shrinkage',
            'diff', result
        );
    END IF;

    IF alias_shrink > alias_max THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'alias_shrinkage',
            'diff', result
        );
    END IF;

    IF jsonb_array_length(floor_violations) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'floor_violation',
            'diff', result
        );
    END IF;

    IF jsonb_array_length(orphan_violations) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'orphan_violation',
            'diff', result
        );
    END IF;

    -- 7. Atomic upserts ----------------------------------------------------
    WITH ins AS (
        INSERT INTO entities (
            canonical_name, entity_type, discipline, source, source_version,
            ambiguity_class, link_policy, properties, harvest_run_id
        )
        SELECT
            s.canonical_name,
            s.entity_type,
            s.discipline,
            s.source,
            s.source_version,
            s.ambiguity_class::entity_ambiguity_class,
            s.link_policy::entity_link_policy,
            COALESCE(s.properties, '{}'::jsonb),
            run_id
          FROM entities_staging s
         WHERE s.staging_run_id = run_id
        ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
            SET discipline     = COALESCE(EXCLUDED.discipline, entities.discipline),
                source_version = COALESCE(EXCLUDED.source_version, entities.source_version),
                ambiguity_class = COALESCE(EXCLUDED.ambiguity_class, entities.ambiguity_class),
                link_policy    = COALESCE(EXCLUDED.link_policy, entities.link_policy),
                properties     = entities.properties || EXCLUDED.properties,
                harvest_run_id = EXCLUDED.harvest_run_id,
                updated_at     = now()
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ent FROM ins;

    WITH resolved AS (
        SELECT DISTINCT e.id AS entity_id, sa.alias, sa.alias_source
          FROM entity_aliases_staging sa
          JOIN entities e
            ON e.canonical_name = sa.canonical_name
           AND e.entity_type    = sa.entity_type
           AND e.source         = sa.source
         WHERE sa.staging_run_id = run_id
           AND sa.alias IS NOT NULL
    ),
    ins AS (
        INSERT INTO entity_aliases (entity_id, alias, alias_source)
        SELECT entity_id, alias, alias_source FROM resolved
        ON CONFLICT (entity_id, alias) DO UPDATE
            SET alias_source = COALESCE(EXCLUDED.alias_source, entity_aliases.alias_source)
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ali FROM ins;

    WITH resolved AS (
        SELECT DISTINCT e.id AS entity_id, si.id_scheme, si.external_id,
               COALESCE(si.is_primary, false) AS is_primary
          FROM entity_identifiers_staging si
          JOIN entities e
            ON e.canonical_name = si.canonical_name
           AND e.entity_type    = si.entity_type
           AND e.source         = si.source
         WHERE si.staging_run_id = run_id
           AND si.id_scheme IS NOT NULL
           AND si.external_id IS NOT NULL
    ),
    ins AS (
        INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
        SELECT entity_id, id_scheme, external_id, is_primary FROM resolved
        ON CONFLICT (id_scheme, external_id) DO UPDATE
            SET entity_id  = EXCLUDED.entity_id,
                is_primary = EXCLUDED.is_primary
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ids FROM ins;

    result := result || jsonb_build_object(
        'promoted_entities', n_promoted_ent,
        'promoted_aliases', n_promoted_ali,
        'promoted_identifiers', n_promoted_ids
    );

    UPDATE harvest_runs SET status = 'promoted', finished_at = now()
     WHERE id = run_id;

    PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
    RETURN jsonb_build_object(
        'accepted', true,
        'reason', NULL,
        'diff', result
    );
EXCEPTION WHEN OTHERS THEN
    IF lock_acquired THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
    END IF;
    RAISE;
END
$fn$;

COMMENT ON FUNCTION promote_harvest(BIGINT, JSONB, NUMERIC, NUMERIC, INTEGER) IS
    'Atomic promote of *_staging rows into public entity tables with '
    'shadow-diff gating (shrinkage, floor, orphan checks). '
    'Consolidates migration 030 stub + promote_harvest_v2 into single function.';

COMMIT;
