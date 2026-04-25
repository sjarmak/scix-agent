-- migration: 060_staging_extractions_section_columns — body NER provenance
--
-- Adds section-aware provenance columns to ``staging.extractions`` so the
-- M2 body NER pilot (``scripts/run_ner_bodies.py``) can record which
-- section of a paper a mention came from. Both columns are nullable —
-- abstract-level extractions (``source='ner_v1'``, ``quant_claim``) leave
-- them NULL.
--
-- Columns added:
--   * section_name TEXT     — canonical section name from
--     ``scix.section_parser`` (e.g. ``methods``, ``results``,
--     ``observations``). For per-paper rollup rows the value is a
--     comma-separated list of distinct kept sections; the per-section
--     breakdown lives in ``payload->'sections'``.
--   * char_offset INTEGER   — character offset within the paper body of
--     the first kept section. Lets downstream tools jump straight to a
--     section without re-parsing.
--
-- Supporting index on ``source`` already exists from migration 049; we
-- add a partial index on ``section_name`` that only materialises rows
-- where it is non-NULL (i.e. body NER), keeping the index tiny.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS + CREATE INDEX IF NOT EXISTS.
-- The LOGGED invariant is re-asserted at the end so a future ``UNLOGGED``
-- redefinition trips the migration immediately.

BEGIN;

ALTER TABLE staging.extractions
    ADD COLUMN IF NOT EXISTS section_name TEXT;

ALTER TABLE staging.extractions
    ADD COLUMN IF NOT EXISTS char_offset INTEGER;

-- Partial index — most rows in staging.extractions are abstract-level and
-- leave section_name NULL. The partial predicate keeps the index small
-- (body NER rows only) while still supporting the MCP-entity-tool query
-- pattern: ``WHERE source = 'ner_body' AND section_name = ANY(...)``.
CREATE INDEX IF NOT EXISTS idx_staging_extractions_section_name
    ON staging.extractions (section_name)
    WHERE section_name IS NOT NULL;

-- Re-assert LOGGED invariant for staging.extractions. A silent UNLOGGED
-- redefinition would lose body NER rows on a postgres restart — see
-- migration 023 + the feedback_unlogged_tables memory.
DO $$
DECLARE
    is_logged "char";
BEGIN
    SELECT cl.relpersistence
      INTO is_logged
      FROM pg_class cl
      JOIN pg_namespace n ON n.oid = cl.relnamespace
     WHERE n.nspname = 'staging'
       AND cl.relname = 'extractions';

    IF is_logged IS NULL THEN
        RAISE EXCEPTION
            'staging.extractions does not exist — run migration 015 / 049 first';
    END IF;

    IF is_logged <> 'p' THEN
        RAISE EXCEPTION
            'staging.extractions must be LOGGED (relpersistence=''p'') to '
            'protect body NER rows from crash loss — see migration 023';
    END IF;
END
$$;

COMMIT;
