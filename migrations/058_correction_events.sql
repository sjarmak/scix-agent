-- Migration 058: papers.correction_events JSONB + denormalized retracted_at.
--
-- Implements the broadened MH-3 from PRD docs/prd/scix_deep_search_v1.md
-- amendment A3. Replaces a single-purpose `retracted_at` flag with a richer
-- JSONB array of correction events sourced from four independent feeds:
--   - Retraction Watch CC0 CSV
--   - OpenAlex `is_retracted` filter
--   - Crossref `update-to` relations
--   - Top-15 astronomy journal Errata RSS feeds
--
-- Each event is a JSON object of the shape:
--   {"type":     "retraction"|"erratum"|"correction"
--               |"expression_of_concern"|"recalibration_supersession",
--    "source":   "retraction_watch"|"openalex"|"crossref"|"journal_rss",
--    "doi":      "<doi>",
--    "date":     "YYYY-MM-DD"}
--
-- The denormalized `papers.retracted_at` column is a convenience derivation:
-- the earliest event date among events of type='retraction' on this paper.
-- The orchestrator (scripts/ingest_corrections.py) keeps it in sync.
--
-- All column adds use ADD COLUMN IF NOT EXISTS and CREATE INDEX IF NOT EXISTS,
-- so the migration is safe to re-run.
--
-- SAFETY: papers MUST remain LOGGED (relpersistence='p'). The DO block at the
-- end re-asserts this — see migration 023 / feedback_unlogged_tables for the
-- 32M-embedding loss that prompted this rule.

BEGIN;

ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS correction_events JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS retracted_at TIMESTAMPTZ;

COMMENT ON COLUMN papers.correction_events IS
    'JSONB array of correction events for this paper. Each element: '
    '{type, source, doi, date}. type in (retraction, erratum, correction, '
    'expression_of_concern, recalibration_supersession). source in '
    '(retraction_watch, openalex, crossref, journal_rss). Populated by '
    'scripts/ingest_corrections.py (PRD A3 / MH-3 broadened).';

COMMENT ON COLUMN papers.retracted_at IS
    'Denormalized convenience: earliest event date among correction_events '
    'where type=retraction. NULL if no retraction event present. Kept in '
    'sync by scripts/ingest_corrections.py.';

CREATE INDEX IF NOT EXISTS idx_papers_correction_events
    ON papers USING GIN (correction_events);

CREATE INDEX IF NOT EXISTS idx_papers_retracted_at
    ON papers (retracted_at)
    WHERE retracted_at IS NOT NULL;

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers does not exist';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
