-- Migration 048: publisher suppress + arxiv/source version columns on papers_fulltext.
--
-- Adds three idempotent columns to papers_fulltext:
--   (a) suppressed_by_publisher BOOLEAN NOT NULL DEFAULT false — true when the
--       publisher has asked us to suppress serving full-text for this paper.
--       Partial index so the (expected-small) suppressed subset is cheap to scan.
--   (b) source_version TEXT NULL — upstream source version identifier
--       (e.g. ar5iv build tag, s2orc snapshot id, docling version). Free-form.
--   (c) arxiv_version SMALLINT NULL — arXiv version number (v1, v2, ...) when
--       the row was parsed from an arXiv LaTeX source. NULL for non-arXiv sources.
--
-- All column adds use ADD COLUMN IF NOT EXISTS and the partial index uses
-- CREATE INDEX IF NOT EXISTS so the migration is safe to re-run.
--
-- SAFETY: papers_fulltext MUST remain LOGGED (relpersistence='p'). The DO
-- block at the end re-asserts this to protect against accidental regressions
-- — see migration 023 and the feedback_unlogged_tables memory for the 32M
-- embedding loss that prompted this rule.

BEGIN;

ALTER TABLE papers_fulltext
    ADD COLUMN IF NOT EXISTS suppressed_by_publisher BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE papers_fulltext
    ADD COLUMN IF NOT EXISTS source_version TEXT;

ALTER TABLE papers_fulltext
    ADD COLUMN IF NOT EXISTS arxiv_version SMALLINT;

COMMENT ON COLUMN papers_fulltext.suppressed_by_publisher IS
    'True when the publisher has requested suppression of full-text serving '
    'for this paper. Default false. Partial index on the true subset.';

COMMENT ON COLUMN papers_fulltext.source_version IS
    'Upstream source version identifier (e.g. ar5iv build tag, s2orc snapshot '
    'id, docling version). NULL when not applicable.';

COMMENT ON COLUMN papers_fulltext.arxiv_version IS
    'arXiv version number (v1, v2, ...) when parsed from an arXiv LaTeX source. '
    'NULL for non-arXiv sources.';

CREATE INDEX IF NOT EXISTS idx_papers_fulltext_suppressed_by_publisher
    ON papers_fulltext (suppressed_by_publisher)
    WHERE suppressed_by_publisher = true;

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_fulltext' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_fulltext does not exist';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_fulltext must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
