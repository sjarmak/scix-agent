-- Migration 046: canonical_bibcode column + alternate_bibcode GIN index.
--
-- Part of the cross-bibcode LaTeX propagation work (ADR-006 addendum). Full-text
-- parsed from ar5iv/arxiv_local is keyed by the arXiv-derived bibcode, but the
-- same paper may be represented in papers under a journal bibcode with the
-- arXiv identifier stored in papers.alternate_bibcode. To propagate the parsed
-- full-text to the canonical record, we need:
--
--   1. papers_fulltext.canonical_bibcode — the canonical papers.bibcode that
--      this full-text row logically belongs to (nullable while backfill runs;
--      resolver populates it by matching papers_fulltext.bibcode against
--      papers.alternate_bibcode).
--
--   2. A GIN index on papers.alternate_bibcode so the resolver can look up
--      "which papers row lists this arXiv bibcode as an alternate?" in O(log n)
--      instead of sequential scanning the 32M-row papers table.
--
-- SAFETY: papers_fulltext MUST remain LOGGED. An UNLOGGED table is truncated on
-- crash recovery — see migration 023 and the feedback_unlogged_tables memory
-- for the 32M-embedding loss that prompted this rule. The DO block at the
-- bottom of this file re-asserts relpersistence='p' and raises if not.
--
-- Idempotent: every DDL statement uses IF NOT EXISTS so this migration can be
-- re-run safely.

BEGIN;

-- 1. Add the canonical_bibcode column on papers_fulltext.
ALTER TABLE papers_fulltext
    ADD COLUMN IF NOT EXISTS canonical_bibcode TEXT;

COMMENT ON COLUMN papers_fulltext.canonical_bibcode IS
    'The canonical papers.bibcode this full-text row logically belongs to. '
    'Nullable while the cross-bibcode resolver backfill is in progress. '
    'Populated by matching papers_fulltext.bibcode against papers.alternate_bibcode.';

-- 2. B-tree index on canonical_bibcode for equality/join lookups.
CREATE INDEX IF NOT EXISTS idx_papers_fulltext_canonical_bibcode
    ON papers_fulltext (canonical_bibcode);

-- 3. GIN index on papers.alternate_bibcode (TEXT[]) so the resolver can
--    efficiently find the papers row whose alternate_bibcode array contains
--    a given arXiv bibcode.
CREATE INDEX IF NOT EXISTS ix_papers_alternate_bibcode_gin
    ON papers USING GIN (alternate_bibcode);

-- Safety assertion: refuse to leave the migration with an UNLOGGED
-- papers_fulltext table. See migration 041 for the original assertion.
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
