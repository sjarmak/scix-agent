-- Migration 039: papers_ads_body — full text from the ADS `body` field.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W1 (wqr.2). Dedicated table for
-- the ADS-harvested full text of each paper. Although papers.body already
-- exists (migration 010), this dedicated table:
--   - Gives the body its own primary key so the 17.8M-row bulk load can drop
--     indexes cleanly without touching the hot `papers` table.
--   - Stores a body-only GENERATED STORED tsvector, enabling BM25-style
--     queries on full text independent of the title/abstract tsv on papers
--     (migration 003), which is trigger-maintained and doesn't cover body.
--   - Isolates the 17.8M-row write volume from papers, so reindexing the body
--     tsv does not block queries against papers.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory. The
-- block at the bottom of this file asserts relpersistence='p'.
--
-- Note on text search config: the tsv column uses the built-in 'english'
-- config, not scix_english (migration 003), because GENERATED STORED columns
-- require an immutable configuration reference that round-trips cleanly
-- through pg_dump. scix_english is reachable by name but upgrading this
-- column to use it is a follow-up (separate bead, not this one). The
-- tradeoff is that hyphenated tokens like "X-ray" get split on hyphen; for
-- full-text recall on body text this is acceptable.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_ads_body (
    bibcode      TEXT PRIMARY KEY REFERENCES papers(bibcode),
    body_text    TEXT NOT NULL,
    body_length  INT  NOT NULL,
    harvested_at TIMESTAMPTZ NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', body_text)) STORED
);

COMMENT ON TABLE papers_ads_body IS
    'Full text from the ADS `body` field, harvested ~55% of ADS papers. '
    'Populated by scripts/ingest_ads_body.py. body-only tsvector enables '
    'full-text search independent of the title/abstract tsv on papers.';

COMMENT ON COLUMN papers_ads_body.tsv IS
    'Generated stored tsvector using the built-in `english` config. See '
    'migration 039 preamble for the tradeoff vs scix_english.';

CREATE INDEX IF NOT EXISTS idx_papers_ads_body_tsv
    ON papers_ads_body USING GIN (tsv);

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_ads_body' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_ads_body did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_ads_body must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
