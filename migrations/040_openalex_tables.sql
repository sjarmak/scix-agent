-- Migration 040: OpenAlex S3 snapshot tables — papers_openalex + works_references.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W2 (wqr.3). OpenAlex is the
-- graph backbone: ~260M works, CC0 licensed, no auth. These two tables hold
-- the pruned Work records and their citation edges independently of the ADS
-- `papers` table, enabling graph analytics across the full OpenAlex corpus
-- without requiring a bibcode match.
--
-- The loader (src/scix/sources/openalex.py) populates these via DuckDB-over-
-- Parquet staging and binary COPY, then joins on DOI/arXiv ID to update
-- papers_external_ids.openalex_id for matched papers.
--
-- SAFETY: both tables MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' for both tables.

BEGIN;

-- ---------------------------------------------------------------------------
-- papers_openalex — pruned OpenAlex Work records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS papers_openalex (
    openalex_id          TEXT PRIMARY KEY,
    doi                  TEXT,
    title                TEXT,
    publication_year     SMALLINT,
    abstract             TEXT,
    topics               JSONB,
    open_access          JSONB,
    best_oa_location     JSONB,
    cited_by_count       INT,
    referenced_works_count INT,
    type                 TEXT,
    updated_date         DATE,
    created_date         DATE
);

COMMENT ON TABLE papers_openalex IS
    'Pruned OpenAlex Work records from the S3 snapshot. Populated by '
    'src/scix/sources/openalex.py. CC0 licensed, ~260M works. The '
    'abstract field is reconstructed from abstract_inverted_index.';

-- Lookup indexes for join operations
CREATE INDEX IF NOT EXISTS idx_papers_openalex_doi
    ON papers_openalex (doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_openalex_year
    ON papers_openalex (publication_year);
CREATE INDEX IF NOT EXISTS idx_papers_openalex_updated
    ON papers_openalex (updated_date);

-- ---------------------------------------------------------------------------
-- works_references — citation edges between OpenAlex works
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS works_references (
    source_openalex_id      TEXT NOT NULL,
    referenced_openalex_id  TEXT NOT NULL,
    PRIMARY KEY (source_openalex_id, referenced_openalex_id)
);

COMMENT ON TABLE works_references IS
    'Citation edges between OpenAlex works. source_openalex_id cites '
    'referenced_openalex_id. No FK to papers_openalex because referenced '
    'works may be outside the ingested corpus (xpac expansion).';

-- Reverse-lookup index for cited-by queries
CREATE INDEX IF NOT EXISTS idx_works_references_target
    ON works_references (referenced_openalex_id);

-- ---------------------------------------------------------------------------
-- Safety assertions: both tables MUST be LOGGED (relpersistence='p')
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    rp CHAR(1);
BEGIN
    -- Check papers_openalex
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_openalex' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_openalex did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_openalex must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;

    -- Check works_references
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'works_references' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'works_references did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'works_references must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
