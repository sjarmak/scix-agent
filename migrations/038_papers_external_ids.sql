-- Migration 038: papers_external_ids crosswalk table.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W1 (wqr.2). First of the
-- external-source ingestion pipeline: a single crosswalk table that joins
-- ADS bibcodes to identifiers from OpenAlex / arXiv / Semantic Scholar /
-- PubMed / PMC, plus has_* boolean flags tracking which structured full-text
-- sources have been harvested.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' and raises if not.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_external_ids (
    bibcode              TEXT PRIMARY KEY REFERENCES papers(bibcode),
    doi                  TEXT,
    arxiv_id             TEXT,
    openalex_id          TEXT,
    s2_corpus_id         BIGINT,
    s2_paper_id          TEXT,
    pmcid                TEXT,
    pmid                 BIGINT,
    has_ads_body         BOOLEAN NOT NULL DEFAULT false,
    has_arxiv_source     BOOLEAN NOT NULL DEFAULT false,
    has_ar5iv_html       BOOLEAN NOT NULL DEFAULT false,
    has_s2orc_body       BOOLEAN NOT NULL DEFAULT false,
    openalex_has_pdf_url BOOLEAN NOT NULL DEFAULT false,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_external_ids IS
    'Crosswalk from ADS bibcode to external identifiers (OpenAlex / arXiv / '
    'Semantic Scholar / PubMed). Populated incrementally by PRD Build 5 work '
    'units W1-W6. has_* flags track which structured full-text sources have '
    'been ingested for each paper.';

-- B-tree lookup indexes on the join keys. Not unique — a single external id
-- can in rare cases point to multiple bibcodes (arxiv preprints with multiple
-- ADS records, DOI collisions between reprints, etc.).
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_doi
    ON papers_external_ids (doi);
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_arxiv_id
    ON papers_external_ids (arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_openalex_id
    ON papers_external_ids (openalex_id);
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_s2_corpus_id
    ON papers_external_ids (s2_corpus_id);

-- updated_at trigger: the DEFAULT now() only fires on INSERT. The column is
-- meaningful only if it also advances on UPDATE.
CREATE OR REPLACE FUNCTION papers_external_ids_touch() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_papers_external_ids_updated_at ON papers_external_ids;
CREATE TRIGGER trig_papers_external_ids_updated_at
    BEFORE UPDATE ON papers_external_ids
    FOR EACH ROW EXECUTE FUNCTION papers_external_ids_touch();

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_external_ids' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_external_ids did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_external_ids must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
