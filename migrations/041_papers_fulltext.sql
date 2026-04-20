-- Migration 041: papers_fulltext — structured full-text from multiple sources.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W4 (wqr.5). Single destination
-- table for structured full-text parsed from ar5iv HTML, arXiv LaTeX, S2ORC,
-- ADS body, Docling, or abstract-only fallback. Each source parser writes
-- sections (with heading/level/text/offset), inline citations, figures, tables,
-- and equations as JSONB arrays.
--
-- The `source` column indicates provenance and determines licensing treatment:
-- ar5iv/arxiv_local are LaTeX-derived (ADR-006 internal-use-only), s2orc has
-- its own license, ads_body is ADS-licensed, abstract is always open.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' and raises if not.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_fulltext (
    bibcode        TEXT PRIMARY KEY REFERENCES papers(bibcode),
    source         TEXT NOT NULL,
    sections       JSONB NOT NULL,
    inline_cites   JSONB NOT NULL,
    figures        JSONB NOT NULL DEFAULT '[]',
    tables         JSONB NOT NULL DEFAULT '[]',
    equations      JSONB NOT NULL DEFAULT '[]',
    parser_version TEXT NOT NULL,
    parsed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_fulltext IS
    'Structured full-text from multiple sources (ar5iv, arxiv_local, s2orc, '
    'ads_body, docling, abstract). Each row holds parsed sections, inline '
    'citations, figures, tables, and equations as JSONB arrays. Source column '
    'determines licensing treatment per ADR-006.';

COMMENT ON COLUMN papers_fulltext.source IS
    'Provenance tag: ar5iv | arxiv_local | s2orc | ads_body | docling | abstract. '
    'ar5iv/arxiv_local are LaTeX-derived (ADR-006 internal-use-only).';

COMMENT ON COLUMN papers_fulltext.sections IS
    'Array of {heading, level, text, offset} objects representing the document '
    'structure. Level 1 = top-level section, 2 = subsection, etc.';

COMMENT ON COLUMN papers_fulltext.inline_cites IS
    'Array of {offset, bib_ref, target_bibcode_or_null} objects representing '
    'inline citation references found during parsing.';

CREATE INDEX IF NOT EXISTS idx_papers_fulltext_source
    ON papers_fulltext (source);

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_fulltext' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_fulltext did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_fulltext must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
