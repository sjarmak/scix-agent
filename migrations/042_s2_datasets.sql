-- Migration 042: Semantic Scholar Datasets tables.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W5 (wqr.6). Creates tables for
-- ingesting data from the Semantic Scholar Open Data Platform:
--   - papers_s2orc_raw: raw S2ORC records (parsed papers with body_text)
--   - papers_s2ag: pruned S2AG metadata (225M papers)
--   - s2_citations: S2AG citation edges with intent + influence flags
--   - ALTER citation_edges to add edge_attrs JSONB column
--
-- SAFETY: all tables MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' for each new table.

BEGIN;

-- ---------------------------------------------------------------------------
-- papers_s2orc_raw — raw S2ORC parsed papers
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS papers_s2orc_raw (
    s2_corpus_id     BIGINT PRIMARY KEY,
    external_ids     JSONB,
    content          JSONB NOT NULL,
    source_release   TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_s2orc_raw IS
    'Raw S2ORC records from the Semantic Scholar Datasets API. Keyed by '
    's2_corpus_id. content column holds body_text[], cite_spans[], bib_entries[]. '
    'Populated by src/scix/sources/s2_datasets.py.';

CREATE INDEX IF NOT EXISTS idx_s2orc_raw_release
    ON papers_s2orc_raw (source_release);

-- ---------------------------------------------------------------------------
-- papers_s2ag — pruned Semantic Scholar Academic Graph metadata
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS papers_s2ag (
    s2_corpus_id     BIGINT PRIMARY KEY,
    external_ids     JSONB,
    title            TEXT,
    authors          JSONB,
    year             SMALLINT,
    venue            TEXT,
    citation_count   INTEGER,
    reference_count  INTEGER,
    influential_citation_count INTEGER,
    is_open_access   BOOLEAN,
    fields_of_study  JSONB,
    publication_types TEXT[],
    publication_date TEXT,
    journal          JSONB,
    source_release   TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_s2ag IS
    'Pruned S2AG metadata from the Semantic Scholar Datasets API. 225M papers. '
    'Pruned to essential fields; raw data can be re-fetched from S2 snapshots. '
    'Populated by src/scix/sources/s2_datasets.py.';

CREATE INDEX IF NOT EXISTS idx_s2ag_year
    ON papers_s2ag (year);
CREATE INDEX IF NOT EXISTS idx_s2ag_release
    ON papers_s2ag (source_release);

-- ---------------------------------------------------------------------------
-- s2_citations — S2AG citation edges with intent and influence
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS s2_citations (
    citing_corpus_id BIGINT NOT NULL,
    cited_corpus_id  BIGINT NOT NULL,
    intents          TEXT[],
    is_influential   BOOLEAN NOT NULL DEFAULT false,
    source_release   TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (citing_corpus_id, cited_corpus_id)
);

COMMENT ON TABLE s2_citations IS
    'S2AG citation edges with intent labels (methodology, background, '
    'result_comparison) and influence flags. 2.8B edges in full dataset. '
    'Populated by src/scix/sources/s2_datasets.py.';

CREATE INDEX IF NOT EXISTS idx_s2_citations_cited
    ON s2_citations (cited_corpus_id);
CREATE INDEX IF NOT EXISTS idx_s2_citations_influential
    ON s2_citations (is_influential) WHERE is_influential = true;

-- ---------------------------------------------------------------------------
-- ALTER citation_edges: add edge_attrs JSONB for S2AG intent/influence merge
-- ---------------------------------------------------------------------------

ALTER TABLE citation_edges
    ADD COLUMN IF NOT EXISTS edge_attrs JSONB;

COMMENT ON COLUMN citation_edges.edge_attrs IS
    'JSONB attributes merged from external sources. Currently holds '
    's2_intents (TEXT[]) and s2_is_influential (BOOLEAN) from S2AG citation data. '
    'Populated by the citation intent merger in src/scix/sources/s2_datasets.py.';

-- ---------------------------------------------------------------------------
-- Safety assertions: all new tables must be LOGGED (relpersistence='p')
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    tbl TEXT;
    rp  CHAR(1);
BEGIN
    FOREACH tbl IN ARRAY ARRAY['papers_s2orc_raw', 'papers_s2ag', 's2_citations']
    LOOP
        SELECT relpersistence INTO rp
        FROM pg_class
        WHERE relname = tbl AND relnamespace = 'public'::regnamespace;
        IF rp IS NULL THEN
            RAISE EXCEPTION '% did not get created', tbl;
        END IF;
        IF rp <> 'p' THEN
            RAISE EXCEPTION '% must be LOGGED (relpersistence=''p''), got %', tbl, rp;
        END IF;
    END LOOP;
END
$$;

COMMIT;
