-- Migration 045: Citation diff table + aggregate materialized views.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W3 (wqr.4). Cross-validates
-- the ADS citation graph (299M edges in citation_edges) against the OpenAlex
-- citation graph (works_references) to produce raw material for paper Section
-- 3.3: "Graph analytics on partial corpora are fundamentally misleading."
--
-- The citation_diff table stores the FULL OUTER JOIN of both edge sets,
-- bucketed by provenance (in_ads, in_openalex). Two materialized views
-- aggregate per-year and per-journal for quick reporting.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p'.

BEGIN;

-- ---------------------------------------------------------------------------
-- citation_diff — edge-level provenance
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS citation_diff (
    source_bibcode    TEXT NOT NULL,
    target_bibcode    TEXT NOT NULL,
    in_ads            BOOLEAN NOT NULL DEFAULT false,
    in_openalex       BOOLEAN NOT NULL DEFAULT false,
    source_attrs      JSONB,
    PRIMARY KEY (source_bibcode, target_bibcode)
);

COMMENT ON TABLE citation_diff IS
    'Full outer join of ADS citation_edges and OpenAlex works_references, '
    'joined via papers_external_ids crosswalk. Each row records whether a '
    'directed citation edge exists in ADS, OpenAlex, or both. Populated by '
    'scripts/analyze_citation_diff.py --populate. See paper Section 3.3.';

-- Indexes for aggregate queries
CREATE INDEX IF NOT EXISTS idx_citation_diff_source
    ON citation_diff (source_bibcode);
CREATE INDEX IF NOT EXISTS idx_citation_diff_target
    ON citation_diff (target_bibcode);
CREATE INDEX IF NOT EXISTS idx_citation_diff_provenance
    ON citation_diff (in_ads, in_openalex);

-- ---------------------------------------------------------------------------
-- Materialized view: per-year edge coverage
-- ---------------------------------------------------------------------------
-- Joins source_bibcode to papers.year for the publication year of the citing
-- paper. Buckets: ads_only, openalex_only, both, total.

CREATE MATERIALIZED VIEW IF NOT EXISTS citation_diff_by_year AS
SELECT
    COALESCE(p.year, 0)                                   AS pub_year,
    COUNT(*)                                              AS total_edges,
    COUNT(*) FILTER (WHERE cd.in_ads AND cd.in_openalex)  AS both_count,
    COUNT(*) FILTER (WHERE cd.in_ads AND NOT cd.in_openalex) AS ads_only_count,
    COUNT(*) FILTER (WHERE NOT cd.in_ads AND cd.in_openalex) AS openalex_only_count,
    ROUND(
        COUNT(*) FILTER (WHERE cd.in_ads AND cd.in_openalex)::numeric
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                                     AS overlap_pct
FROM citation_diff cd
JOIN papers p ON p.bibcode = cd.source_bibcode
GROUP BY p.year
ORDER BY p.year;

-- ---------------------------------------------------------------------------
-- Materialized view: per-journal edge coverage
-- ---------------------------------------------------------------------------
-- Joins source_bibcode to papers.pub (journal/publication name).

CREATE MATERIALIZED VIEW IF NOT EXISTS citation_diff_by_journal AS
SELECT
    COALESCE(p.pub, '(unknown)')                          AS journal,
    COUNT(*)                                              AS total_edges,
    COUNT(*) FILTER (WHERE cd.in_ads AND cd.in_openalex)  AS both_count,
    COUNT(*) FILTER (WHERE cd.in_ads AND NOT cd.in_openalex) AS ads_only_count,
    COUNT(*) FILTER (WHERE NOT cd.in_ads AND cd.in_openalex) AS openalex_only_count,
    ROUND(
        COUNT(*) FILTER (WHERE cd.in_ads AND cd.in_openalex)::numeric
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                                     AS overlap_pct
FROM citation_diff cd
JOIN papers p ON p.bibcode = cd.source_bibcode
GROUP BY p.pub
ORDER BY total_edges DESC;

-- Unique index on the materialized views for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_citation_diff_by_year_pk
    ON citation_diff_by_year (pub_year);
CREATE UNIQUE INDEX IF NOT EXISTS idx_citation_diff_by_journal_pk
    ON citation_diff_by_journal (journal);

-- ---------------------------------------------------------------------------
-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'citation_diff' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'citation_diff did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'citation_diff must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
