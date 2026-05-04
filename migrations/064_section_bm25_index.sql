-- Migration 064: GIN index on papers_fulltext.sections_tsv
--
-- Bead: scix_experiments-zpm4 (parent: scix_experiments-wqr.9.7)
--
-- Run AFTER scripts/backfill_sections_tsv.py reports 100% completion.
-- Building the GIN index against an already-populated column is one bulk
-- operation; building it against a nullable empty column and letting the
-- backfill UPDATE billions of GIN tuples in-place is much slower and
-- bloats the index.
--
-- Idempotent: CREATE INDEX IF NOT EXISTS.
--
-- COST ESTIMATE (informational; verify before running)
-- ----------------------------------------------------
-- - Build time: ~3-6 hours wall-clock at 14.4 M rows with maintenance_work_mem
--   raised to 4 GB. Comparable to ix_papers_body_tsv on papers.body which
--   covers the same corpus.
-- - Disk: ~10-25 GB final index size. Build phase needs roughly 1.5x final
--   size as scratch.
-- - Lock: CREATE INDEX (without CONCURRENTLY) takes a SHARE lock on
--   papers_fulltext, blocking writes but allowing reads. Acceptable because
--   papers_fulltext is fed by the parser pipelines which can pause briefly.
--   Switch to CONCURRENTLY if write availability becomes a constraint —
--   trade-off is roughly 2x build time.

SET maintenance_work_mem = '4GB';

CREATE INDEX IF NOT EXISTS idx_papers_fulltext_sections_tsv
    ON papers_fulltext
    USING gin (sections_tsv);

ANALYZE papers_fulltext;

-- ---------------------------------------------------------------------------
-- Helper SQL function — search_sections_bm25
-- ---------------------------------------------------------------------------
-- Bead zpm4 D3. Two-stage retrieval: (1) GIN-backed paper-level tsv match to
-- bound candidate set, (2) per-section re-tokenization on the bounded set to
-- pick the best-scoring section per paper. Returns section-tagged (bibcode,
-- section_heading, snippet) tuples plus the paper-level rank for ordering.
--
-- Acceptance: must complete in <500 ms p95 on prod for k=10. The over-fetch
-- factor (k * 5) traded against per-section re-tokenization cost is the key
-- knob if latency drifts; 5x is a starting point, raise to 10x if recall
-- holds while p95 stays under budget.
CREATE OR REPLACE FUNCTION search_sections_bm25(
    q text,
    topk int DEFAULT 20
) RETURNS TABLE (
    bibcode text,
    rank real,
    section_index int,
    section_heading text,
    snippet text
) AS $$
    WITH paper_hits AS (
        SELECT pf.bibcode,
               ts_rank_cd(pf.sections_tsv, plainto_tsquery('english', q)) AS rank,
               pf.sections
        FROM papers_fulltext pf
        WHERE pf.sections_tsv @@ plainto_tsquery('english', q)
        ORDER BY rank DESC
        LIMIT topk * 5
    ),
    sectioned AS (
        SELECT ph.bibcode,
               ph.rank,
               (ord - 1)::int AS section_index,
               s->>'heading'  AS section_heading,
               ts_headline(
                   'english',
                   coalesce(s->>'text', ''),
                   plainto_tsquery('english', q),
                   'MaxWords=40, MinWords=20'
               ) AS snippet,
               ts_rank_cd(
                   to_tsvector('english', coalesce(s->>'text', '')),
                   plainto_tsquery('english', q)
               ) AS section_rank
        FROM paper_hits ph,
             LATERAL jsonb_array_elements(ph.sections) WITH ORDINALITY AS arr(s, ord)
        WHERE to_tsvector('english', coalesce(s->>'text', ''))
              @@ plainto_tsquery('english', q)
    ),
    best_section AS (
        SELECT DISTINCT ON (bibcode) *
        FROM sectioned
        ORDER BY bibcode, section_rank DESC
    )
    SELECT bibcode, rank, section_index, section_heading, snippet
    FROM best_section
    ORDER BY rank DESC
    LIMIT topk;
$$ LANGUAGE sql STABLE;

COMMENT ON FUNCTION search_sections_bm25(text, int) IS
    'Two-stage BM25 over papers_fulltext.sections_tsv: GIN paper-level filter '
    'then per-section re-tokenization to pick the best section per paper. '
    'Returns (bibcode, rank, section_index, section_heading, snippet). Bead '
    'scix_experiments-zpm4 D3 / wqr.9.7 — used by the section_retrieval MCP '
    'tool and the section_bm25 eval mode.';

INSERT INTO schema_migrations (version, filename)
    VALUES (64, '064_section_bm25_index.sql')
    ON CONFLICT (version) DO NOTHING;
