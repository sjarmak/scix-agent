-- Migration 061: section_embeddings — per-section halfvec(1024) embeddings + BM25 tsvector
--
-- Bead: scix_experiments-wqr.9
-- Part of PRD Build (section-embeddings-mcp-consolidation). Establishes the
-- storage layer for section-grain retrieval: one row per entry of
-- papers_fulltext.sections (which is a JSONB array of {heading, level, text,
-- offset} objects parsed from ar5iv/arxiv_local/s2orc/ads_body/docling/abstract
-- in migration 041).
--
-- HALFVEC RATIONALE
-- -----------------
-- We use halfvec(1024) (16-bit floats) instead of vector(1024) for two reasons:
--   1. Storage: halfvec halves the bytes per row vs. full float32 vectors. The
--      paper_embeddings cutover (migrations 053/054, bead scix_experiments-0vy)
--      already validated that halfvec_cosine_ops HNSW gives effectively
--      identical recall to vector_cosine_ops on this corpus while cutting the
--      index size roughly in half.
--   2. Throughput: smaller vectors → more rows fit in shared_buffers and
--      maintenance_work_mem during HNSW build, so build time and query
--      latency both improve.
-- Dimension 1024 matches the upcoming section embedder model (NV-Embed /
-- nomic-embed-text-v2 / SPECTER2-section family). If the embedder dimension
-- is later revised, that is a new migration with a shadow column, not an
-- in-place ALTER (rewriting halfvec(1024) on tens of millions of rows would
-- hold ACCESS EXCLUSIVE for hours; see migration 053 header for the lesson).
--
-- ROW SHAPE
-- ---------
-- One row per section, keyed by (bibcode, section_index). section_index is
-- the 0-based position of the section inside papers_fulltext.sections so the
-- linkage back to the source JSONB array is unambiguous. section_heading is
-- denormalized for cheap filtering / display. section_text_sha256 is the
-- SHA-256 of the exact text that was embedded — the embedder script uses it
-- as a resumability key (skip rows whose hash matches; re-embed on mismatch
-- after a parser bump).
--
-- BM25 INDEX
-- ----------
-- We add a STORED generated tsvector column on papers_fulltext concatenating
-- section headings + section text from the JSONB array, plus a GIN index on
-- it. Trade-off vs. an expression index: the generated column writes ~once
-- per parsed paper and avoids per-query function evaluation, at the cost of
-- some extra disk for the tsvector. Given papers_fulltext is written once
-- per (bibcode, parser_version) and read on every BM25 query, the generated
-- column is the right call.
--
-- SAFETY: section_embeddings MUST be LOGGED. An UNLOGGED table is truncated
-- on crash recovery — see migration 023 and feedback_unlogged_tables for the
-- 32M-embedding loss that prompted this rule. The DO block at the bottom
-- asserts relpersistence='p' and raises if not.
--
-- Idempotent: every DDL uses IF NOT EXISTS so this file is safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. section_embeddings table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_embeddings (
    bibcode             TEXT NOT NULL REFERENCES papers(bibcode),
    section_index       INT  NOT NULL,
    section_heading     TEXT,
    section_text_sha256 TEXT NOT NULL,
    embedding           halfvec(1024) NOT NULL,
    PRIMARY KEY (bibcode, section_index)
);

COMMENT ON TABLE section_embeddings IS
    'Per-section halfvec(1024) embeddings. One row per entry of '
    'papers_fulltext.sections, keyed by (bibcode, section_index). Bead '
    'scix_experiments-wqr.9.';

COMMENT ON COLUMN section_embeddings.section_index IS
    '0-based position of this section inside papers_fulltext.sections '
    '(the source JSONB array). Stable for a given parser_version.';

COMMENT ON COLUMN section_embeddings.section_text_sha256 IS
    'SHA-256 of the exact text that was embedded. Used by the embedder '
    'script as a resumability key: skip rows whose hash matches the current '
    'section text; re-embed on mismatch after a parser bump.';

COMMENT ON COLUMN section_embeddings.embedding IS
    'halfvec(1024). See header comment for the halfvec rationale and the '
    'paper_embeddings precedent (migrations 053/054, bead 0vy).';

-- ---------------------------------------------------------------------------
-- 2. HNSW index on the embedding column
-- ---------------------------------------------------------------------------
-- m=16, ef_construction=64 matches migration 054 (idx_embed_hnsw_indus_hv) so
-- recall/latency tuning carries over directly. Parallel maintenance workers
-- raised to 7 to use the available cores during the build.
SET max_parallel_maintenance_workers = 7;

CREATE INDEX IF NOT EXISTS idx_section_embeddings_hnsw
    ON section_embeddings
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- 3. BM25 tsvector on papers_fulltext.sections
-- ---------------------------------------------------------------------------
-- Generated STORED column concatenating heading + text across all sections in
-- the JSONB array. jsonb_path_query_array(...) returns a JSONB array whose
-- ::text rendering is a JSON-encoded list of strings — that is fine for
-- to_tsvector('english', ...): it tokenizes the text content and the JSON
-- punctuation/quoting is filtered as non-word noise. Concatenating headings
-- and body text into a single tsvector is acceptable for a first cut; if we
-- later need to weight headings (setweight 'A' vs 'B'), that is a follow-up.
ALTER TABLE papers_fulltext
    ADD COLUMN IF NOT EXISTS sections_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            coalesce(jsonb_path_query_array(sections, '$[*].text')::text, '')
            || ' ' ||
            coalesce(jsonb_path_query_array(sections, '$[*].heading')::text, '')
        )
    ) STORED;

COMMENT ON COLUMN papers_fulltext.sections_tsv IS
    'GENERATED tsvector over heading + text from sections JSONB. Backs the '
    'BM25 leg of section-grain retrieval. Bead scix_experiments-wqr.9.';

CREATE INDEX IF NOT EXISTS idx_papers_fulltext_sections_tsv
    ON papers_fulltext USING gin (sections_tsv);

-- ---------------------------------------------------------------------------
-- 4. Safety assertion: section_embeddings must be LOGGED
-- ---------------------------------------------------------------------------
-- An UNLOGGED table is truncated on crash recovery. We previously lost 32M
-- SPECTER2 embeddings to exactly this failure mode — see migration 023 and
-- feedback_unlogged_tables. Refuse to leave the migration with an UNLOGGED
-- section_embeddings.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'section_embeddings'
      AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'section_embeddings did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'section_embeddings must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
