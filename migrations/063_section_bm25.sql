-- Migration 063: section-grain BM25 column + tsvector helper (nullable, backfilled)
--
-- Bead: scix_experiments-zpm4 (parent: scix_experiments-wqr.9.7)
--
-- HISTORY
-- -------
-- v1 of this migration used `ADD COLUMN sections_tsv tsvector GENERATED ALWAYS
-- AS (...) STORED`. That approach was abandoned for two reasons discovered
-- during pre-migration validation:
--
--   1. Disk pressure. ALTER TABLE ADD COLUMN with a non-trivial expression
--      rewrites the entire heap. papers_fulltext is 242 GB total (heap 4 GB,
--      TOAST 236 GB, idx 1.5 GB) so peak demand was 534-584 GB during the
--      rewrite + new column data. Available DS headroom was 269 GB.
--      See blocker mail gc-65820 (2026-04-29 00:30) for the headroom ledger.
--
--   2. tsvector size limit. The serialized JSONB-array-as-text expression
--      hit `string is too long for tsvector (1525914 bytes, max 1048575)`
--      on a 0.0001%-tail outlier. The GENERATED-ALWAYS expression has no
--      catch path; one bad row would have aborted the entire rewrite at
--      hour ~5 of the ALTER TABLE, costing a full retry.
--
-- v2 (this file) splits the work into two phases that fit available headroom
-- and degrade gracefully on outliers:
--
--   Phase A (this migration): add a regular nullable tsvector column +
--   create the per-row helper function `safe_sections_tsv(jsonb)` that caps
--   raw-text input at 900 KB before tokenization (well under the 1 MB
--   tsvector ceiling) and returns the empty tsvector on any unexpected
--   per-row failure. ADD COLUMN with no default does not rewrite the heap
--   in PG12+ — it sets a system attribute and adds a NULL placeholder
--   per row, so this migration completes in seconds.
--
--   Phase B (migration 064): the GIN index. Created AFTER backfill so the
--   index build is a single bulk operation rather than 14 M individual
--   GIN-tuple inserts during the UPDATE backfill.
--
--   Backfill: scripts/backfill_sections_tsv.py iterates the table in
--   ctid-paged batches and runs `UPDATE ... SET sections_tsv =
--   safe_sections_tsv(sections) WHERE sections_tsv IS NULL AND ctid = ANY(...)`.
--   Resumable; idempotent under restart.
--
-- TSVECTOR COMPOSITION
-- --------------------
-- safe_sections_tsv(jsonb) walks the sections array, accumulates
-- `heading || ' ' || text` per element separated by spaces, hard-caps the
-- accumulator at 900 000 characters (≈880 KB UTF-8 bytes — leaves headroom
-- for tsvector compression to stay under the 1 MB ceiling on virtually all
-- inputs), then runs to_tsvector('english', ...) once.
--
-- TRADE-OFFS vs. v1's GENERATED expression:
--   - Slight content loss on the long tail (papers with >900 KB combined
--     section text). p95 sections=124 with avg ~5 KB each ≈ 620 KB which
--     fits cleanly. Outliers (max=2901 sections) get truncated to the head;
--     those papers' tail-section content is unindexed.
--   - Recoverability: a parser bump can re-run safe_sections_tsv on the
--     affected rows without table-rewrite gymnastics.
--   - Headings concatenated with body text (no setweight). Heading-weighting
--     is a follow-up; v1 made the same call.
--
-- SAFETY
-- ------
-- papers_fulltext is LOGGED (asserted in migrations 041, 047). ADD COLUMN
-- does not change persistence. No safety assertion needed.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS + CREATE OR REPLACE FUNCTION.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Nullable tsvector column (heap not rewritten)
-- ---------------------------------------------------------------------------
ALTER TABLE papers_fulltext
    ADD COLUMN IF NOT EXISTS sections_tsv tsvector;

COMMENT ON COLUMN papers_fulltext.sections_tsv IS
    'tsvector over heading + text from sections JSONB. Backs the BM25 leg of '
    'section-grain retrieval. Bead scix_experiments-zpm4. Populated via the '
    'safe_sections_tsv() helper and scripts/backfill_sections_tsv.py — see '
    'migration 063 header for the rewrite-vs-backfill rationale.';

-- ---------------------------------------------------------------------------
-- 2. Helper function — bounded tsvector builder
-- ---------------------------------------------------------------------------
-- Walks sections JSONB array, accumulates heading + text per element, caps
-- at 900 000 chars (≈880 KB) before tokenization to stay clear of the 1 MB
-- tsvector ceiling. Returns empty tsvector on NULL or non-array input.
CREATE OR REPLACE FUNCTION safe_sections_tsv(j jsonb)
RETURNS tsvector AS $$
DECLARE
    accumulated text := '';
    s           jsonb;
    chunk       text;
    max_chars   int := 900000;
BEGIN
    IF j IS NULL OR jsonb_typeof(j) <> 'array' THEN
        RETURN ''::tsvector;
    END IF;

    FOR s IN SELECT * FROM jsonb_array_elements(j) LOOP
        IF length(accumulated) >= max_chars THEN
            EXIT;
        END IF;
        chunk := coalesce(s->>'heading', '') || ' ' || coalesce(s->>'text', '') || ' ';
        accumulated := accumulated || chunk;
    END LOOP;

    IF length(accumulated) > max_chars THEN
        accumulated := left(accumulated, max_chars);
    END IF;

    RETURN to_tsvector('english', accumulated);
EXCEPTION WHEN program_limit_exceeded THEN
    -- Belt-and-suspenders: extreme outliers compress poorly. Halve the input
    -- and retry; if that still busts, give up on this row's tsvector rather
    -- than poison a whole batch.
    BEGIN
        accumulated := left(accumulated, max_chars / 2);
        RETURN to_tsvector('english', accumulated);
    EXCEPTION WHEN program_limit_exceeded THEN
        RETURN ''::tsvector;
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL UNSAFE;
-- PARALLEL UNSAFE because the BEGIN/EXCEPTION block opens a subtransaction,
-- which is forbidden inside parallel workers. The backfill driver runs
-- serially against a single connection so this is not a throughput regression.

COMMENT ON FUNCTION safe_sections_tsv(jsonb) IS
    'Build a tsvector over a papers_fulltext.sections JSONB array, capping '
    'raw-text input at 900 KB to stay under PostgreSQL''s 1 MB tsvector '
    'limit. Drops trailing sections on overflow rather than failing. Used '
    'by the section_bm25 backfill — see migration 063.';

INSERT INTO schema_migrations (version, filename)
    VALUES (63, '063_section_bm25.sql')
    ON CONFLICT (version) DO NOTHING;

COMMIT;
