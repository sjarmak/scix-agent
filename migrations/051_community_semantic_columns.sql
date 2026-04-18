-- migration: 051_community_semantic_columns — semantic-communities work unit (M2)
--
-- PRD: docs/prd/semantic_communities.md (semantic-communities work unit)
--
-- Adds three INT columns to paper_metrics that record per-paper semantic
-- community assignments from minibatch k-means over INDUS embeddings. The
-- three resolutions mirror the existing citation-graph Leiden communities
-- (coarse / medium / fine) but use vector clustering rather than graph
-- modularity:
--
--     community_semantic_coarse   — k=20
--     community_semantic_medium   — k=200
--     community_semantic_fine     — k=2000
--
-- Btree indexes are created on each column so community-membership lookups
-- (e.g. "give me all papers in the same semantic-coarse cluster as X") run
-- without a full scan.  These are low-cardinality-friendly btrees — bitmap
-- index scans kick in naturally for point lookups.
--
-- Idempotent: every DDL uses IF NOT EXISTS so the migration can be re-run
-- against databases that already have some or all of the columns/indexes.
-- The final DO block asserts that the columns exist so a silent failure
-- (e.g. a typo that creates the wrong column name) cannot go unnoticed.
--
-- Dependencies:
--   - paper_metrics table (created in migration 008 / 012; no PK change).
--   - paper_embeddings with model_name='indus' rows (no DDL dependency,
--     read-only at script-execution time).

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Columns: three semantic community IDs on paper_metrics
-- ---------------------------------------------------------------------------

ALTER TABLE paper_metrics
    ADD COLUMN IF NOT EXISTS community_semantic_coarse INT;

ALTER TABLE paper_metrics
    ADD COLUMN IF NOT EXISTS community_semantic_medium INT;

ALTER TABLE paper_metrics
    ADD COLUMN IF NOT EXISTS community_semantic_fine INT;

-- ---------------------------------------------------------------------------
-- 2. Btree indexes on each semantic column
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_pm_community_semantic_coarse
    ON paper_metrics (community_semantic_coarse);

CREATE INDEX IF NOT EXISTS idx_pm_community_semantic_medium
    ON paper_metrics (community_semantic_medium);

CREATE INDEX IF NOT EXISTS idx_pm_community_semantic_fine
    ON paper_metrics (community_semantic_fine);

-- ---------------------------------------------------------------------------
-- 3. Post-migration invariant — columns + indexes must exist
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    missing_col  TEXT;
    missing_idx  TEXT;
BEGIN
    SELECT c
      INTO missing_col
      FROM (VALUES
          ('community_semantic_coarse'),
          ('community_semantic_medium'),
          ('community_semantic_fine')
      ) AS v(c)
      WHERE NOT EXISTS (
          SELECT 1 FROM information_schema.columns
           WHERE table_schema = 'public'
             AND table_name   = 'paper_metrics'
             AND column_name  = v.c
      )
      LIMIT 1;

    IF missing_col IS NOT NULL THEN
        RAISE EXCEPTION
            '051: paper_metrics is missing semantic community column %',
            missing_col;
    END IF;

    SELECT i
      INTO missing_idx
      FROM (VALUES
          ('idx_pm_community_semantic_coarse'),
          ('idx_pm_community_semantic_medium'),
          ('idx_pm_community_semantic_fine')
      ) AS v(i)
      WHERE NOT EXISTS (
          SELECT 1 FROM pg_indexes
           WHERE schemaname = 'public'
             AND tablename  = 'paper_metrics'
             AND indexname  = v.i
      )
      LIMIT 1;

    IF missing_idx IS NOT NULL THEN
        RAISE EXCEPTION
            '051: paper_metrics is missing semantic community index %',
            missing_idx;
    END IF;
END
$$;

COMMIT;
