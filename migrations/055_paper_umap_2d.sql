-- migration: 055_paper_umap_2d — 2-D UMAP projections of paper embeddings
--
-- Work unit: unit-v3-schema
--
-- Introduces paper_umap_2d, a storage table for 2-D UMAP projections of
-- paper embeddings. One row per projected paper (bibcode PK); the
-- `resolution` column labels which projection run / hyper-parameter set
-- produced the coordinates so multi-resolution projections coexist in a
-- single table (per the optional multi-resolution clause in the PRD —
-- multi-resolution rollouts migrate the PK to (bibcode, resolution) in a
-- later migration; for now one row per paper is the design).
--
-- Schema:
--   bibcode       text  PK, FK -> papers(bibcode) ON DELETE CASCADE
--   x, y          double precision NOT NULL  (projection coordinates)
--   community_id  integer  (optional community label; may be NULL)
--   resolution    text NOT NULL              (projection label / run id)
--   projected_at  timestamptz NOT NULL DEFAULT now()
--
-- Index:
--   ix_paper_umap_2d_resolution_community  ON (resolution, community_id)
--     — supports "all papers in community X at resolution R" lookups for
--     UI/visualization queries that drive the agent-navigation surface.
--
-- Idempotent: every DDL uses IF NOT EXISTS so the migration can be re-run
-- against databases that already have the table / index. The final DO
-- block asserts that the table, PK, FK, and index exist so a silent
-- failure (e.g. a typo in a column name) cannot go unnoticed.
--
-- Dependencies:
--   - papers table (migration 001). FK target is papers(bibcode), type text.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS paper_umap_2d (
    bibcode       text PRIMARY KEY REFERENCES papers(bibcode) ON DELETE CASCADE,
    x             double precision NOT NULL,
    y             double precision NOT NULL,
    community_id  integer,
    resolution    text NOT NULL,
    projected_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE paper_umap_2d IS
    '2-D UMAP projections of paper embeddings, one row per projected paper. '
    'The `resolution` column labels the projection run so multi-resolution '
    'projections can coexist.';

-- ---------------------------------------------------------------------------
-- 2. Index on (resolution, community_id)
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS ix_paper_umap_2d_resolution_community
    ON paper_umap_2d (resolution, community_id);

-- ---------------------------------------------------------------------------
-- 3. Post-migration invariants — table, PK, FK, and index must exist
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    -- Table exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_name   = 'paper_umap_2d'
    ) THEN
        RAISE EXCEPTION '055: paper_umap_2d table is missing';
    END IF;

    -- Primary key is on bibcode
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint c
          JOIN pg_class t ON t.oid = c.conrelid
          JOIN unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
          JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
         WHERE t.relname = 'paper_umap_2d'
           AND t.relnamespace = 'public'::regnamespace
           AND c.contype = 'p'
         GROUP BY c.conname
        HAVING string_agg(a.attname, ',' ORDER BY k.ord) = 'bibcode'
    ) THEN
        RAISE EXCEPTION '055: paper_umap_2d PK must be (bibcode)';
    END IF;

    -- FK to papers(bibcode) exists
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint c
          JOIN pg_class t  ON t.oid  = c.conrelid
          JOIN pg_class rt ON rt.oid = c.confrelid
         WHERE t.relname  = 'paper_umap_2d'
           AND rt.relname = 'papers'
           AND c.contype  = 'f'
    ) THEN
        RAISE EXCEPTION '055: paper_umap_2d is missing FK to papers(bibcode)';
    END IF;

    -- Index on (resolution, community_id)
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = 'public'
           AND tablename  = 'paper_umap_2d'
           AND indexname  = 'ix_paper_umap_2d_resolution_community'
    ) THEN
        RAISE EXCEPTION '055: paper_umap_2d is missing index ix_paper_umap_2d_resolution_community';
    END IF;
END
$$;

COMMIT;
