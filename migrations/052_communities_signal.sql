-- migration: 052_communities_signal — community-labels work unit (M4)
--
-- PRD: docs/prd/community_labels.md (community-labels work unit)
--
-- Adds a signal column to `communities` so we can distinguish label rows
-- produced by the three community signals (citation Leiden, semantic
-- k-means, taxonomic arXiv-class). Until M4 the `communities` table was
-- implicitly citation-only; the signal column formalises that separation.
--
-- The primary key is replaced to include `signal` so each
-- (signal, resolution, community_id) tuple is uniquely labelled. The
-- previous 2-column PK would collide across signals (e.g. community 1 at
-- resolution 'coarse' can exist for both citation AND semantic signals).
--
-- Idempotent: every DDL is wrapped in DO-IF-NOT-EXISTS style blocks, so
-- the migration can be re-applied against databases at any point in its
-- rollout without error.
--
-- Dependencies:
--   - communities table (created pre-M4). Table may be empty or populated.
--   - paper_metrics columns (citation_id_{coarse,medium,fine},
--     community_semantic_{coarse,medium,fine}, community_taxonomic) exist
--     — read-only at script-execution time; no DDL dependency here.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Column: signal TEXT
-- ---------------------------------------------------------------------------

ALTER TABLE communities
    ADD COLUMN IF NOT EXISTS signal TEXT;

-- ---------------------------------------------------------------------------
-- 2. CHECK constraint: signal IN ('citation','semantic','taxonomic')
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'communities_signal_check'
           AND conrelid = 'public.communities'::regclass
    ) THEN
        ALTER TABLE communities
            ADD CONSTRAINT communities_signal_check
            CHECK (signal IS NULL OR signal IN ('citation','semantic','taxonomic'));
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 3. Back-fill: any pre-existing rows are citation (the only signal
--    that had labels before M4). Idempotent via the WHERE clause.
-- ---------------------------------------------------------------------------

UPDATE communities SET signal = 'citation' WHERE signal IS NULL;

-- ---------------------------------------------------------------------------
-- 4. NOT NULL on signal now that all rows have a value
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name   = 'communities'
           AND column_name  = 'signal'
           AND is_nullable  = 'YES'
    ) THEN
        -- Safe because step 3 has just populated any NULLs.
        ALTER TABLE communities ALTER COLUMN signal SET NOT NULL;
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 5. Primary key: (signal, resolution, community_id). Drop the old 2-col
--    PK if present; create the new 3-col PK if not already present.
--    A UNIQUE constraint on the same columns would also satisfy the PRD
--    requirement; using a PK gives us a clean NOT NULL on signal and an
--    automatic unique index.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    pk_cols TEXT;
BEGIN
    SELECT string_agg(a.attname, ',' ORDER BY k.ord)
      INTO pk_cols
      FROM pg_constraint c
      JOIN pg_class t ON t.oid = c.conrelid
      JOIN unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
      JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
     WHERE t.relname = 'communities'
       AND t.relnamespace = 'public'::regnamespace
       AND c.contype = 'p';

    IF pk_cols IS DISTINCT FROM 'signal,resolution,community_id' THEN
        IF pk_cols IS NOT NULL THEN
            -- Drop whatever PK currently exists (e.g. 'community_id,resolution')
            EXECUTE (
                'ALTER TABLE communities DROP CONSTRAINT '
                || (SELECT conname FROM pg_constraint c
                     JOIN pg_class t ON t.oid = c.conrelid
                    WHERE t.relname = 'communities'
                      AND t.relnamespace = 'public'::regnamespace
                      AND c.contype = 'p'
                    LIMIT 1)
            );
        END IF;
        ALTER TABLE communities
            ADD CONSTRAINT communities_pkey
            PRIMARY KEY (signal, resolution, community_id);
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 6. Post-migration invariants — signal column + 3-col PK must exist
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name   = 'communities'
           AND column_name  = 'signal'
    ) THEN
        RAISE EXCEPTION '052: communities is missing signal column';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint c
          JOIN pg_class t ON t.oid = c.conrelid
          JOIN unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
          JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
         WHERE t.relname = 'communities'
           AND t.relnamespace = 'public'::regnamespace
           AND c.contype = 'p'
         GROUP BY c.conname
        HAVING string_agg(a.attname, ',' ORDER BY k.ord) = 'signal,resolution,community_id'
    ) THEN
        RAISE EXCEPTION '052: communities missing 3-col PK (signal,resolution,community_id)';
    END IF;
END
$$;

COMMIT;
