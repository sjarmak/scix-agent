-- Migration 068: OA/preprint gate for body-AI pipelines
--
-- Bead: scix_experiments-8584
--
-- Publisher-agreement risk surface: applying AI tools (NER, embeddings,
-- citation-context extraction) to non-OA body text is constrained by Wiley /
-- Springer / etc. TDM clauses. This migration installs a single source of
-- truth — an IMMUTABLE SQL function plus a partial expression index — that
-- body-AI scripts call in their WHERE clauses to gate themselves to OA or
-- preprint papers.
--
-- WHY EXPRESSION INDEX, NOT GENERATED COLUMN
-- ------------------------------------------
-- An earlier draft proposed `ALTER TABLE papers ADD COLUMN is_oa_or_preprint
-- BOOLEAN GENERATED ALWAYS AS (...) STORED`. Rejected: PG16 ADD COLUMN with
-- a STORED generated expression requires a full table rewrite (every row's
-- expression must be computed and stored), holding AccessExclusiveLock for
-- the duration. On 32M papers with ~40-column metadata + JSONB raw + GIN
-- indexes, that's multi-hour blocking — worse than building the index here.
-- Migration 053 took the same lesson: nullable shadow column + out-of-band
-- backfill. We go further and avoid the column entirely; the planner's
-- expression-index matching gives the same query plan as a STORED column
-- would have, with no rewrite cost and no backfill.
--
-- IDEMPOTENCY
-- -----------
-- - CREATE OR REPLACE FUNCTION is idempotent.
-- - CREATE INDEX CONCURRENTLY IF NOT EXISTS is idempotent.
--
-- TRANSACTION SHAPE
-- -----------------
-- This file has NO BEGIN/COMMIT — CREATE INDEX CONCURRENTLY cannot run
-- inside a transaction. Migration runner (scripts/migrate.py) executes
-- such files with autocommit; convention established in migration 054.
--
-- COST ESTIMATE (informational; verify before running on prod)
-- ------------------------------------------------------------
-- - Function creation: instant (~ms).
-- - Index build: ~5-10 minutes wall clock for 14.95M papers with body
--   (the partial-index population). Comparable to a btree on a small
--   boolean predicate; far cheaper than a GIN.
-- - Disk: ~150 MB for the partial index (bead 8584 estimate).
-- - Lock: CONCURRENTLY takes ShareUpdateExclusiveLock — does not block
--   reads or writes, only other DDL on the same table. Safe to run in
--   the foreground without scix-batch wrapping.
-- - If CREATE INDEX CONCURRENTLY fails midway, the index is left INVALID.
--   Drop it and retry:
--     DROP INDEX CONCURRENTLY IF EXISTS idx_papers_is_oa;
--
-- POST-VERIFICATION
-- -----------------
--   SELECT count(*) FILTER (WHERE papers_is_oa_or_preprint(papers)) AS oa,
--          count(*) FILTER (WHERE NOT papers_is_oa_or_preprint(papers)) AS closed,
--          count(*) AS total
--     FROM papers
--    WHERE body IS NOT NULL;
--   -- Expect: oa ≈ 5.4M, closed ≈ 9.5M, total ≈ 14.95M (per bead 8584).

-- ---------------------------------------------------------------------------
-- The predicate function
-- ---------------------------------------------------------------------------
-- IMMUTABLE: required for use in an index expression.
-- PARALLEL SAFE: lets parallel sequential scans evaluate it without a
-- planner-imposed worker cap.
-- COALESCE: both branches return NULL on NULL inputs (`ANY(NULL)` and
-- `array_length(NULL, 1)` are NULL); wrapping each in COALESCE makes the
-- result strictly BOOLEAN so callers using `WHERE NOT papers_is_oa_or_preprint(p)`
-- (e.g. diagnostics for closed-access coverage) get the same population
-- they'd get from `WHERE papers_is_oa_or_preprint(p) = FALSE`. Without the
-- COALESCE, `NOT NULL` is NULL and the closed-access query would silently
-- drop rows whose property and arxiv_class are both NULL.

CREATE OR REPLACE FUNCTION papers_is_oa_or_preprint(p papers)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT COALESCE('OPENACCESS' = ANY(p.property), FALSE)
        OR COALESCE(array_length(p.arxiv_class, 1) > 0, FALSE);
$$;

COMMENT ON FUNCTION papers_is_oa_or_preprint(papers) IS
    'OA/preprint gate for body-AI pipelines. TRUE iff property contains OPENACCESS '
    'or arxiv_class is non-empty. Single source of truth — body-AI scripts call '
    'this in WHERE clauses; matching index is idx_papers_is_oa.';

-- ---------------------------------------------------------------------------
-- Partial expression index
-- ---------------------------------------------------------------------------
-- Indexes only papers with body — every body-AI pipeline filters
-- `body IS NOT NULL` already, so the partial form keeps the index small
-- (~150 MB vs ~250 MB for the full-table form) without losing any matches.
-- Index expression must be syntactically the same shape the planner sees
-- in WHERE clauses; PG inlines simple SQL functions, so callers writing
-- `WHERE papers_is_oa_or_preprint(p)` will hit this index.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_is_oa
    ON papers ((papers_is_oa_or_preprint(papers)))
    WHERE body IS NOT NULL;
