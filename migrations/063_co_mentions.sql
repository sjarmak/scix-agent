-- Migration 063: co_mentions — entity↔entity co-occurrence within papers.
--
-- Part of epic scix_experiments-dbl (cross-discipline expansion). After
-- dbl.3 (NER pass) populates document_entities for ~20M papers, the
-- cheapest valuable graph product is the co-mention graph: pairs of
-- entities that appear in the same paper.
--
-- Schema choices:
--
--   * One row per *unordered* pair, stored in canonical order
--     (entity_a_id < entity_b_id), enforced by CHECK. This halves storage
--     and removes ambiguity at the cost of forcing query callers to UNION
--     ALL when looking up "top-k partners of X" — they don't know in
--     advance whether X is the smaller or the larger id of the pair.
--
--   * n_papers is the support count: the number of *distinct* bibcodes
--     where both entities co-occur. We do not store individual paper
--     instances here — that would be a join table on the order of
--     hundreds of millions of rows. (paper, entity) lives in
--     document_entities; (paper, paper) lives via citations; this table
--     is strictly the marginal-counted (entity, entity) summary.
--
--   * first_year / last_year track temporal envelope. Useful for filters
--     like "only show partners that co-occurred since 2015" without
--     having to re-scan document_entities + papers.
--
--   * Population strategy filters HAVING n_papers >= 2 to drop singleton
--     coincidences (one paper that happened to mention both X and Y).
--     Single-paper co-mentions are noise relative to repeatable
--     associations and balloon row count by ~5x. The 2+ floor is encoded
--     here as a CHECK so any future ad-hoc INSERT respects it.
--
-- Refresh model:
--
--   The table is rebuilt by scripts/populate_co_mentions.py, which runs
--   a chunked aggregation over document_entities + papers and TRUNCATEs +
--   re-INSERTs the table. See docs/prd/co_mentions.md for full vs
--   incremental refresh discussion. Refresh runs are logged in
--   co_mention_runs.
--
-- Indexes:
--
--   * Primary key (entity_a_id, entity_b_id) enforces uniqueness and
--     supports point lookups for pair strength queries.
--   * (entity_a_id, n_papers DESC) — top-k partners when X is the
--     smaller id of the pair.
--   * (entity_b_id, n_papers DESC) — top-k partners when X is the larger
--     id of the pair. The two indexes cover the symmetric-lookup case via
--     UNION ALL in the query layer.

BEGIN;

-- Note: foreign keys to entities(id) are intentionally omitted from this
-- initial migration because creating a FK takes ShareRowExclusive on the
-- referenced table, which conflicts with long-running CREATE INDEX
-- CONCURRENTLY operations on entities (multi-hour trigram index builds
-- are routine in this project). Referential integrity is instead
-- maintained by the rebuild semantics of scripts/populate_co_mentions.py:
-- the table is TRUNCATEd and re-INSERTed from a JOIN against entities,
-- so any deleted/superseded entity is dropped on the next refresh.
-- A follow-up migration can attach FKs once the entities table is quiet.

CREATE TABLE IF NOT EXISTS co_mentions (
    entity_a_id INTEGER NOT NULL,
    entity_b_id INTEGER NOT NULL,
    n_papers    INTEGER NOT NULL,
    first_year  SMALLINT,
    last_year   SMALLINT,
    PRIMARY KEY (entity_a_id, entity_b_id),
    CONSTRAINT co_mentions_a_lt_b      CHECK (entity_a_id < entity_b_id),
    CONSTRAINT co_mentions_n_papers_ge CHECK (n_papers >= 2),
    CONSTRAINT co_mentions_year_order  CHECK (first_year IS NULL OR last_year IS NULL OR first_year <= last_year)
);

COMMENT ON TABLE co_mentions IS
    'Entity-entity co-mention edges. One row per unordered pair (a<b) with n_papers >= 2. Rebuilt by scripts/populate_co_mentions.py — see docs/prd/co_mentions.md.';

COMMENT ON COLUMN co_mentions.n_papers IS
    'Distinct bibcodes where both entities are linked via document_entities (any match_method).';

COMMENT ON COLUMN co_mentions.first_year IS
    'Earliest papers.year across the supporting bibcodes (NULL when no supporting paper has a year).';

COMMENT ON COLUMN co_mentions.last_year IS
    'Latest papers.year across the supporting bibcodes (NULL when no supporting paper has a year).';

-- Top-k by entity_a_id (X is the smaller id of the pair).
CREATE INDEX IF NOT EXISTS ix_co_mentions_a_npapers
    ON co_mentions (entity_a_id, n_papers DESC);

-- Top-k by entity_b_id (X is the larger id of the pair).
CREATE INDEX IF NOT EXISTS ix_co_mentions_b_npapers
    ON co_mentions (entity_b_id, n_papers DESC);

-- ---------------------------------------------------------------------------
-- co_mention_runs — refresh log
--
-- Each row records a full or incremental refresh of co_mentions: when it
-- ran, how many papers + pair-rows it touched, and the wall-clock cost.
-- Operators read it to detect drift between document_entities and the
-- co-mentions snapshot ("table is N days stale; n_papers may be wrong").
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS co_mention_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    refresh_kind    TEXT NOT NULL,
    n_papers_input  BIGINT,
    n_pairs_output  BIGINT,
    min_n_papers    INTEGER NOT NULL DEFAULT 2,
    git_sha         TEXT,
    notes           TEXT,
    CONSTRAINT co_mention_runs_kind CHECK (refresh_kind IN ('full', 'incremental', 'pilot'))
);

CREATE INDEX IF NOT EXISTS ix_co_mention_runs_started_at
    ON co_mention_runs (started_at DESC);

COMMENT ON TABLE co_mention_runs IS
    'Audit log of co_mentions table rebuilds. Read this to assess staleness vs document_entities.';

-- ---------------------------------------------------------------------------
-- Safety: refuse LOGGED-but-empty illusions. Mirrors migrations 041, 062.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp FROM pg_class
        WHERE relname = 'co_mentions' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'co_mentions did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'co_mentions must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;

    SELECT relpersistence INTO rp FROM pg_class
        WHERE relname = 'co_mention_runs' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'co_mention_runs did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'co_mention_runs must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
