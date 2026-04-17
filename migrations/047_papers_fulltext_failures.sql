-- Migration 047: papers_fulltext_failures — negative cache for full-text parse failures.
--
-- Records bibcodes for which full-text parsing failed, along with the parser
-- version, failure reason, attempt count, and a retry_after timestamp that
-- encodes R15 exponential backoff (24h -> 3d -> 7d -> 30d). Harvesters consult
-- this table to avoid re-attempting parses that are not yet due for retry.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' and raises if not.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_fulltext_failures (
    bibcode        TEXT PRIMARY KEY,
    parser_version TEXT NOT NULL,
    failure_reason TEXT,
    attempts       INTEGER NOT NULL DEFAULT 1,
    first_attempt  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_attempt   TIMESTAMPTZ NOT NULL DEFAULT now(),
    retry_after    TIMESTAMPTZ NOT NULL
);

COMMENT ON TABLE papers_fulltext_failures IS
    'Negative cache for full-text parse failures. Each row records a bibcode '
    'whose parse failed, the parser version that failed, the failure reason, '
    'attempt count, and retry_after timestamp encoding R15 exponential backoff '
    '(24h -> 3d -> 7d -> 30d). Harvesters skip rows where now() < retry_after.';

COMMENT ON COLUMN papers_fulltext_failures.parser_version IS
    'Parser version string at the time of failure. Rows produced by an older '
    'parser_version may be ignored/reattempted when a newer parser is deployed.';

COMMENT ON COLUMN papers_fulltext_failures.failure_reason IS
    'Human-readable classification of the failure (e.g. "no_source_found", '
    '"latex_parse_error", "pdf_ocr_failed"). Nullable to allow opaque failures.';

COMMENT ON COLUMN papers_fulltext_failures.attempts IS
    'Number of parse attempts so far. Drives R15 backoff: 1 -> retry_after=+24h, '
    '2 -> +3d, 3 -> +7d, >=4 -> +30d.';

COMMENT ON COLUMN papers_fulltext_failures.retry_after IS
    'Timestamp before which this bibcode should not be reattempted. Encodes '
    'R15 exponential backoff (24h -> 3d -> 7d -> 30d) based on attempts.';

CREATE INDEX IF NOT EXISTS idx_papers_fulltext_failures_retry_after
    ON papers_fulltext_failures (retry_after);

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_fulltext_failures' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_fulltext_failures did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_fulltext_failures must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
