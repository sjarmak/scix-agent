# Plan — u08-fusion-mv

## Files

1. `migrations/033_fusion_mv.sql`
2. `src/scix/fusion_mv.py`
3. `tests/test_fusion_mv.py`

## Migration 033 structure

```
BEGIN;
-- tier_weight SQL function (IMMUTABLE LEAKPROOF PARALLEL SAFE)
CREATE OR REPLACE FUNCTION tier_weight(tier SMALLINT) RETURNS DOUBLE PRECISION
  LANGUAGE sql IMMUTABLE LEAKPROOF PARALLEL SAFE AS $$
  SELECT CASE tier
    WHEN 1 THEN 0.98::float8
    WHEN 2 THEN 0.85::float8
    WHEN 3 THEN 0.92::float8
    WHEN 4 THEN 0.50::float8
    WHEN 5 THEN 0.88::float8
    ELSE 0.50::float8
  END
$$;

-- Calibration log
CREATE TABLE IF NOT EXISTS tier_weight_calibration_log (
  id SERIAL PRIMARY KEY,
  version TEXT NOT NULL UNIQUE,
  weights JSONB NOT NULL,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
INSERT ... 'placeholder_2026-04-12' ... ON CONFLICT DO NOTHING;

-- Dirty state table
CREATE TABLE IF NOT EXISTS fusion_mv_state (
  id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  dirty BOOL NOT NULL DEFAULT true,
  last_refresh_at TIMESTAMPTZ
);
INSERT INTO fusion_mv_state (id) VALUES (1) ON CONFLICT DO NOTHING;

-- MV (drop + recreate idempotent)
DROP MATERIALIZED VIEW IF EXISTS document_entities_canonical CASCADE;
CREATE MATERIALIZED VIEW document_entities_canonical AS
SELECT
  bibcode,
  entity_id,
  1 - exp(sum(ln(1 - LEAST(0.9999, GREATEST(0.0, confidence::float8 * tier_weight(tier))))))
    AS fused_confidence,
  array_agg(DISTINCT tier ORDER BY tier) AS tiers,
  count(*) AS link_count,
  max(tier_version) AS max_tier_version
FROM document_entities
WHERE confidence IS NOT NULL
GROUP BY bibcode, entity_id;

CREATE UNIQUE INDEX idx_dec_bibcode_entity
  ON document_entities_canonical (bibcode, entity_id);
CREATE INDEX idx_dec_entity_confidence
  ON document_entities_canonical (entity_id, fused_confidence DESC);
CREATE INDEX idx_dec_bibcode
  ON document_entities_canonical (bibcode);
COMMIT;
```

## Python helper `src/scix/fusion_mv.py`

- `mark_dirty(conn=None) -> None`
- `refresh_if_due(conn=None, min_interval_seconds=3600) -> bool`
- Both open their own connection if not given; refresh path uses autocommit because REFRESH CONCURRENTLY cannot run in a transaction.
- No SELECT from MV — lint-safe.

## Tests

Structure:

- `requires_test_dsn` skip marker using `get_test_dsn()`.
- Module fixture: apply migration 033 (idempotent), clean `document_entities` / `entities` / `fusion_mv_state`.
- Tests:
  1. `test_migration_file_text` — static checks of migration (function, table, unique index, etc.).
  2. `test_tier_weight_function_values` — SELECT each tier → expected value.
  3. `test_tier_weight_function_immutable` — check proisstrict/provolatile in pg_proc.
  4. `test_calibration_log_has_initial_row` — SELECT version='placeholder_2026-04-12'.
  5. `test_mv_unique_index_exists` — pg_indexes lookup.
  6. `test_mv_matches_closed_form` — seed ≥5 tiers for (bib, entity), REFRESH, compute closed form in Python, compare within 1e-9.
  7. `test_refresh_concurrently_succeeds` — explicit REFRESH CONCURRENTLY call.
  8. `test_mark_dirty_and_refresh_if_due` — mark_dirty, first call returns True, second within interval returns False.
  9. `test_query_latency` — 100 synthetic links on one entity, REFRESH, run SELECT and assert wall-clock <100ms (using `# noqa: resolver-lint` on the SELECT literal).

## Lint safety

- `fusion_mv.py` issues only `REFRESH MATERIALIZED VIEW document_entities_canonical` and `fusion_mv_state` queries. Mark the REFRESH line `# noqa: resolver-lint` just in case the lint regex extends.
- Actually: lint bans only `SELECT ... FROM document_entities_canonical`. REFRESH is not banned. Safe.
- Test file SELECT from MV is marked `# noqa: resolver-lint`.
