# Test Results — staging-ner-schema

## Migration applied to scix_test

```
$ psql -d scix_test -v ON_ERROR_STOP=1 -f migrations/049_staging_ner_extractions.sql
BEGIN
... CREATE SCHEMA / CREATE TABLE / CREATE INDEX / ADD COLUMN
... DO (LOGGED assertion passes)
COMMIT
exit 0
```

Re-applied a second time: exit 0, all statements no-op via
IF NOT EXISTS / ADD COLUMN IF NOT EXISTS / DO block guard.

## pytest with SCIX_TEST_DSN set

```
$ SCIX_TEST_DSN=dbname=scix_test \\
    .venv/bin/python -m pytest tests/test_promote_staging_extractions.py -v
...
tests/test_promote_staging_extractions.py::test_migration_is_idempotent PASSED
tests/test_promote_staging_extractions.py::test_staging_tables_are_logged PASSED
tests/test_promote_staging_extractions.py::test_entity_links_is_partitioned_by_list PASSED
tests/test_promote_staging_extractions.py::test_provenance_columns_on_staging_extractions PASSED
tests/test_promote_staging_extractions.py::test_promote_extractions_roundtrip PASSED
tests/test_promote_staging_extractions.py::test_promote_entity_links_roundtrip PASSED
tests/test_promote_staging_extractions.py::test_dry_run_rolls_back PASSED
tests/test_promote_staging_extractions.py::test_on_conflict_do_nothing PASSED
tests/test_promote_staging_extractions.py::test_source_filter_restricts_promotion PASSED

============================== 9 passed in 1.40s ===============================
```

## pytest without SCIX_TEST_DSN (skip-clean check)

```
$ unset SCIX_TEST_DSN && .venv/bin/python -m pytest tests/test_promote_staging_extractions.py -v
...
9 skipped in 0.12s
```

All 9 tests skip cleanly; 0 failures.

## Acceptance criteria verification

| AC | Status | Evidence |
| -- | ------ | -------- |
| 1  | PASS   | migrations/049_staging_ner_extractions.sql exists, uses CREATE SCHEMA/TABLE IF NOT EXISTS throughout |
| 2  | PASS   | schema `staging` created; tables `staging.extractions` + `staging.extraction_entity_links` created |
| 3  | PASS   | `staging.extraction_entity_links` is PARTITION BY LIST (entity_type); partitions: software, instrument, dataset, method + DEFAULT |
| 4  | PASS   | staging tables have source TEXT, confidence_tier SMALLINT, extraction_version TEXT, created_at TIMESTAMPTZ DEFAULT now() |
| 5  | PASS   | `psql -d scix_test -f migrations/049_...sql` exits 0 on first and second application |
| 6  | PASS   | `scripts/promote_staging_extractions.py` has `promote(batch_size, dry_run, source_filter)` using INSERT...SELECT ON CONFLICT DO NOTHING |
| 7  | PASS   | `tests/test_promote_staging_extractions.py` uses `get_test_dsn()` guard, inserts mock staging rows, asserts public rows |
| 8  | PASS   | 9 pass with SCIX_TEST_DSN set; 9 skip with 0 failures when unset |
