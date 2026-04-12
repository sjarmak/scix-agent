# Test Results — u07 query_log + curated core

Command:

```
SCIX_TEST_DSN=dbname=scix_test pytest tests/test_query_log.py tests/test_curated_core.py -v
```

Result: **6 passed in 0.19s**

```
tests/test_query_log.py::test_log_query_writes_row_with_recent_ts PASSED
tests/test_query_log.py::test_log_query_zero_result PASSED
tests/test_query_log.py::test_backfill_inserts_rows_from_fixture PASSED
tests/test_curated_core.py::test_curate_entity_core_three_pass PASSED
tests/test_curated_core.py::test_promote_auto_demotes_at_cap PASSED
tests/test_curated_core.py::test_demote_removes_and_logs PASSED
```

## Acceptance criteria mapping

- **AC1** — `test_log_query_writes_row_with_recent_ts` asserts a row written
  with `ts` within 30s of `now()`. Legacy NOT NULL columns
  (`tool_name`, `success`) are also populated.
- **AC2** — `test_backfill_inserts_rows_from_fixture` writes a 7-line fixture
  JSONL (5 valid records + 1 malformed + 1 missing `tool`) and asserts 5
  rows inserted.
- **AC3** — `test_curate_entity_core_three_pass` runs the three-pass
  ranking against seeded fixture data, asserts the CSV and
  stratification file exist, the 7 required CSV columns are present,
  pass-1 gap candidates and pass-3 unique+hits rows are all present,
  and excluded ambiguity classes (homograph, domain_safe) are NOT in
  the core.
- **AC4** — `test_promote_auto_demotes_at_cap` monkeypatches `CORE_MAX=5`,
  fills the core, then promotes a 6th entity and asserts (a) total size
  ≤ CORE_MAX after promote, (b) at least one `auto_demote_cap` event
  in `core_promotion_log`, (c) the new entity is in the core.
- **AC5** — `test_curate_entity_core_three_pass` asserts the
  stratification markdown has a per-source header and contains at
  least one of the seeded sources.
- **AC6** — all 6 tests pass with 0 failures.

## Artifacts

- `build-artifacts/curated_core.csv` — produced by a direct script run
  against `scix_test` (empty because production query traffic is not
  replayed into the test DB, but the file and header are present).
- `build-artifacts/curated_core_stratification.md` — same.

## Migration

- `migrations/032_core_promotion_log.sql` applied to `scix_test`:
  `BEGIN`, `CREATE TABLE curated_entity_core`, `CREATE TABLE
core_promotion_log`, `COMMIT`.
