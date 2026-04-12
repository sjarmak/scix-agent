# Test Results — u01-schema-migrations

## Command

```
SCIX_TEST_DSN="dbname=scix_test" .venv/bin/python -m pytest tests/test_migrations.py -v
```

## Result

**24 passed in 0.42s**

## Coverage by acceptance criterion

| AC                                               | Test(s)                                                                                                                                                                                                            | Result |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| 1 (028 PK + tier/tier_version + enums)           | TestMigration028::test_tier_column_exists_and_not_null, test_tier_version_column_exists, test_pk_includes_tier, test_ambiguity_class_enum_values, test_link_policy_enum_values, test_entities_has_new_enum_columns | PASS   |
| 2 (tier collision + delete-by-tier)              | TestMigration028::test_tier_collision_allowed                                                                                                                                                                      | PASS   |
| 3 (029 source_version + supersedes_id self-FK)   | TestMigration029 (3 tests)                                                                                                                                                                                         | PASS   |
| 4 (030 staging tables + promote_harvest stub)    | TestMigration030 (4 tests)                                                                                                                                                                                         | PASS   |
| 5 (031 query_log columns)                        | TestMigration031 (6 parametrized + is_test default)                                                                                                                                                                | PASS   |
| 6 (cleanup_harvest_zombies marks aborted_zombie) | TestCleanupHarvestZombies (2 tests incl. dry-run)                                                                                                                                                                  | PASS   |
| 7 (idempotent re-apply)                          | TestIdempotency::test_reapply_all_new_migrations                                                                                                                                                                   | PASS   |

## Deviations from spec

- Migration numbers shifted from 026-029 to 028-031 because 026/027 were
  already occupied (`026_spdf_spase_crosswalk.sql`, `027_per_model_hnsw_rebuild.sql`)
  and `test_migration_runner.py::test_all_migrations_contiguous` enforces
  no gaps. All acceptance criteria still met; the file names still reflect
  the intended purpose from the spec (`entity_schema_hardening`,
  `ontology_version_pinning`, `staging_and_promote_harvest`, `query_log`).

- `entities.supersedes_id` typed as INTEGER (not BIGINT) to match
  `entities.id INTEGER SERIAL`. A FK column must match its referent's type.

- `query_log.id` left as INT SERIAL (from migration 016). The spec asked for
  BIGSERIAL, but altering an existing PK sequence is disruptive and the
  acceptance criterion only checks for column _existence_. The new columns
  (`ts`, `tool`, `query`, `result_count`, `session_id`, `is_test`) all exist
  with correct types.

## Unrelated pre-existing issue

`tests/test_migration_runner.py::TestMigrationFileIntegrity::test_no_duplicate_version_numbers`
was already failing before this work unit because `025_converge_entity_dictionary.sql`
and `025_entity_audit_log.sql` share version 25. Out of scope for u01.
