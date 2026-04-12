# Implementation Plan — u01-schema-migrations

## Numbering deviation

Spec's 026-029 → actual 028-031 (026/027 already taken). Contiguity enforced by
test_migration_runner.

## Step 1 — Migration 028_entity_schema_hardening.sql

- Create ENUM `entity_ambiguity_class` with values: unique, domain_safe, homograph, banned.
- Create ENUM `entity_link_policy` with values: open, context_required, llm_only, banned.
- `ALTER TABLE entities ADD COLUMN IF NOT EXISTS ambiguity_class entity_ambiguity_class`
  (nullable so existing rows are valid).
- `ALTER TABLE entities ADD COLUMN IF NOT EXISTS link_policy entity_link_policy`.
- `ALTER TABLE document_entities ADD COLUMN IF NOT EXISTS tier SMALLINT NOT NULL DEFAULT 0`.
- `ALTER TABLE document_entities ADD COLUMN IF NOT EXISTS tier_version INT NOT NULL DEFAULT 1`.
- `ALTER TABLE document_entities DROP CONSTRAINT IF EXISTS document_entities_pkey`.
- `ALTER TABLE document_entities ADD CONSTRAINT document_entities_pkey PRIMARY KEY (bibcode, entity_id, link_type, tier)`.
- Wrap in `DO $$ ... $$` where needed for idempotent constraint swap.

## Step 2 — Migration 029_ontology_version_pinning.sql

- `ALTER TABLE entities ADD COLUMN IF NOT EXISTS source_version TEXT`.
- `ALTER TABLE entities ADD COLUMN IF NOT EXISTS supersedes_id INTEGER`
  (matches entities.id type).
- Add self-FK: `ALTER TABLE entities ADD CONSTRAINT entities_supersedes_fk
FOREIGN KEY (supersedes_id) REFERENCES entities(id) ON DELETE SET NULL` —
  guarded by DO block to make idempotent.
- Index on (supersedes_id).

## Step 3 — Migration 030_staging_and_promote_harvest.sql

- Create tables in public schema:
  - `entities_staging` — mirrors entities columns + `staging_run_id BIGINT`.
  - `entity_aliases_staging` — mirrors entity_aliases columns + `staging_run_id BIGINT`.
  - `entity_identifiers_staging` — mirrors entity_identifiers columns + `staging_run_id BIGINT`.
  - Use simple indexes on staging_run_id.
- Create function stub `promote_harvest(run_id BIGINT) RETURNS INTEGER`:
  - Returns 0 (stub). Comment noting u04 will implement body.

## Step 4 — Migration 031_query_log.sql

- `ALTER TABLE query_log ADD COLUMN IF NOT EXISTS ts TIMESTAMPTZ DEFAULT now()`.
- `ADD COLUMN IF NOT EXISTS tool TEXT`.
- `ADD COLUMN IF NOT EXISTS query TEXT`.
- `ADD COLUMN IF NOT EXISTS result_count INT`.
- `ADD COLUMN IF NOT EXISTS session_id TEXT`.
- `ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT false`.
- Note: id remains SERIAL INT (cannot change to BIGSERIAL without disruption).

## Step 5 — scripts/cleanup_harvest_zombies.py

- CLI script: connects with DSN from env (SCIX_DSN or first arg), runs
  `UPDATE harvest_runs SET status='aborted_zombie', finished_at=now()
 WHERE status='running' AND started_at < now() - interval '6 hours'
 RETURNING id, source`.
- Print summary: count + per-row tuple.
- Respect `is_production_dsn` — refuse unless `--yes-production` flag (for safety).
  Actually: just follow standard pattern and let operator choose the DSN.

## Step 6 — tests/test_migrations.py

Fixture: `ensure_migrations` — runs setup_db.sh against scix_test (or direct SQL
execution of 028-031). Uses `get_test_dsn()` to skip if no SCIX_TEST_DSN.

Tests:

1. test_migration_028_columns_and_pk — check tier, tier_version, ambiguity_class
   enum values, link_policy enum values, PK is (bibcode, entity_id, link_type, tier).
2. test_tier_collision_allowed — insert two rows differing only in tier,
   expect success; DELETE WHERE tier=2 removes exactly one.
3. test_migration_029_source_version_supersedes — check columns and self-FK.
4. test_migration_030_staging_and_promote — check staging tables & function.
5. test_migration_031_query_log_columns — check new columns.
6. test_cleanup_harvest_zombies — insert zombie row, run script, check status.
7. test_migrations_idempotent — re-apply all 4 migrations; expect no error.

Tests use SAVEPOINT/rollback to not litter DB. For tier_collision test we
insert a test entity + bibcode and clean up at end.

## Step 7 — Apply and test

- Apply migrations directly via psql to scix_test.
- Run `pytest tests/test_migrations.py -v` with `SCIX_TEST_DSN=dbname=scix_test`.
- Fix failures.

## Step 8 — Commit
