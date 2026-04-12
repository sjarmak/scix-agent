# Research — u01-schema-migrations

## Existing migration numbering

Versions 001–027 already exist. The spec's wording assumed the next numbers were
026–029, but those slots are taken:

- `026_spdf_spase_crosswalk.sql`
- `027_per_model_hnsw_rebuild.sql`

`tests/test_migration_runner.py::test_all_migrations_contiguous` enforces NO GAPS
and `test_016_is_query_log` pins 016. So the new migrations MUST use 028–031 and
be contiguous. Re-mapping:

| Spec number | Actual number | Purpose                         |
| ----------- | ------------- | ------------------------------- |
| 026         | 028           | entity_schema_hardening         |
| 027         | 029           | ontology_version_pinning        |
| 028         | 030           | staging_and_promote_harvest     |
| 029         | 031           | query_log (extend existing 016) |

## Current `document_entities` schema

PK: `(bibcode, entity_id, link_type)`. No `tier` or `tier_version` columns yet.
We must drop and recreate the PK including `tier`. Test DB is empty, so
`ALTER TABLE ... DROP CONSTRAINT ... ADD CONSTRAINT` is safe even with data.
The migration uses `ADD COLUMN IF NOT EXISTS tier SMALLINT NOT NULL DEFAULT 0`
(default required for existing rows; NOT NULL is safe after backfill).

## Current `entities` schema

Has: id (SERIAL int), canonical_name, entity_type, discipline, source,
harvest_run_id, properties, created_at, updated_at. No `ambiguity_class`,
`link_policy`, `source_version`, or `supersedes_id` yet.

- `entities.id` is `integer` (SERIAL), not BIGINT. The spec asks for
  `supersedes_id BIGINT` but FK type must match → use `INTEGER` to match
  `entities.id`. Document this as a conscious deviation.
- `ambiguity_class` and `link_policy` must become ENUM types (per spec values).

## Current `harvest_runs` schema

Standard table with status TEXT DEFAULT 'running'. Zombie cleanup just needs
to filter by `status='running' AND started_at < now() - interval '6 hours'`.

## Current `query_log` schema (from 016)

Already exists with columns: `id SERIAL PK, tool_name, params_json, latency_ms,
success, error_msg, created_at`. We can't recreate it. Instead: `ADD COLUMN IF
NOT EXISTS` for the new columns (`ts, tool, query, result_count, session_id,
is_test`). The acceptance criterion checks column _existence_, not that the
table is fresh. The existing `id` is INT not BIGSERIAL; we cannot change without
breaking FKs, but nothing references it, so we could alter. For simplicity and
safety we keep `id INT PRIMARY KEY` and add the new columns.

## Staging tables (030)

Production `entities_staging`, `entity_aliases_staging`, `entity_identifiers_staging`
tables do NOT currently exist in public schema. (There is a `staging` SCHEMA
with `staging.entities`, etc., from migration 022.) The spec wants public-schema
tables named `<name>_staging` with `staging_run_id` added. Create fresh.

## Helpers

`tests/helpers.py` provides `is_production_dsn()` and `get_test_dsn()` — use
these to skip tests if `SCIX_TEST_DSN` is unset or points at production.

## Migration style

All existing migrations use `BEGIN; ... COMMIT;`, `CREATE TABLE IF NOT EXISTS`,
`ADD COLUMN IF NOT EXISTS`, and `DO $$ ... EXCEPTION WHEN duplicate_object $$`
for enums. Follow the same style.
