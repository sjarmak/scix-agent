# Research — unit-v3-schema (migration 055_paper_umap_2d)

## Migration style (reading 050–054)

- All migrations are plain `.sql`, executed via `psql ... -v ON_ERROR_STOP=1 -f <file>`.
  - Example runner call: `tests/test_migration_045.py` uses `subprocess.run(["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(path)], ...)`.
- Multi-statement DDL is wrapped in `BEGIN; ... COMMIT;` (051, 052, 053).
  - Exception: 054 uses `CREATE INDEX CONCURRENTLY`, which cannot run in a transaction; therefore no BEGIN/COMMIT. Not relevant here — 055 is a plain table + regular index.
- Idempotency is enforced via `IF NOT EXISTS` on `CREATE TABLE`, `CREATE INDEX`, and `ALTER TABLE ... ADD COLUMN`; constraint additions use a `DO $$ ... IF NOT EXISTS ... $$` guard.
- Post-migration invariant blocks (`DO $$ ... RAISE EXCEPTION ... $$`) are common (051, 052, 053) and help catch silent failures. Worth mirroring for 055.
- File naming: `NNN_short_snake_case.sql` — next number is `055`.

## `papers` table

- Defined in `migrations/001_initial_schema.sql`: `CREATE TABLE IF NOT EXISTS papers (bibcode TEXT PRIMARY KEY, ...)`.
  - So FK target is `papers(bibcode)`, type `text`.

## `scix.db` helpers

- `src/scix/db.py` exposes `is_production_dsn(dsn: str | None) -> bool`.
  - Uses `psycopg.conninfo.conninfo_to_dict` so it handles both `key=value` and URI DSNs.
  - Production db name set = `{"scix"}`. Anything else (e.g. `scix_test`) is non-prod.
- `tests/helpers.py` provides `get_test_dsn()` → returns `SCIX_TEST_DSN` if set AND not production, else `None`. This is the canonical guard used by every destructive migration test in the repo.

## Migration runner

- There is no dedicated `scripts/apply_migration.py` or `scripts/migrate.py` in the repo. Inspected:
  - `ls scripts/ | grep -i migrat` → (empty).
  - `Glob scripts/**/migrat*.py` → no files.
- The pattern used across `tests/test_migration_014.py`, `tests/test_migration_045.py`, `tests/test_halfvec_migration.py` is to invoke the system `psql` via `subprocess.run`. Migration 055 will follow the same pattern.

## Test pattern to mirror (from 045)

- Module-scoped `dsn` fixture calls `get_test_dsn()` and skips if `None`.
- Module-scoped `ensure_migration_applied` fixture applies prerequisite migrations (at minimum `001_initial_schema.sql`) followed by 055, via `psql -v ON_ERROR_STOP=1 -f`.
- Separate `conn` fixture for schema inspection (autocommit).
- Tests query `information_schema.columns`, `pg_indexes`, `pg_constraint`, and `pg_attribute` to assert shape.
- Teardown: `DROP TABLE IF EXISTS paper_umap_2d CASCADE` so `scix_test` stays clean.

## Design for 055_paper_umap_2d

- Table columns (per acceptance criteria):
  - `bibcode text PRIMARY KEY REFERENCES papers(bibcode) ON DELETE CASCADE`
  - `x double precision NOT NULL`
  - `y double precision NOT NULL`
  - `community_id integer` (nullable — not every projected paper is tied to a community)
  - `resolution text NOT NULL` (per-resolution rows; a (bibcode, resolution) pair could repeat if multi-resolution — but the spec says "primary key on bibcode", so one row per paper. Multi-resolution support is "optional" per spec and is represented by `resolution` being a column even though PK is just bibcode.)
  - `projected_at timestamptz NOT NULL DEFAULT now()`
- Index: `CREATE INDEX IF NOT EXISTS ix_paper_umap_2d_resolution_community ON paper_umap_2d (resolution, community_id);`
- Wrapped in `BEGIN/COMMIT` with a final `DO $$` invariant block.

## Unit-test fallback

- Integration test will `pytest.skip()` if `SCIX_TEST_DSN` is unset or points at production. That is the common case in CI / ad-hoc worktrees.
- To still exercise the file in every environment, a plain unit test (no DB) reads `migrations/055_paper_umap_2d.sql` and asserts the expected DDL fragments are present.
