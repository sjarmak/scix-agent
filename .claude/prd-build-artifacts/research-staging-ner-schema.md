# Research — staging-ner-schema (Migration 049)

## Objective

Add migration 049 that creates a `staging` schema with:
- `staging.extractions` — mirrors `public.extractions` plus provenance columns
- `staging.extraction_entity_links` — partitioned by `entity_type`
  (software / instrument / dataset / method + default partition)

Plus `scripts/promote_staging_extractions.py` that batch-loads canonical
results from staging into `public.extractions` with `ON CONFLICT DO NOTHING`,
and a test with SCIX_TEST_DSN guard.

## Existing schema (from migration 001)

`public.extractions`:
```sql
CREATE TABLE extractions (
    id                 SERIAL PRIMARY KEY,
    bibcode            TEXT NOT NULL REFERENCES papers(bibcode),
    extraction_type    TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    payload            JSONB NOT NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
```

Migration 009 adds a UNIQUE constraint
`(bibcode, extraction_type, extraction_version)` named
`uq_extractions_bibcode_type_version` and a GIN index on `payload`.

## Existing staging (from migration 015)

Migration 015 already created a basic `staging.extractions` without the
provenance columns (`source`, `confidence_tier`, `extraction_version` is
there but not `extraction_version` interpreted as provenance). It also
installed `staging.promote_extractions()` as an SQL function that upserts
and truncates. It does NOT have an `extraction_entity_links` table.

Migration 049 must be **additive** and **idempotent** over 015:
- Re-use `CREATE SCHEMA IF NOT EXISTS staging` (no-op if already exists)
- Use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on `staging.extractions`
  for the new provenance columns
  (`source`, `confidence_tier`, `extraction_version` already there)
- Create the new `staging.extraction_entity_links` partitioned table
  with `CREATE TABLE IF NOT EXISTS`

## Existing public.extraction_entity_links

There is NO existing `public.extraction_entity_links` table. Migration
049 must therefore ALSO create the public table so the promotion target
exists. The shape is implied by the work unit spec (bibcode + entity
identifiers + provenance). I'll mirror it against the staging table.

## Conventions observed in recent migrations (041-048)

1. **Header** `-- Migration NNN: <short description>` or
   `-- NNN_filename.sql\n-- <description>`. Either is acceptable; I'll use
   the explicit "Migration 049:" form matching 048.
2. **Block comment** describing purpose, idempotency strategy, and any
   safety assertions.
3. **BEGIN / COMMIT** wrapping the whole migration.
4. **Idempotent DDL** — `CREATE SCHEMA IF NOT EXISTS`,
   `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`,
   `CREATE INDEX IF NOT EXISTS`.
5. **LOGGED assertion** — a trailing DO block that queries
   `pg_class.relpersistence` and `RAISE EXCEPTION` if any table is `'u'`
   (unlogged) — per migration 048 and the
   `feedback_unlogged_tables` memory (32M embeddings lost).
6. **Partitioning** — migration 034 demonstrates `PARTITION BY RANGE`
   with a DEFAULT partition. For `PARTITION BY LIST (entity_type)` the
   syntax is `PARTITION OF parent FOR VALUES IN ('software')`.

## SCIX_TEST_DSN test pattern

From `tests/test_promote_harvest.py` and `tests/helpers.py`:

```python
from tests.helpers import get_test_dsn
from scix.db import is_production_dsn

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

pytestmark = pytest.mark.skipif(
    TEST_DSN is None or is_production_dsn(TEST_DSN),
    reason="test requires SCIX_TEST_DSN pointing at a non-production DB",
)
```

The canonical `get_test_dsn()` helper in `tests/helpers.py` combines both
checks and returns `None` when the DSN is missing or production — that is
the pattern used by `test_migrations.py`. I will use this helper.

## db.py helpers

- `DEFAULT_DSN` — resolves `SCIX_DSN` env or `dbname=scix`.
- `is_production_dsn(dsn)` — checks for `dbname=scix` in various DSN
  forms.
- `get_connection(dsn, autocommit)` — returns a psycopg connection.
- `redact_dsn(dsn)` — safe for logging.

The promotion script will accept `--dsn`, default to `DEFAULT_DSN`, and
use `get_connection()`.

## No existing `scripts/promote_*.py`

There are no other promotion scripts on disk — the existing staging
promotion logic lives in SQL (`staging.promote_extractions()`) and is
invoked via `SELECT`. The new Python script is the first of its kind.

## Partition key

Per the work unit spec: `PARTITION BY LIST (entity_type)` with values
`software`, `instrument`, `dataset`, `method`, plus a DEFAULT partition
for unexpected types.

## Provenance columns (required by AC #4)

All staging tables must include:
- `source TEXT`
- `confidence_tier SMALLINT`
- `extraction_version TEXT`
- `created_at TIMESTAMPTZ DEFAULT now()`

`staging.extractions` already has `extraction_version` and `created_at`
from migration 015. I will `ALTER TABLE ADD COLUMN IF NOT EXISTS` to add
`source` and `confidence_tier`.

## Test DB availability

`psql -d scix_test -c "SELECT 1"` succeeds in this worktree, so the
migration and integration test can be exercised against `scix_test`.
