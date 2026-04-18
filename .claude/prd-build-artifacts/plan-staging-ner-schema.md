# Plan — staging-ner-schema

## Deliverables

1. `migrations/049_staging_ner_extractions.sql`
2. `scripts/promote_staging_extractions.py`
3. `tests/test_promote_staging_extractions.py`

## Migration 049 — schema design

### 049a. `staging.extractions` (extend existing migration-015 table)

Provenance adds (ADD COLUMN IF NOT EXISTS):
- `source TEXT`                    — upstream producer, e.g. `"ner_v1"`
- `confidence_tier SMALLINT`       — 1=canonical, 2=probable, 3=candidate
- `extraction_version` — already present (TEXT NOT NULL)
- `created_at`         — already present (TIMESTAMPTZ DEFAULT NOW())

### 049b. `staging.extraction_entity_links` — NEW partitioned table

```sql
CREATE TABLE IF NOT EXISTS staging.extraction_entity_links (
    id                  BIGSERIAL,
    extraction_id       BIGINT,         -- FK semantics only; no hard FK
    bibcode             TEXT    NOT NULL,
    entity_type         TEXT    NOT NULL,
    entity_id           INT,            -- soft-link to entities.id; nullable
    entity_surface      TEXT    NOT NULL,     -- the surface form we matched
    entity_canonical    TEXT,                  -- resolved canonical (optional)
    span_start          INT,
    span_end            INT,
    source              TEXT    NOT NULL,
    confidence_tier     SMALLINT NOT NULL,
    confidence          REAL,
    extraction_version  TEXT    NOT NULL,
    payload             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, entity_type)
) PARTITION BY LIST (entity_type);
```

Primary key must include the partition key. `BIGSERIAL id` + `entity_type`
as a composite PK is idiomatic for partitioned tables.

Partitions (CREATE IF NOT EXISTS):
- `staging.extraction_entity_links_software`  — `FOR VALUES IN ('software')`
- `staging.extraction_entity_links_instrument` — `FOR VALUES IN ('instrument')`
- `staging.extraction_entity_links_dataset`    — `FOR VALUES IN ('dataset')`
- `staging.extraction_entity_links_method`     — `FOR VALUES IN ('method')`
- `staging.extraction_entity_links_default`    — `DEFAULT`

Indexes on parent (inherited):
- `(bibcode)` — lookup by paper
- `(source, confidence_tier, extraction_version)` — promotion filter
- `(created_at)` — time-based partition pruning & cleanup

### 049c. `public.extraction_entity_links` — target for promotion

Create `public.extraction_entity_links` with the same columns (minus
partition constraint) + FK to `papers(bibcode)` so promotion has somewhere
to land. Use `CREATE TABLE IF NOT EXISTS`. Add a UNIQUE constraint on
`(bibcode, entity_type, entity_surface, extraction_version, source)` so
the promotion script can use `ON CONFLICT DO NOTHING`.

### 049d. Safety assertions (DO block)

Assert LOGGED for every table we created:
- `staging.extractions`
- `staging.extraction_entity_links`
- All four list partitions + default partition
- `public.extraction_entity_links`

If `relpersistence <> 'p'`, `RAISE EXCEPTION`.

### 049e. Idempotency verified by

- `CREATE SCHEMA IF NOT EXISTS`
- `CREATE TABLE IF NOT EXISTS`
- `ADD COLUMN IF NOT EXISTS`
- `CREATE INDEX IF NOT EXISTS`
- Partition CREATE TABLE uses IF NOT EXISTS clauses

## promote_staging_extractions.py

```
usage: promote_staging_extractions.py [-h] [--dsn DSN]
                                      [--batch-size N] [--dry-run]
                                      [--source-filter SOURCE]
```

Public API:

```python
def promote(
    conn: psycopg.Connection,
    batch_size: int = 10_000,
    dry_run: bool = False,
    source_filter: str | None = None,
) -> PromotionCounts:
    """Promote rows from staging.extractions and
    staging.extraction_entity_links into their public counterparts.

    Returns a dataclass with counts. Rolls back if dry_run.
    """
```

Implementation:
- Open transaction.
- `INSERT INTO public.extractions (bibcode, extraction_type,
   extraction_version, payload, source, confidence_tier, created_at)
   SELECT ... FROM staging.extractions WHERE 1=1 [AND source = %s]
   LIMIT %s
   ON CONFLICT (bibcode, extraction_type, extraction_version)
   DO NOTHING`
- Capture rowcount via `RETURNING 1` pattern.
- Same for `extraction_entity_links`.
- Commit or rollback depending on `dry_run`.

Note: `public.extractions` does NOT currently have `source`/`confidence_tier`
columns. The promotion script will NOT select them for that table (only
core columns). The provenance columns exist on staging, but the canonical
public.extractions schema is the migration-001 shape. The
extraction_entity_links public table (new, created in 049) DOES carry the
provenance columns, so they do get promoted there.

## Test plan

`tests/test_promote_staging_extractions.py`:

- `pytestmark = pytest.mark.skipif(get_test_dsn() is None, ...)` — skip
  cleanly when SCIX_TEST_DSN is absent or points at production.
- Module-scope fixture applies migration 049 against the test DB.
- Per-test savepoint fixture isolates mutations.
- Tests:
  - `test_migration_is_idempotent` — apply 049 twice; no exceptions.
  - `test_staging_tables_are_logged` — query `pg_class.relpersistence`.
  - `test_promote_extractions_roundtrip` — insert 3 staging.extractions
    rows; call `promote()`; assert they land in public.extractions.
  - `test_promote_entity_links_roundtrip` — insert 3 rows into
    staging.extraction_entity_links (one per partition); assert they
    land in public.extraction_entity_links.
  - `test_dry_run_rolls_back` — dry_run=True leaves public unchanged.
  - `test_on_conflict_do_nothing` — re-promoting same rows is a no-op.

## Commit message

`prd-build: staging-ner-schema — M4 staging schema + promotion script`
