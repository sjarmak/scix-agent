# Research: dict-schema-module

## Patterns Found

### Database Connection

- `scix.db.get_connection(dsn)` returns `psycopg.Connection`
- DSN from `SCIX_DSN` env var, defaults to `dbname=scix`
- Functions accept `conn: psycopg.Connection` as first arg

### Migration Style (007_uat_concepts.sql)

- Wrapped in `BEGIN; ... COMMIT;`
- `CREATE TABLE IF NOT EXISTS` for idempotency
- `CREATE INDEX IF NOT EXISTS` for indexes
- TEXT[] arrays with `DEFAULT '{}'`
- JSONB not yet used in migrations but standard PostgreSQL

### Module Style (uat.py)

- `from __future__ import annotations`
- `psycopg` with `dict_row` row factory for dict returns
- Functions are module-level, take `conn` as first param
- Uses `COPY` for bulk loading, `ON CONFLICT DO UPDATE` for upserts
- Frozen dataclasses for data types

### Test Style (test_uat.py)

- Imports `DSN` from `helpers`
- Unit tests: no DB, test pure logic
- Integration tests: `@pytest.mark.integration`, skip if DB/tables unavailable
- `db_conn` fixture with try/except for connection, table existence check
- Cleanup before and after each test

### helpers.py

- `DSN` constant from env
- Helper functions for checking table/data existence
