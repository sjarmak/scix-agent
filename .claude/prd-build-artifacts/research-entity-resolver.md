# Research: entity-resolver

## Schema (migration 021)

### entities table

- `id SERIAL PRIMARY KEY`
- `canonical_name TEXT NOT NULL`
- `entity_type TEXT NOT NULL`
- `discipline TEXT`
- `source TEXT NOT NULL`
- `UNIQUE (canonical_name, entity_type, source)`
- Index: `idx_entities_canonical_lower ON entities(lower(canonical_name))`

### entity_aliases table

- `entity_id INT REFERENCES entities(id) ON DELETE CASCADE`
- `alias TEXT NOT NULL`
- `alias_source TEXT`
- `PRIMARY KEY (entity_id, alias)`
- Index: `idx_entity_aliases_lower ON entity_aliases(lower(alias))`

### entity_identifiers table

- `entity_id INT REFERENCES entities(id) ON DELETE CASCADE`
- `id_scheme TEXT NOT NULL`
- `external_id TEXT NOT NULL`
- `is_primary BOOLEAN DEFAULT false`
- `PRIMARY KEY (id_scheme, external_id)`
- Index on `entity_id`

## Existing dictionary.py patterns

- Uses `psycopg.Connection` passed to functions
- `dict_row` row factory for cursor
- `lower()` for case-insensitive canonical name lookup
- Falls back from canonical_name to alias search
- Uses `LIMIT 1` (our replacement must NOT do this)

## db.py patterns

- `get_connection(dsn)` returns `psycopg.Connection`
- Module-level functions and classes taking conn as first arg

## Test patterns (test_dictionary.py)

- Integration tests use real DB with `@pytest.mark.integration`
- Unit tests don't need DB
- `helpers.DSN` for connection string
- Fixtures with cleanup via DELETE
- Tests grouped in classes by feature
