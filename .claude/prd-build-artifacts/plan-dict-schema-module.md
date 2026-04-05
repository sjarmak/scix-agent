# Plan: dict-schema-module

## Step 1: Create migration 013_entity_dictionary.sql

- BEGIN/COMMIT wrapper
- entity_dictionary table with: id SERIAL PK, canonical_name TEXT NOT NULL, entity_type TEXT NOT NULL, source TEXT NOT NULL, external_id TEXT, aliases TEXT[] DEFAULT '{}', metadata JSONB DEFAULT '{}'
- UNIQUE(canonical_name, entity_type, source)
- GIN index on aliases
- Index on entity_type

## Step 2: Create src/scix/dictionary.py

- `upsert_entry(conn, canonical_name, entity_type, source, external_id, aliases, metadata)` -> dict
- `lookup(conn, name, entity_type=None)` -> dict or None (search canonical_name and aliases)
- `bulk_load(conn, entries: list[dict])` -> int (batch INSERT ... ON CONFLICT DO UPDATE)
- `get_stats(conn)` -> dict with counts by entity_type

## Step 3: Create tests/test_dictionary.py

- Unit tests for argument validation / dict structure
- Integration tests for DB operations (upsert, lookup, bulk_load, get_stats)
- Test lookup returns correct keys: canonical_name, entity_type, source, external_id, aliases, metadata

## Step 4: Run tests, fix failures
