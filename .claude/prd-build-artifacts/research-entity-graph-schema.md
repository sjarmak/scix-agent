# Research: entity-graph-schema

## Existing Schema

### entity_dictionary (013)

- Columns: id SERIAL PK, canonical_name TEXT, entity_type TEXT, source TEXT, external_id TEXT, aliases TEXT[], metadata JSONB
- UNIQUE(canonical_name, entity_type, source)
- Indexes: entity_type, GIN on aliases

### Extensions (014)

- Added discipline TEXT column
- Added indexes: discipline btree, lower(canonical_name) functional

### Provenance pattern (017)

- ALTER TABLE with ADD COLUMN IF NOT EXISTS
- CHECK constraints for enum-like columns
- Indexes on filter columns

### harvest_runs (020)

- id SERIAL PK, source TEXT, started_at, finished_at, status, records_fetched, records_upserted, cursor_state JSONB, error_message TEXT, config JSONB, counts JSONB

## dictionary.py Column Expectations

The module queries entity_dictionary with these columns:

- id, canonical_name, entity_type, source, external_id, aliases, metadata
- Uses lower(canonical_name) for lookups
- Uses ANY(aliases) for alias search

The compat view must expose these exact column names for backward compatibility.

## Migration Patterns

- All migrations wrapped in BEGIN/COMMIT
- Use IF NOT EXISTS for CREATE TABLE/INDEX
- Comments at top describing purpose
- Consistent naming: idx*{table}*{column}
