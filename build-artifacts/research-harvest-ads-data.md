# Research: harvest-ads-data

## Findings

### Data column

- `papers.data` is a `TEXT[]` array column added in migration 012
- Field mapping in `field_mapping.py` lists `data` in `DIRECT_ARRAY_FIELDS`
- ADS `data` field contains archive-level dataset source labels (e.g. "CDS:1", "MAST:5")

### Entity dictionary

- Table `entity_dictionary` created in migration 013
- Unique constraint on `(canonical_name, entity_type, source)`
- Columns: id, canonical_name, entity_type, source, external_id, aliases, metadata (JSONB)
- `bulk_load()` in `src/scix/dictionary.py` does upsert via ON CONFLICT DO UPDATE
- Each entry dict needs: canonical_name, entity_type, source; optional: external_id, aliases, metadata

### DB connection

- `src/scix/db.py` provides `get_connection(dsn)` using `SCIX_DSN` env var (default "dbname=scix")

### Test patterns

- Tests use `from helpers import DSN` and `psycopg.connect(DSN)`
- Integration tests use `@pytest.mark.integration`
- Test fixtures clean up with DELETE WHERE source = 'test-...'
- Skip if DB unavailable or table missing

### Query approach

- `SELECT unnest(data) as source, count(*) FROM papers GROUP BY 1 ORDER BY 2 DESC`
- This unnests the TEXT[] array and counts paper occurrences per data source label
