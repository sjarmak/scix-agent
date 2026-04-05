# Research: harvest-ascl

## Key Findings

### dictionary.py API

- `bulk_load(conn, entries)` accepts list of dicts with keys: canonical_name, entity_type, source, plus optional external_id, aliases, metadata
- `lookup(conn, name, entity_type=None)` does case-insensitive canonical_name match then alias fallback
- Uses psycopg (v3), ON CONFLICT upsert

### db.py

- `get_connection(dsn=None)` reads SCIX_DSN env var, returns psycopg.Connection

### CLI Pattern (load_uat.py)

- sys.path.insert for src/scix
- argparse with --dsn, -v/--verbose
- logging.basicConfig

### Download Pattern (uat.py)

- urllib.request with User-Agent header
- Retry loop with exponential backoff
- timeout=60

### Test Pattern (test_dictionary.py)

- Integration tests use db_conn fixture with DSN from helpers
- Tests import from helpers (DSN constant)
- @pytest.mark.integration for DB tests

### ASCL JSON API

- URL: https://ascl.net/code/json
- Returns JSON array of objects
- Fields: title, ascl_id, bibcode, credit
- ~3800+ entries expected
