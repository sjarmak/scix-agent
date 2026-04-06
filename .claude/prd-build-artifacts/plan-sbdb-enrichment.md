# Plan: SBDB Enrichment Harvester

## Step 1: Create scripts/harvest_sbdb.py

### 1a: Module structure

- Shebang, docstring, imports (sys.path hack, argparse, logging, json, time)
- Import from scix: get_connection, HarvestRunLog, ResilientClient

### 1b: Constants

- SBDB_API_BASE = "https://ssd-api.jpl.nasa.gov/sbdb.api"
- SOURCE = "sbdb"
- ENRICHES_SOURCE = "ssodnet"

### 1c: Lazy ResilientClient

- rate_limit=1.0 (1 req/s for ssd.jpl.nasa.gov)
- cache_dir=".cache/sbdb", cache_ttl=86400

### 1d: fetch_sbdb_record(client, designation) -> dict | None

- GET ?des=<designation>&phys-par=true&discovery=true&ca-data=false
- Parse response: extract orbital_class, neo, pha, discovery_date, discovery_site
- Return enrichment dict or None on error/404

### 1e: parse_sbdb_response(data) -> dict

- Extract: object.orbit_class.name -> orbital_class
- Extract: object.neo (bool), object.pha (bool)
- Extract: discovery.date, discovery.site, discovery.name (discoverer)
- Return flat dict of enrichment fields

### 1f: get_last_cursor(conn) -> int | None

- Query harvest_runs for last completed sbdb run's cursor_state
- Return last_entity_id or None

### 1g: save_cursor(conn, run_id, entity_id)

- UPDATE harvest_runs SET cursor_state = '{"last_entity_id": N}' WHERE id = run_id

### 1h: fetch_ssodnet_entities(conn, after_id=None) -> list[tuple[int, str]]

- SELECT id, canonical_name FROM entities WHERE source='ssodnet' ORDER BY id
- If after_id: add WHERE id > after_id

### 1i: update_entity_properties(conn, entity_id, enrichment_props)

- Merge new properties into existing: UPDATE entities SET properties = properties || %s WHERE id = %s

### 1j: run_harvest(dsn, dry_run, resume) -> int

- Connect, get entities, optionally resume from cursor
- HarvestRunLog lifecycle: start -> loop -> complete/fail
- For each entity: fetch SBDB, parse, update properties, save cursor
- Return count of enriched entities

### 1k: main() with argparse

- --dsn, --dry-run, --no-resume, -v/--verbose, --limit

## Step 2: Create tests/test_harvest_sbdb.py

### Tests:

1. test_parse_sbdb_response — correct field extraction
2. test_parse_sbdb_response_missing_fields — graceful handling
3. test_fetch_sbdb_record — mocked client.get, verify URL construction
4. test_fetch_sbdb_record_error — returns None on exception
5. test_get_last_cursor — mock DB query returns cursor
6. test_get_last_cursor_no_prior_run — returns None
7. test_save_cursor — verify SQL update
8. test_fetch_ssodnet_entities — verify query with/without after_id
9. test_update_entity_properties — verify jsonb merge SQL
10. test_run_harvest_full — integration: mock client + mock conn, verify lifecycle
11. test_run_harvest_resume — verify cursor resumption
12. test_rate_limit_config — verify ResilientClient gets rate_limit=1.0
