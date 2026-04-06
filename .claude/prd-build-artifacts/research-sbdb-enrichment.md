# Research: SBDB Enrichment Harvester

## Reference Files Reviewed

### harvest_gcmd.py

- Pattern: module-level lazy ResilientClient, HarvestRunLog lifecycle (start/complete/fail), argparse CLI
- Uses `upsert_entity`, `upsert_entity_identifier`, `upsert_entity_alias` from harvest_utils
- HarvestRunLog: `start(config=dict)` -> `complete(records_fetched, records_upserted, counts)` or `fail(error_message)`

### harvest_ssodnet.py

- Populates entities with `source='ssodnet'`, `entity_type='target'`, `discipline='planetary_science'`
- Properties include: diameter, albedo, taxonomy, sso_number
- Identifiers: `id_scheme='ssodnet'` (primary), `id_scheme='sbdb_spkid'`

### harvest_utils.py

- `HarvestRunLog(conn, source)` — no cursor_state update method built in
- `upsert_entity()` — ON CONFLICT updates properties, harvest_run_id, updated_at
- Need manual SQL to update `cursor_state` in harvest_runs mid-run

### http_client.py

- `ResilientClient(rate_limit=1.0)` — float, requests per second per host
- No `rate_limits` dict parameter exists — use `rate_limit=1.0` instead

### migrations/020_harvest_runs.sql

- `cursor_state JSONB` column EXISTS in the table
- Can store `{"last_entity_id": N}` directly

### migrations/021_entity_graph.sql

- entities: `(id, canonical_name, entity_type, discipline, source, harvest_run_id, properties JSONB)`
- UNIQUE constraint: `(canonical_name, entity_type, source)`
- GIN index on properties

## JPL SBDB API

- Base URL: `https://ssd-api.jpl.nasa.gov/sbdb.api`
- Query: `?des=<name>&phys-par=true&discovery=true&ca-data=false`
- Response fields: `object.orbit_class.{name, code}`, `discovery.{name, site, date}`, `phys_par[]`, `object.neo`, `object.pha`
- Rate limit: 1 req/s per IP (strict)

## Cursor Resumption Strategy

- SBDB enriches existing ssodnet entities — query `SELECT id, canonical_name FROM entities WHERE source='ssodnet' ORDER BY id`
- Store `{"last_entity_id": N}` in `harvest_runs.cursor_state`
- On resume: find last completed SBDB run, read its cursor_state, filter `WHERE id > last_entity_id`
- Update cursor_state after each successful entity update via direct SQL

## Key Design Decision

- This is an ENRICHMENT harvester: it does NOT create new entities, it UPDATES existing ones
- Merges new fields into existing `properties` JSONB (preserving existing fields)
- Uses `jsonb_concat` / Python dict merge to avoid overwriting ssodnet-provided properties
