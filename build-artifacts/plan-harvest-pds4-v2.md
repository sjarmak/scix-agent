# Plan: harvest-pds4-v2

## Changes to scripts/harvest_pds4.py

### 1. Replace urllib with ResilientClient

- Remove urllib imports, add `from scix.http_client import ResilientClient`
- Replace `fetch_pds4_page()` to use `ResilientClient.get()` instead of manual urllib+retry
- Remove `MAX_RETRIES` constant (handled by ResilientClient)
- Create client instance in `run_harvest()` and pass through

### 2. Write to entity graph tables after bulk_load

- After `bulk_load()` (backward compat), INSERT into:
  - `entities` — canonical_name, entity_type, discipline='planetary_science', source='pds4', harvest_run_id, properties=metadata
  - `entity_identifiers` — PDS URN as external_id, id_scheme='pds_urn', is_primary=true
  - `entity_aliases` — each alias with alias_source='pds4'

### 3. Create entity_relationships

- **part_of_mission**: Parse instrument URNs to extract parent spacecraft/investigation name. Match against entities table to find the mission entity_id. Insert (instrument_id, 'part_of_mission', mission_id, source='pds4').
- **observes_target**: PDS API doesn't encode target-mission links in URNs. Use PDS API `ref_lid_*` fields if available, otherwise create relationships based on URN structure where possible. For v2, we'll request additional reference fields from the API.

### 4. Log harvest_run

- INSERT into harvest_runs at start (status='running')
- UPDATE with finished_at, status='completed', records_fetched, records_upserted, counts on success
- UPDATE with status='failed', error_message on failure

### 5. Entity graph helper function

- `_write_entity_graph(conn, entries, harvest_run_id)` — handles entities, identifiers, aliases
- `_write_relationships(conn, harvest_run_id)` — queries entities table to build relationships from URN structure

## Changes to tests/test_harvest_pds4.py

### New test classes:

- `TestResilientClientUsage` — verify ResilientClient imported and used
- `TestEntityGraph` — verify entities, entity_identifiers, entity_aliases populated
- `TestEntityRelationships` — verify part_of_mission, observes_target predicates
- `TestHarvestRuns` — verify harvest_runs completion record
- Existing tests updated for new mock targets (ResilientClient instead of urllib)

## Backward Compatibility

- `bulk_load()` call preserved — entity_dictionary continues to be populated
- All existing public API (run_harvest, parse_pds4_products, etc.) signatures unchanged
- dry_run still skips DB writes
