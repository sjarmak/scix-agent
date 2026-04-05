# Research: CMR Collection Harvester

## Reference Pattern Analysis

### harvest_gcmd.py

- Uses `sys.path.insert(0, ...)` for src/ imports
- Lazy `_get_client()` for ResilientClient singleton
- `HarvestRunLog` lifecycle: start -> complete/fail
- argparse CLI with --dsn, --dry-run, -v
- Entity graph writes: upsert_entity, upsert_entity_identifier, upsert_entity_alias

### harvest_spdf.py

- `_upsert_dataset()` helper: INSERT ON CONFLICT(source, canonical_id) DO UPDATE RETURNING id
- `_upsert_dataset_entity()` helper: INSERT ON CONFLICT DO NOTHING for bridge table
- `store_harvest()` orchestrates all DB writes within HarvestRunLog lifecycle
- `run_harvest()` top-level: client creation, fetch, parse, DB write

### harvest_utils.py

- HarvestRunLog(conn, source).start(config) -> run_id
- .complete(records_fetched, records_upserted, counts)
- .fail(error_message)
- upsert_entity, upsert_entity_identifier helpers available

### http_client.py

- ResilientClient.get(url, params, \*\*kwargs) -> Response|CachedResponse
- kwargs includes headers dict
- Response has .json(), .headers, .text

### datasets table (migration 021)

- Columns: id, name, discipline, source, canonical_id, description, temporal_start, temporal_end, properties (JSONB), harvest_run_id
- UNIQUE(source, canonical_id)

## CMR API Details

- Base URL: https://cmr.earthdata.nasa.gov/search/collections
- Accept header: application/vnd.nasa.cmr.umm_results+json
- Pagination: Search-After header (NOT offset). Response includes CMR-Search-After header.
- Query param: page_size=2000
- Response: {"hits": N, "items": [{meta: {concept-id}, umm: {ShortName, EntryTitle, Abstract, Platforms, ScienceKeywords}}]}
- Platforms[].ShortName, Platforms[].Instruments[].ShortName (nested)
- ScienceKeywords[].Category/Topic/Term

## GCMD Cross-referencing

- GCMD entities stored in entity_identifiers with id_scheme='gcmd_uuid'
- CMR collections reference GCMD instruments/platforms by name
- Look up entities by canonical_name matching, then link via dataset_entities
