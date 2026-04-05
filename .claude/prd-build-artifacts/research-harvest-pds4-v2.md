# Research: harvest-pds4-v2

## Current State

### scripts/harvest_pds4.py

- Uses `urllib.request` for HTTP (manual retry loop with `MAX_RETRIES=3`)
- Downloads PDS4 context products (investigation, instrument, target) via cursor-based pagination
- Parses products into entity_dictionary format via `parse_pds4_products()`
- Loads via `scix.dictionary.bulk_load()` with `discipline='planetary_science'`
- No harvest_runs logging
- No entity graph (entities, entity_identifiers, entity_relationships) integration

### Key Data Flow

1. `download_pds4_context()` → raw API products by type
2. `parse_pds4_products()` → list of dicts with canonical_name, entity_type, source, external_id (PDS URN), aliases, metadata
3. `run_harvest()` → calls `bulk_load(conn, all_entries, discipline="planetary_science")`

### src/scix/http_client.py — ResilientClient

- `get(url, params, **kwargs)` → Response or CachedResponse
- Both have `.json()` method
- Built-in retry, rate limiting, circuit breaker, disk caching
- Replaces urllib retry logic cleanly

### Migration 020 — harvest_runs

- Fields: id, source, started_at, finished_at, status, records_fetched, records_upserted, cursor_state, error_message, config, counts

### Migration 021 — entity_graph

- `entities`: canonical_name, entity_type, discipline, source, harvest_run_id, properties; UNIQUE(canonical_name, entity_type, source)
- `entity_identifiers`: entity_id, id_scheme, external_id, is_primary; PK(id_scheme, external_id)
- `entity_aliases`: entity_id, alias, alias_source; PK(entity_id, alias)
- `entity_relationships`: subject_entity_id, predicate, object_entity_id, source, harvest_run_id, confidence; UNIQUE(subject, predicate, object)

### Relationship Extraction from PDS URNs

- Instrument URNs encode parent spacecraft: `urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.cirs`
  - The `spacecraft.cassini-huygens` segment maps to investigation `mission.cassini-huygens`
  - This gives us: instrument `part_of_mission` investigation
- Target-to-mission relationships are NOT encoded in URNs — would need PDS reference API
  - For now: can use instrument URN parent to derive `observes_target` if instrument metadata links to targets
  - Simpler: parse PDS API's internal references if available, or skip observes_target for v2

### Backward Compatibility

- `entity_dictionary` table still used by `scix.dictionary.bulk_load()`
- Must continue calling `bulk_load()` to maintain entity_dictionary
- New code writes to entities/entity_identifiers/entity_aliases/entity_relationships in addition
