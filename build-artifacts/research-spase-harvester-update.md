# Research: SPASE Harvester Update

## Current State

- `harvest_spase.py` uses `urllib.request` with manual retry logic in `download_tab_file()`
- `SPASE_VERSION = "2.7.0"`
- Writes only to `entity_dictionary` via `bulk_load()`
- No HarvestRunLog lifecycle
- No entity graph writes (entities, entity_identifiers, entity_aliases)

## Reference Pattern (harvest_gcmd.py)

- Lazy `_client` / `_get_client()` pattern for ResilientClient
- `_write_entity_graph()` calls upsert_entity, upsert_entity_identifier, upsert_entity_alias
- `run_harvest()` creates HarvestRunLog, calls start/complete/fail lifecycle
- Still writes to entity_dictionary via bulk_load() for backward compat

## Changes Required

1. Remove `urllib.request`, `urllib.error`, `time` (only used in download_tab_file retry), add ResilientClient lazy init
2. Update SPASE_VERSION to "2.7.1"
3. Replace `download_tab_file()` with `_download_tab_file()` using ResilientClient `.get().text`
4. Add `_write_entity_graph()` with upsert_entity + upsert_entity_identifier (id_scheme='spase_resource_id') + upsert_entity_alias
5. Update `run_harvest()` with HarvestRunLog lifecycle
6. Keep bulk_load() for backward compat
7. Update tests to mock ResilientClient instead of urllib
