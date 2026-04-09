# Plan: SPASE Harvester Update

## Step 1: Update imports in harvest_spase.py

- Remove: `import time`, `import urllib.error`, `import urllib.request`
- Add: `from scix.harvest_utils import HarvestRunLog, upsert_entity, upsert_entity_alias, upsert_entity_identifier`
- Add: `from scix.http_client import ResilientClient`
- Keep `time` for elapsed timing in run_harvest

## Step 2: Update SPASE_VERSION

- Change from "2.7.0" to "2.7.1"

## Step 3: Replace download_tab_file with ResilientClient

- Add module-level `_client` and `_get_client()` lazy init (same pattern as harvest_gcmd.py)
- Replace `download_tab_file()` with `_download_tab_file()` that uses `_get_client().get(url).text`

## Step 4: Add \_write_entity_graph()

- For each entry: upsert_entity with properties from metadata
- upsert_entity_identifier with id_scheme='spase_resource_id', external_id from entry
- upsert_entity_alias for each alias
- conn.commit() at end

## Step 5: Update run_harvest() with HarvestRunLog

- Create HarvestRunLog(conn, "spase"), call start()
- Keep bulk_load() for backward compat
- Call \_write_entity_graph()
- Call run_log.complete() or run_log.fail()

## Step 6: Update tests

- Change all `@patch("harvest_spase.download_tab_file")` to mock `_get_client`
- Add test for \_write_entity_graph
- Add test for HarvestRunLog lifecycle in run_harvest
- Add test for entity_identifiers with id_scheme='spase_resource_id'
