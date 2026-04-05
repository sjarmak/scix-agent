# Plan: harvest-gcmd-v2

## Step 1: Replace urllib with ResilientClient in harvest_gcmd.py

- Remove `urllib.request`, `urllib.error`, `time` (retry-related) imports
- Add `from scix.http_client import ResilientClient`
- Replace `_fetch_url()` with a module-level or function-scoped ResilientClient instance
- Update `download_github_scheme()`: use `client.get(url).json()`
- Update `download_kms_scheme()`: use `client.get(url).json()` for each page

## Step 2: Add harvest_runs tracking to run_harvest()

- At start: INSERT into harvest_runs (source='gcmd', status='running', config=schemes)
- On success: UPDATE status='completed', records_fetched, records_upserted, finished_at
- On error: UPDATE status='failed', error_message, finished_at

## Step 3: Add entity graph writes after bulk_load()

- After bulk_load (backward compat), write to entities table:
  - canonical_name, entity_type, discipline='earth_science', source='gcmd'
  - properties = {gcmd_scheme, gcmd_hierarchy} from entry metadata
  - harvest_run_id from step 2
- For each entity: INSERT into entity_identifiers (id_scheme='gcmd_uuid', external_id=uuid, is_primary=true)
- For each entity with aliases: INSERT into entity_aliases

## Step 4: Update tests

- Mock ResilientClient instead of urllib
- Add tests for entities table writes
- Add tests for entity_identifiers with id_scheme='gcmd_uuid'
- Add tests for harvest_runs record
- Keep existing backward compat tests for entity_dictionary
