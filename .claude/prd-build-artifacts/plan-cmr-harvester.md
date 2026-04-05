# Plan: CMR Collection Harvester

## Step 1: Create scripts/harvest_cmr.py

### 1a: Module structure

- Shebang, docstring, imports (sys.path.insert for src/)
- Import ResilientClient, HarvestRunLog, upsert_entity_identifier
- Constants: CMR_BASE_URL, SOURCE='cmr', DISCIPLINE='earth_science'
- UMM_JSON_ACCEPT header constant

### 1b: Client setup

- `_make_client()` -> ResilientClient configured for CMR

### 1c: Pagination with Search-After

- `fetch_collections(client, page_size=2000)` -> list of items
- First request: no Search-After header
- Each response: read CMR-Search-After from response headers
- Send it back as Search-After in next request
- Stop when all items fetched (len(items) >= hits or no more items)
- Track page count for logging

### 1d: Parse UMM-JSON items

- `parse_collection(item)` -> dict with:
  - concept_id from meta.concept-id
  - name from umm.EntryTitle or umm.ShortName
  - short_name from umm.ShortName
  - abstract from umm.Abstract
  - platforms: list of platform ShortNames
  - instruments: list of instrument ShortNames (nested under Platforms)
  - science_keywords: list of Category/Topic/Term dicts

### 1e: DB writes

- Reuse `_upsert_dataset()` pattern from harvest_spdf.py
- Reuse `_upsert_dataset_entity()` pattern from harvest_spdf.py
- For each collection:
  1. Upsert into datasets (source='cmr', canonical_id=concept-id)
  2. Store science_keywords in properties JSONB
  3. Look up GCMD entities by name, link via dataset_entities
- GCMD cross-reference: query entities table for matching instrument/platform names with source='gcmd', link via dataset_entities

### 1f: Harvest orchestration

- `store_collections(conn, collections, run_id)` -> counts dict
- `run_harvest(dsn, dry_run)` -> full pipeline with HarvestRunLog
- `main()` with argparse (--dsn, --dry-run, -v)

## Step 2: Create tests/test_harvest_cmr.py

- Mock ResilientClient.get() for API calls
- Test Search-After pagination (multi-page scenario)
- Test Accept header is UMM-JSON format
- Test parse_collection extracts instruments/platforms/keywords
- Test datasets stored with source='cmr'
- Test GCMD cross-referencing logic
- Test concept-id deduplication (ON CONFLICT)
- Test HarvestRunLog lifecycle (start/complete/fail)
