# Research: harvest-gcmd-v2

## Current State

### scripts/harvest_gcmd.py

- Uses `urllib.request` for HTTP (manual retry with exponential backoff in `_fetch_url()`)
- Downloads from GitHub (instruments, platforms, sciencekeywords) and KMS API (providers, projects)
- Parses into entry dicts: canonical_name, entity_type, source, external_id, aliases, metadata
- Calls `bulk_load()` from `src/scix/dictionary.py` to write to `entity_dictionary` table
- `run_harvest()` accepts dsn, schemes, dry_run params

### src/scix/http_client.py — ResilientClient

- Drop-in replacement with retry, rate limiting, circuit breaker, disk caching
- `get(url, params=None, **kwargs)` → returns `requests.Response` or `CachedResponse`
- Both have `.json()` method for parsing
- Uses `requests` library under the hood
- Default user_agent: "scix-harvester/1.0", timeout: 60s

### src/scix/dictionary.py — bulk_load()

- Batch upserts into entity_dictionary with ON CONFLICT
- Accepts discipline param (stored as metadata.\_discipline)

### migrations/020_harvest_runs.sql

- `harvest_runs` table: id, source, started_at, finished_at, status, records_fetched, records_upserted, cursor_state, error_message, config, counts

### migrations/021_entity_graph.sql

- `entities`: canonical_name, entity_type, discipline, source, harvest_run_id, properties; UNIQUE(canonical_name, entity_type, source)
- `entity_identifiers`: entity_id FK, id_scheme, external_id, is_primary; PK(id_scheme, external_id)
- `entity_aliases`: entity_id FK, alias, alias_source; PK(entity_id, alias)

## Key Design Decisions

1. Replace urllib with ResilientClient.get() — response.json() works for both GitHub and KMS
2. Keep entity_dictionary writes via bulk_load() for backward compat
3. After bulk_load, also write to entities/entity_identifiers/entity_aliases
4. Create harvest_run at start (status='running'), update at end with counts
5. Store gcmd_scheme and gcmd_hierarchy in entities.properties
