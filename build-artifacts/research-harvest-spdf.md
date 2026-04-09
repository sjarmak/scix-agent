# Research: harvest-spdf

## Existing Harvester Patterns

- All harvesters in `scripts/harvest_*.py` use `sys.path.insert` to import from `src/scix/`
- They use `scix.db.get_connection(dsn)` and `scix.dictionary.bulk_load()` for entity_dictionary
- None of the existing harvesters use `ResilientClient` yet or the new entity graph tables (migration 021)
- Pattern: argparse CLI with --dsn, --dry-run, --verbose flags

## ResilientClient (src/scix/http_client.py)

- `ResilientClient.get(url, params=None, **kwargs)` — returns `requests.Response | CachedResponse`
- Features: retry with backoff, rate limiting, circuit breaker, disk caching
- Both Response and CachedResponse have `.json()` method
- Accepts `headers` kwarg (merged with User-Agent default)

## Entity Graph Schema (migration 021)

- `entities`: id, canonical_name, entity_type, discipline, source, harvest_run_id, properties, created_at, updated_at; UNIQUE(canonical_name, entity_type, source)
- `entity_identifiers`: entity_id FK, id_scheme, external_id, is_primary; PK(id_scheme, external_id)
- `entity_aliases`: entity_id FK, alias, alias_source; PK(entity_id, alias)
- `entity_relationships`: id, subject_entity_id FK, predicate, object_entity_id FK, source, harvest_run_id, confidence; UNIQUE(subject, predicate, object)
- `datasets`: id, name, discipline, source, canonical_id, description, temporal_start, temporal_end, properties, harvest_run_id, created_at; UNIQUE(source, canonical_id)
- `dataset_entities`: dataset_id FK, entity_id FK, relationship; PK(dataset_id, entity_id, relationship)

## harvest_runs (migration 020)

- id, source, started_at, finished_at, status, records_fetched, records_upserted, cursor_state, error_message, config, counts (JSONB)

## entity_dictionary backward compat

- `bulk_load(conn, entries, discipline=)` — upserts into entity_dictionary table
- entity_type must be in ALLOWED_ENTITY_TYPES: instrument, dataset, software, method, mission, observable, target

## CDAWeb REST API

- Base: https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/
- Endpoints: /datasets, /observatories, /instruments
- Returns XML by default — need `Accept: application/json` header
- Dataset fields: Id, Label, TimeInterval (Start/End), ObservatoryGroup, InstrumentType, Notes, etc.
