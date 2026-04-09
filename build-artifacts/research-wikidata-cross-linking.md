# Research: Wikidata Cross-Linking

## Current State

`scripts/enrich_wikidata_multi.py` enriches `entity_dictionary` entries with Wikidata QIDs and aliases:

- Queries `entity_dictionary` via `fetch_entries()` for source/entity_type combos
- Runs batched SPARQL queries against Wikidata (up to 50 names per batch)
- Updates `entity_dictionary` via `upsert_entry()` from `scix.dictionary`
- Stores QID in `metadata.wikidata_qid`, merges aliases into `aliases[]`

## Target State

Add an `--entity-source entities` mode that:

1. Reads from `entities` table instead of `entity_dictionary`
2. Writes QIDs to `entity_identifiers` table with `id_scheme='wikidata'`
3. Writes aliases to `entity_aliases` table with `alias_source='wikidata'`
4. Uses `upsert_entity_identifier()` and `upsert_entity_alias()` from `harvest_utils`

## Key Files

- `scripts/enrich_wikidata_multi.py` - main script
- `src/scix/harvest_utils.py` - `upsert_entity_identifier()`, `upsert_entity_alias()`
- `src/scix/dictionary.py` - `upsert_entry()` (existing path)
- `migrations/021_entity_graph.sql` - entities, entity_identifiers, entity_aliases schema
- `tests/test_enrich_wikidata_multi.py` - existing tests

## Schema Details

- `entities`: (id, canonical_name, entity_type, discipline, source, harvest_run_id, properties)
  - Unique on (canonical_name, entity_type, source)
- `entity_identifiers`: (entity_id FK, id_scheme, external_id, is_primary)
  - PK on (id_scheme, external_id)
- `entity_aliases`: (entity_id FK, alias, alias_source)
  - PK on (entity_id, alias)

## Approach

- Add `entity_source` parameter to `run_enrich()` with values "dictionary" (default) or "entities"
- Add `fetch_entities_from_graph()` function to query `entities` table
- Add `apply_enrichments_entities()` function that calls `upsert_entity_identifier` + `upsert_entity_alias`
- Add CLI flag `--entity-source {dictionary,entities}` defaulting to "dictionary"
- Existing code path unchanged when entity_source="dictionary"
