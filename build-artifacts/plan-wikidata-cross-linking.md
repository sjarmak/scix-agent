# Plan: Wikidata Cross-Linking

## Step 1: Add `fetch_entities_from_graph()` function

New function in `enrich_wikidata_multi.py` that queries the `entities` table:

```sql
SELECT id, canonical_name, entity_type, source
FROM entities
WHERE source = %(source)s AND entity_type = %(entity_type)s
ORDER BY canonical_name
```

Returns list of dicts with same shape as `fetch_entries()` but with `id` as entity graph id.

## Step 2: Add `apply_enrichments_entities()` function

New function that takes entities-table rows + SPARQL matches and:

- For each match, calls `upsert_entity_identifier(conn, entity_id=id, id_scheme='wikidata', external_id=qid, is_primary=False)`
- For each alias, calls `upsert_entity_alias(conn, entity_id=id, alias=alias, alias_source='wikidata')`
- Supports dry_run mode
- Returns count of enriched entities

## Step 3: Add `entity_source` parameter to `run_enrich()`

- New param `entity_source: str = "dictionary"` with values "dictionary" or "entities"
- When "dictionary": use existing `fetch_entries()` + `apply_enrichments()` (no change)
- When "entities": use `fetch_entities_from_graph()` + `apply_enrichments_entities()`
- Import `upsert_entity_identifier` and `upsert_entity_alias` from `harvest_utils`

## Step 4: Add CLI flag `--entity-source`

- `--entity-source {dictionary,entities}` defaulting to "dictionary"
- Pass through to `run_enrich()`

## Step 5: Update tests

Add to `tests/test_enrich_wikidata_multi.py`:

- `TestFetchEntitiesFromGraph` - mock cursor, verify SQL query
- `TestApplyEnrichmentsEntities` - mock upsert_entity_identifier/upsert_entity_alias, verify calls
- `TestApplyEnrichmentsEntitiesDryRun` - verify no DB writes in dry_run
- `TestRunEnrichEntitiesMode` - mock pipeline with entity_source="entities"
- `TestCliEntitySource` - verify --entity-source appears in --help
- Verify existing tests still pass (no regression)
