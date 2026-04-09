# Plan: enrich-wikidata-multi

## Step 1: Create scripts/enrich_wikidata_multi.py

### Structure

1. ENTITY_TYPE_CONFIGS dict mapping (source, entity_type) tuples to query config
2. `build_batch_sparql_query(names: list[str]) -> str` — VALUES clause with up to 50 names
3. `execute_sparql(query, endpoint)` — reuse retry pattern from instruments script
4. `parse_batch_results(sparql_json) -> dict[str, tuple[str, list[str]]]` — map name -> (qid, aliases)
5. `merge_aliases(existing, new)` — same as instruments script
6. `load_cache(path) -> dict | None` — read cached JSON from disk
7. `save_cache(path, data)` — write JSON to disk
8. `fetch_entries(conn, source, entity_type) -> list[dict]` — query entity_dictionary
9. `enrich_batch(conn, entries, batch_idx, ...) -> int` — enrich one batch of up to 50
10. `run_enrich(dsn, source, entity_type, ...) -> tuple[int, int]` — main pipeline
11. `main()` — argparse CLI

### CLI Flags

- `--dsn` — DB connection string
- `--source` — filter by source (gcmd, pds4)
- `--entity-type` — filter by entity_type
- `--batch-size` — max names per SPARQL query (default 50)
- `--delay` — seconds between batches (default 2.0)
- `--dry-run` — no DB writes
- `--no-cache` — skip cache read/write
- `--cache-dir` — cache directory (default data/wikidata_cache/)
- `-v/--verbose` — debug logging

### Batching

- Collect entries, chunk into groups of 50
- For each batch: check cache -> query SPARQL -> save cache -> parse -> update DB
- Sleep >= 2s between batches

### Caching

- Directory: data/wikidata_cache/
- Filename: {source}_{entity_type}_{batch_idx}.json
- On cache hit, skip SPARQL query

## Step 2: Create tests/test_enrich_wikidata_multi.py

### Test Classes

1. TestBuildBatchSparqlQuery — VALUES clause, batch size, escaping
2. TestParseBatchResults — name-to-QID mapping, aliases extraction
3. TestMergeAliases — reuse pattern
4. TestCache — write/read round-trip
5. TestBatchSizeEnforcement — max 50
6. TestSleepBetweenBatches — mock time.sleep, verify >= 2
7. TestDbUpdate — aliases merged, wikidata_qid set
8. TestCli — --help exits 0
