# Plan: harvest-ads-data

## Steps

1. Create `scripts/harvest_ads_data_field.py`:
   - Parse CLI args (--dsn, --min-count, --dry-run, --limit)
   - Connect to DB via `scix.db.get_connection()`
   - Run: `SELECT unnest(data) AS source, count(*) FROM papers GROUP BY 1 ORDER BY 2 DESC`
   - Optionally filter by min_count threshold
   - Build entity dicts: canonical_name=source_label, entity_type='dataset', source='ads_data', metadata={'paper_count': N}
   - Call `dictionary.bulk_load()` to upsert into entity_dictionary
   - Print summary stats

2. Create `tests/test_harvest_ads_data.py`:
   - Unit test: verify query construction logic (function extracts data sources from papers)
   - Integration test: insert test papers with data arrays, run harvest, verify entity_dictionary entries
   - Test dry-run mode (no DB writes)
   - Clean up test data after each test

3. Run tests, fix failures

4. Commit
