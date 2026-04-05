# Test Results: harvest-ascl

## Run: 2026-04-04

```
22 passed in 0.06s
```

## Test Breakdown

### TestParseAsclEntries (14 tests) - ALL PASSED

- Parses all valid entries
- Entry has required keys
- entity_type is 'software'
- source is 'ascl'
- external_id is ASCL ID
- canonical_name from title
- bibcode in metadata
- credit in metadata
- aliases include lowercase
- No duplicate lowercase alias
- Skips entries without title
- Skips entries without ascl_id
- Entry without bibcode has no bibcode in metadata
- Empty input returns empty

### TestDownloadAsclCatalog (3 tests) - ALL PASSED

- Returns parsed JSON from mocked HTTP
- Retries on transient failure
- Raises after max retries

### TestRunHarvest (4 tests) - ALL PASSED

- Calls bulk_load with correct entries
- Closes DB connection on success
- Closes DB connection on error
- Astropy entry has ASCL ID and bibcode in metadata

### TestLargeCatalog (1 test) - ALL PASSED

- Parses catalog with 4000 entries (>3500 threshold)
