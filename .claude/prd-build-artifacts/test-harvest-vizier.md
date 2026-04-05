# Test Results: harvest-vizier

## Summary

- **29 tests passed, 0 failures**
- Runtime: 0.22s

## Test Coverage

### TestParseVotableCatalogs (7 tests)

- Parses all rows from sample VOTable XML
- Extracts table_name, description, utype fields correctly
- Handles empty utype and empty TABLEDATA
- Handles missing descriptions (returns empty string)

### TestBuildDictionaryEntries (12 tests)

- Correct count, entity_type='dataset', source='vizier'
- external_id maps to table_name
- canonical_name from description, fallback to table_name
- utype in metadata when present, absent when empty
- Empty aliases, required keys present
- Skips entries without table_name
- Empty input returns empty list

### TestQueryTapVizier (4 tests)

- Returns raw bytes from mocked response
- Retries on failure with backoff
- Raises after max retries
- Sends POST request with data

### TestRunHarvest (4 tests)

- Calls bulk_load with correct entries (entity_type, source)
- Closes connection on success and on error
- Verifies 2MASS entry structure end-to-end

### TestLargeCatalog (2 tests)

- Parses 30,000 VOTable rows (> 25,000 threshold)
- Builds 30,000 dictionary entries from large catalog

## Acceptance Criteria Verification

| Criterion                                                        | Status       |
| ---------------------------------------------------------------- | ------------ |
| scripts/harvest_vizier.py exists and queries TAPVizieR via TAP   | PASS         |
| Parses entries with canonical_name=title, external_id=catalog ID | PASS         |
| Large catalog test verifies > 25,000 entries can be processed    | PASS         |
| pytest tests/test_harvest_vizier.py passes with 0 failures       | PASS (29/29) |
