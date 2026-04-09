# Test Results: CMR Collection Harvester

## Run: 2026-04-05

```
25 passed in 0.11s
```

## Test Coverage

| Test Class                 | Tests | Status |
| -------------------------- | ----- | ------ |
| TestResilientClientUsed    | 1     | PASS   |
| TestSearchAfterPagination  | 3     | PASS   |
| TestUmmJsonAcceptHeader    | 1     | PASS   |
| TestParseCollection        | 7     | PASS   |
| TestDatasetStorage         | 2     | PASS   |
| TestGcmdCrossReference     | 3     | PASS   |
| TestConceptIdDeduplication | 1     | PASS   |
| TestHarvestRunLogLifecycle | 3     | PASS   |
| TestCLI                    | 2     | PASS   |
| TestImportable             | 1     | PASS   |

## Acceptance Criteria Verification

- [x] scripts/harvest_cmr.py exists and is importable
- [x] Script imports and uses ResilientClient from scix.http_client
- [x] Script imports and uses HarvestRunLog from scix.harvest_utils
- [x] Script uses Search-After header pagination (not offset-based)
- [x] Script requests UMM-JSON format via Accept header
- [x] Script stores datasets with source='cmr'
- [x] Script extracts instruments, platforms, science_keywords from UMM-JSON
- [x] Script cross-references GCMD entities via entity_identifiers with id_scheme='gcmd_uuid'
- [x] Script deduplicates on concept-id
- [x] Tests pass: pytest tests/test_harvest_cmr.py
