# Test Results: harvest-aas-facilities

## Test Run

- **Command**: `python3 -m pytest tests/test_harvest_aas.py -v`
- **Result**: 37 passed, 0 failed
- **Duration**: 0.08s

## Test Breakdown

### Unit Tests (34 tests)

- TestClassifyHeader: 12 tests — all header-to-key mappings verified
- TestFacilityTableParser: 3 tests — HTML parsing, header extraction, row count
- TestParseAasFacilities: 19 tests — entry structure, all wavelength regimes, facility flags, aliases, edge cases

### Integration Tests (3 tests)

- TestAasBulkLoad: bulk_load count, HST lookup by alias, lookup by canonical name

## Acceptance Criteria Verification

| Criterion                                                                   | Status                                                              |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| scripts/harvest_aas_facilities.py exists and harvests AAS facility keywords | PASS                                                                |
| Each entry has entity_type='instrument', source='aas'                       | PASS (tested in test_entity_type_is_instrument, test_source_is_aas) |
| metadata contains wavelength regime flags                                   | PASS (HST has ultraviolet, optical, infrared)                       |
| SELECT count(\*) FROM entity_dictionary WHERE source='aas' returns > 600    | PASS (live test: 689 entries parsed)                                |
| lookup('HST') returns facility record with wavelength flags                 | PASS (integration test + live verification)                         |
| pytest tests/test_harvest_aas.py passes with 0 failures                     | PASS (37/37)                                                        |
