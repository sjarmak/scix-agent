# Test Results: Harvester Modernization

## Test Run

```
python -m pytest tests/test_harvest_ascl.py tests/test_harvest_aas.py tests/test_harvest_vizier.py tests/test_harvest_pwc.py tests/test_harvest_physh.py tests/test_harvest_astromlab.py -v
```

## Results: 166 passed, 4 skipped

### test_harvest_ascl.py (22 tests)

- 14 parse tests: PASSED
- 2 download tests (ResilientClient mocked): PASSED
- 5 run_harvest tests (HarvestRunLog mocked): PASSED
- 1 large catalog test: PASSED

### test_harvest_aas.py (20 tests)

- 12 classify_header tests: PASSED
- 3 table parser tests: PASSED
- 18 parse_aas_facilities tests: PASSED
- 3 integration tests: SKIPPED (DB not available)

### test_harvest_vizier.py (17 tests)

- 7 parse_votable tests: PASSED
- 11 build_dictionary tests: PASSED
- 3 query_tap_vizier tests (ResilientClient mocked): PASSED
- 5 run_harvest tests (HarvestRunLog mocked): PASSED
- 2 large catalog tests: PASSED

### test_harvest_pwc.py (15 tests)

- 10 parse_methods tests: PASSED
- 2 download_methods tests (ResilientClient mocked): PASSED
- 3 run_pipeline tests (HarvestRunLog mocked): PASSED
- 1 integration test: SKIPPED (DB not available)

### test_harvest_physh.py (20 tests)

- 16 parse_physh tests: PASSED
- 4 download_physh tests (ResilientClient mocked): PASSED
- 4 run_harvest tests (HarvestRunLog mocked): PASSED

### test_harvest_astromlab.py (37 tests)

- 6 category mapping tests: PASSED
- 11 CSV parse tests: PASSED
- 9 JSON parse tests: PASSED
- 6 parse_concepts tests: PASSED
- 4 download_concepts tests (ResilientClient mocked): PASSED
- 6 run_pipeline tests (HarvestRunLog mocked): PASSED

## Skipped Tests (4)

All integration tests requiring a running database were skipped as expected.
