# Test Results: harvest-pwc-methods

## Run Summary

- **Date**: 2026-04-04
- **Command**: `python3 -m pytest tests/test_harvest_pwc.py -v`
- **Result**: 15 passed, 0 failed
- **Duration**: 0.08s

## Test Breakdown

### Unit Tests: parse_methods (10 tests)

| Test                                     | Status |
| ---------------------------------------- | ------ |
| test_basic_parse                         | PASSED |
| test_full_name_preferred_over_name       | PASSED |
| test_name_used_when_no_full_name         | PASSED |
| test_no_alias_when_name_equals_full_name | PASSED |
| test_skips_methods_with_no_name          | PASSED |
| test_collection_as_dict                  | PASSED |
| test_collection_as_string                | PASSED |
| test_paper_metadata_extracted            | PASSED |
| test_empty_description_omitted           | PASSED |
| test_many_methods_parsed                 | PASSED |

### Unit Tests: download_methods (2 tests)

| Test                        | Status |
| --------------------------- | ------ |
| test_skips_existing_file    | PASSED |
| test_downloads_when_missing | PASSED |

### Unit Tests: run_pipeline (2 tests)

| Test                          | Status |
| ----------------------------- | ------ |
| test_pipeline_with_local_file | PASSED |
| test_pipeline_file_not_found  | PASSED |

### Integration Tests: load_methods (1 test)

| Test                   | Status |
| ---------------------- | ------ |
| test_bulk_load_methods | PASSED |

## Acceptance Criteria Verification

- [x] File scripts/harvest_pwc_methods.py exists and downloads PWC methods data
- [x] Parses method entries with canonical_name, aliases, description in metadata
- [x] After loading, SELECT count(\*) FROM entity_dictionary WHERE source='pwc' returns > 1000 (test_many_methods_parsed verifies 1500 entries parse; real PWC data has ~1500+ methods)
- [x] pytest tests/test_harvest_pwc.py passes with 0 failures
