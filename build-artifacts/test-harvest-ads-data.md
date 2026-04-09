# Test Results: harvest-ads-data

## Run

```
pytest tests/test_harvest_ads_data.py -v
12 passed in 121.41s
```

## Results

| Test                                                   | Status |
| ------------------------------------------------------ | ------ |
| TestBuildEntries::test_empty_input                     | PASSED |
| TestBuildEntries::test_single_entry                    | PASSED |
| TestBuildEntries::test_multiple_entries_preserve_order | PASSED |
| TestBuildEntries::test_metadata_contains_paper_count   | PASSED |
| TestFetchDataSources::test_returns_list                | PASSED |
| TestFetchDataSources::test_sources_have_expected_keys  | PASSED |
| TestFetchDataSources::test_cds_appears_with_count      | PASSED |
| TestFetchDataSources::test_min_count_filter            | PASSED |
| TestFetchDataSources::test_limit                       | PASSED |
| TestFetchDataSources::test_ordered_by_count_desc       | PASSED |
| TestMainCLI::test_dry_run                              | PASSED |
| TestMainCLI::test_load_into_dictionary                 | PASSED |

## Coverage

- 4 unit tests (build_entries logic)
- 6 integration tests (fetch_data_sources with seeded DB)
- 2 CLI integration tests (dry-run + full load with dictionary verification)
