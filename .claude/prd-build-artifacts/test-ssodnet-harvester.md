# Test Results: SsODNet Harvester

## Run: 28 passed in 0.11s

```
tests/test_harvest_ssodnet.py::test_module_importable PASSED
tests/test_harvest_ssodnet.py::test_imports_resilient_client PASSED
tests/test_harvest_ssodnet.py::test_imports_harvest_run_log PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_full PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_properties PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_identifiers_ssodnet PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_identifiers_spkid PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_aliases PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_minimal PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_empty_name PASSED
tests/test_harvest_ssodnet.py::test_parse_sso_record_missing_name PASSED
tests/test_harvest_ssodnet.py::test_parse_ssocard PASSED
tests/test_harvest_ssodnet.py::test_parse_ssocard_empty PASSED
tests/test_harvest_ssodnet.py::test_cli_mode_bulk PASSED
tests/test_harvest_ssodnet.py::test_cli_mode_seed PASSED
tests/test_harvest_ssodnet.py::test_cli_default_mode PASSED
tests/test_harvest_ssodnet.py::test_cli_dsn_flag PASSED
tests/test_harvest_ssodnet.py::test_run_harvest_dispatches_bulk PASSED
tests/test_harvest_ssodnet.py::test_run_harvest_dispatches_seed PASSED
tests/test_harvest_ssodnet.py::test_run_harvest_invalid_mode PASSED
tests/test_harvest_ssodnet.py::test_write_staging_entities_calls_copy PASSED
tests/test_harvest_ssodnet.py::test_promote_staging_calls_function PASSED
tests/test_harvest_ssodnet.py::test_seed_harvest_uses_upsert_helpers PASSED
tests/test_harvest_ssodnet.py::test_harvest_run_log_lifecycle PASSED
tests/test_harvest_ssodnet.py::test_harvest_run_log_fail_on_error PASSED
tests/test_harvest_ssodnet.py::test_seed_dry_run_skips_db PASSED
tests/test_harvest_ssodnet.py::test_download_parquet_computes_sha256 PASSED
tests/test_harvest_ssodnet.py::test_write_staging_deduplicates_entities PASSED
```

## Coverage of Acceptance Criteria

| Criterion               | Test(s)                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| File importable         | test_module_importable                                                                           |
| Uses ResilientClient    | test_imports_resilient_client                                                                    |
| Uses HarvestRunLog      | test_imports_harvest_run_log, test_harvest_run_log_lifecycle, test_harvest_run_log_fail_on_error |
| --mode bulk/seed CLI    | test_cli_mode_bulk, test_cli_mode_seed, test_cli_default_mode                                    |
| Staging schema for bulk | test_write_staging_entities_calls_copy, test_promote_staging_calls_function                      |
| id_scheme='ssodnet'     | test_parse_sso_record_identifiers_ssodnet, test_seed_harvest_uses_upsert_helpers                 |
| id_scheme='sbdb_spkid'  | test_parse_sso_record_identifiers_spkid, test_seed_harvest_uses_upsert_helpers                   |
| Entity aliases          | test_parse_sso_record_aliases, test_seed_harvest_uses_upsert_helpers                             |
| Properties JSONB        | test_parse_sso_record_properties, test_parse_ssocard                                             |
