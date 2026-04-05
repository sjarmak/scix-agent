# Plan: harvest-pds4

## Implementation steps

1. Create `scripts/harvest_pds4.py` following established harvester pattern
2. Constants: API base URL, fields to request, entity type mapping
3. `fetch_pds4_products(context_type, limit, start)` — paginated API fetch with retry
4. `download_pds4_context(product_types)` — download all products for given types, handle pagination
5. `extract_aliases(title, properties)` — extract abbreviations from title parentheses + alternate_title fields
6. `parse_pds4_products(raw_products, entity_type)` — transform API response into dictionary records
7. `run_harvest(dsn, product_types, dry_run)` — orchestrate full pipeline
8. `main()` — CLI with --dsn, --verbose, --dry-run, --product-type flags
9. Create `tests/test_harvest_pds4.py` with mock HTTP responses
10. Test parsing for each product type, URN extraction, alias extraction, pagination

## Entity type mapping

| PDS context type | URN segment   | entity_type |
| ---------------- | ------------- | ----------- |
| Investigation    | investigation | mission     |
| Instrument       | instrument    | instrument  |
| Target           | target        | target      |

## CLI flags

- `--dsn`: DB connection string
- `--verbose` / `-v`: Debug logging
- `--dry-run`: Parse and print stats without DB load
- `--product-type`: Filter to specific types (investigation, instrument, target)
