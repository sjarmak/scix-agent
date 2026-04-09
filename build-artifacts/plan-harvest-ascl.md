# Plan: harvest-ascl

## Step 1: Create scripts/harvest_ascl.py

1. Add shebang, docstring, imports (urllib.request, json, argparse, logging, sys, pathlib)
2. sys.path.insert for src/scix
3. Import get_connection from scix.db, bulk_load from scix.dictionary
4. Constants: ASCL_URL = "https://ascl.net/code/json"
5. `download_ascl_catalog()` function:
   - Uses urllib.request with User-Agent header, timeout=60
   - Retry with exponential backoff (3 attempts)
   - Returns parsed JSON (list of dicts)
6. `parse_ascl_entries(raw_entries)` function:
   - Iterates over raw JSON entries
   - For each entry: extract title as canonical_name, ascl_id as external_id
   - Build aliases list from title variations (lowercase)
   - Build metadata dict with bibcode (if present)
   - Set entity_type='software', source='ascl'
   - Skip entries missing title or ascl_id
   - Returns list of dict entries for bulk_load
7. `run_harvest(dsn=None)` function:
   - Downloads catalog
   - Parses entries
   - Opens DB connection
   - Calls bulk_load
   - Logs count
8. `main()` CLI:
   - argparse with --dsn, -v/--verbose
   - Calls run_harvest

## Step 2: Create tests/test_harvest_ascl.py

1. Mock urllib.request.urlopen to return sample ASCL JSON
2. Test parse_ascl_entries with sample data:
   - Correct canonical_name, entity_type, source, external_id
   - Bibcode in metadata
   - Skips entries without title
3. Test download_ascl_catalog with mocked HTTP
4. Test end-to-end with mocked HTTP + mocked bulk_load
