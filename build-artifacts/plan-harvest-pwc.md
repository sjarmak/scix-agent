# Plan: harvest-pwc-methods

## Step 1: Create `scripts/harvest_pwc_methods.py`

### 1a: Download function

- Download `methods.json.gz` from PWC production media URL
- Save to `data/methods.json.gz`
- Skip if already exists (same pattern as UAT download)
- Retry logic (3 attempts, exponential backoff)
- Decompress gzip and parse JSON array

### 1b: Parse function

- Parse each method object from the JSON array
- Extract: name, full_name, description, paper info, introduced_year, source_url, main_collection
- Build entity_dictionary entries:
  - `canonical_name` = full_name if present and non-empty, else name
  - `entity_type` = "method"
  - `source` = "pwc"
  - `aliases` = [name] when name differs from canonical_name
  - `metadata` = {description, introduced_year, source_url, collection, paper}
- Return list of entry dicts

### 1c: Load function

- Accept parsed entries and a DB connection
- Call `dictionary.bulk_load(conn, entries)`
- Log count loaded

### 1d: CLI main

- argparse with --data-file, --dsn, --verbose
- Wire download -> parse -> load pipeline

## Step 2: Create `tests/test_harvest_pwc.py`

### 2a: Unit tests (no DB, no network)

- Test parse function with mock JSON data (fake method objects)
- Test that canonical_name uses full_name when available
- Test that aliases include name when different from canonical_name
- Test empty/missing fields handled gracefully

### 2b: Integration test (marked, needs DB)

- Load parsed entries via bulk_load
- Verify count > 0

### 2c: Mock download test

- Patch urllib to return fake gzip data
- Verify download + parse pipeline works end-to-end

## Step 3: Run tests, fix failures

## Step 4: Commit
