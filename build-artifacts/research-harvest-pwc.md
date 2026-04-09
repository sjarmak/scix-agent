# Research: harvest-pwc-methods

## Codebase Patterns

### Download pattern (from `src/scix/uat.py`)

- Uses `urllib.request` with retry logic (3 attempts, exponential backoff)
- Saves to `data/` directory with `.parent.mkdir(parents=True, exist_ok=True)`
- Skips download if file already exists and is non-empty
- Custom User-Agent header

### Entity dictionary (`src/scix/dictionary.py`)

- `bulk_load(conn, entries)` accepts list of dicts with keys:
  - Required: `canonical_name`, `entity_type`, `source`
  - Optional: `external_id`, `aliases` (list[str]), `metadata` (dict)
- Uses ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
- Returns count of rows upserted

### DB connection (`src/scix/db.py`)

- `get_connection(dsn=None)` reads `SCIX_DSN` env var, defaults to `dbname=scix`

### Script pattern (from `scripts/load_uat.py`)

- `sys.path.insert(0, ...)` to add `src/` to path
- argparse CLI with `--dsn`, `-v/--verbose`
- logging.basicConfig setup

### Test pattern (from `tests/test_dictionary.py`)

- Uses `helpers.DSN` for connection string
- Integration tests marked with `@pytest.mark.integration`
- Fixture creates/cleans up test data

## Papers With Code Data Source

URL: `https://production-media.paperswithcode.com/about/methods.json.gz`
(Also available from GitHub raw: `https://github.com/paperswithcode/paperswithcode-data`)

The methods.json.gz contains a JSON array of method objects with fields:

- `name`: short method name (e.g. "Attention")
- `full_name`: longer name (e.g. "Attention Mechanism")
- `description`: markdown description of the method
- `paper`: object with paper details (title, url, arxiv_id)
- `introduced_year`: year the method was introduced
- `source_url`: URL to the method page on PWC
- `main_collection`: category/collection info

Expected count: ~1500+ methods (well over 1000 acceptance criterion).

## Mapping to entity_dictionary

- `canonical_name` = `full_name` or `name` (prefer full_name if present)
- `entity_type` = `'method'`
- `source` = `'pwc'`
- `aliases` = [name] if name != canonical_name
- `metadata` = {description, introduced_year, source_url, collection, paper info}
