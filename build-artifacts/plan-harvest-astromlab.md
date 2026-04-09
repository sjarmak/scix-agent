# Plan: harvest-astromlab

## Step 1: Create scripts/harvest_astromlab.py

### Constants

- `ASTROMLAB_CONCEPTS_URL`: GitHub raw URL for AstroMLab concept data
- `CATEGORY_TO_ENTITY_TYPE`: mapping dict for category -> entity_type
- `_DEFAULT_DEST`: Path("data/astromlab_concepts.csv")

### Functions

1. **`map_category_to_entity_type(category: str) -> str`**
   - Maps category string to entity_type using substring matching
   - "instrument" keywords: instrumental, instrumentation
   - "dataset" keywords: data, survey, catalog, database
   - Default: "method"

2. **`download_concepts(dest: Path | None = None, url: str = ASTROMLAB_CONCEPTS_URL) -> Path`**
   - Skip if file exists and non-empty
   - Download with retry (3 attempts, exponential backoff)
   - Return path to downloaded file

3. **`parse_concepts(data_path: Path) -> list[dict[str, Any]]`**
   - Auto-detect CSV vs JSON format
   - Extract concept_name, category from each row
   - Map category to entity_type via map_category_to_entity_type()
   - Build entry dict with source='astromlab', metadata={category: ...}
   - Skip entries with no concept name

4. **`load_concepts(entries: list[dict[str, Any]], dsn: str | None = None) -> int`**
   - get_connection(dsn), bulk_load(conn, entries), close

5. **`run_pipeline(data_path: Path | None, dsn: str | None) -> int`**
   - download (if no data_path) -> parse -> load
   - Log timing

6. **`main()`**
   - argparse: --data-file, --dsn, --url, -v/--verbose

## Step 2: Create tests/test_harvest_astromlab.py

### Test classes:

1. **TestMapCategory** - unit tests for category mapping
   - instrument categories map correctly
   - method categories map correctly
   - dataset categories map correctly
   - unknown categories default to method

2. **TestParseConcepts** - unit tests with mock CSV data
   - Basic parse with multiple categories
   - Skips empty concept names
   - Category stored in metadata
   - Large number of concepts

3. **TestDownloadConcepts** - mocked network
   - Skips existing file
   - Downloads when missing

4. **TestRunPipeline** - mocked DB
   - Pipeline with local file
   - File not found error

5. **TestLoadIntegration** (integration, marked) - real DB
   - Bulk load with test source tag
   - Cleanup after

## Step 3: Run tests, fix failures
