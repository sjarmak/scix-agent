# Research: harvest-astromlab

## Codebase Patterns

### Existing harvest scripts follow a consistent pattern:

1. **download** function: fetches data from URL with retry/backoff
2. **parse** function: transforms raw data into `entity_dictionary` entry dicts
3. **load** function: calls `bulk_load(conn, entries)` via `get_connection(dsn)`
4. **run pipeline** function: orchestrates download -> parse -> load
5. **CLI main**: argparse with --dsn, --verbose, optional --data-file

### Entity dictionary entry format (from `dictionary.bulk_load()`):

```python
{
    "canonical_name": str,      # required
    "entity_type": str,         # required (e.g. 'method', 'instrument')
    "source": str,              # required (e.g. 'pwc', 'aas')
    "external_id": str | None,  # optional
    "aliases": list[str],       # optional, defaults to []
    "metadata": dict,           # optional, defaults to {}
}
```

### Test patterns:

- Mock data factories (no network, no DB for unit tests)
- `@pytest.mark.integration` for DB tests
- Use `helpers.DSN` for DB connection
- sys.path.insert for script imports
- Cleanup test data by source tag (e.g. 'pwc-test')

## AstroMLab Data Source

The AstroMLab project (https://github.com/AstroMLab) publishes astronomy concept vocabularies. The main concept list is from their benchmark paper with ~9,999 concepts across astronomical categories.

### Data format (expected):

CSV or JSON with fields like:

- concept_name / concept
- category (e.g. "Instrumental Design", "Numerical Methods", etc.)
- Possibly: description, subcategory

### Category -> entity_type mapping:

- Instrumental Design, Instrumentation -> instrument
- Numerical Methods, Computational Methods -> method
- Data/Surveys categories -> dataset
- Everything else -> method (default, most are techniques/approaches)

### Fallback strategy:

If GitHub URL is unavailable at runtime, the script should accept a local --data-file path. Tests will use mock data matching the expected format.

## Key imports needed:

- `src/scix/db.get_connection`
- `src/scix/dictionary.bulk_load`
- stdlib: urllib.request, csv, json, argparse, logging
