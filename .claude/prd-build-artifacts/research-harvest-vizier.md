# Research: harvest-vizier

## Codebase Patterns

### Harvest script pattern (from harvest_ascl.py, harvest_aas_facilities.py)

- Located in `scripts/` directory
- Uses `sys.path.insert(0, ...)` to add `src/` and import `scix.db.get_connection` and `scix.dictionary.bulk_load`
- Three main functions: `download_*()`, `parse_*()`, `run_harvest()`
- Uses `urllib.request` with retry/backoff for HTTP
- `argparse` CLI with `--dsn` and `--verbose` flags
- `bulk_load()` accepts list of dicts with keys: canonical_name, entity_type, source, external_id, aliases, metadata

### Test pattern (from test_harvest_ascl.py)

- Uses `unittest.mock.patch` to mock HTTP responses and DB connections
- Sample data defined as module-level constants
- Test classes: TestParse*, TestDownload*, TestRunHarvest, TestLargeCatalog
- Mocks `urllib.request.urlopen` for download tests
- Mocks `get_connection`, `bulk_load`, and download function for run_harvest tests

### entity_dictionary table

- Unique constraint on (canonical_name, entity_type, source)
- Fields: id, canonical_name, entity_type, source, external_id, aliases (array), metadata (JSONB)

## TAPVizieR Specifics

- Endpoint: `http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync`
- Protocol: POST with `REQUEST=doQuery`, `LANG=ADQL`, `FORMAT=votable`, `QUERY=<ADQL>`
- ADQL query: `SELECT name, description, utype FROM TAP_SCHEMA.tables WHERE schema_name NOT IN ('TAP_SCHEMA', 'ivoa')`
- Response format: VOTable XML with namespace `http://www.ivoa.net/xml/VOTable/v1.3`
- VOTable structure: `VOTABLE > RESOURCE > TABLE > DATA > TABLEDATA > TR > TD`
- The `name` field contains catalog IDs like `J/A+A/...`, `II/...`, etc.
- The `description` field contains the catalog title
- VizieR has ~25,000+ catalogs

## Mapping to entity_dictionary

- `canonical_name` = description (catalog title)
- `entity_type` = 'dataset'
- `source` = 'vizier'
- `external_id` = name (catalog ID, e.g. 'J/A+A/680/A81')
- `aliases` = [] (no meaningful aliases from TAP_SCHEMA.tables)
- `metadata` = {"utype": utype} if utype is non-empty
