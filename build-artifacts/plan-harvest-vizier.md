# Plan: harvest-vizier

## Implementation Steps

### 1. Create scripts/harvest_vizier.py

Functions to implement:

1. **`query_tap_vizier(url) -> bytes`** — POST ADQL query to TAPVizieR sync endpoint, return raw VOTable XML bytes. Retry with exponential backoff (3 attempts).

2. **`parse_votable_catalogs(xml_bytes) -> list[dict]`** — Parse VOTable XML using xml.etree.ElementTree. Find FIELD elements to determine column order (name, description, utype). Extract TR/TD rows. Return list of dicts with keys: name, description, utype.

3. **`build_dictionary_entries(catalogs) -> list[dict]`** — Convert raw catalog dicts into entity_dictionary format:
   - canonical_name = description (catalog title), fallback to name if description empty
   - entity_type = 'dataset'
   - source = 'vizier'
   - external_id = name (catalog ID)
   - aliases = []
   - metadata = {"utype": utype} if utype present

4. **`run_harvest(dsn) -> int`** — Orchestrate: query -> parse -> build -> bulk_load. Return count.

5. **`main()`** — argparse CLI with --dsn, --verbose.

### 2. Create tests/test_harvest_vizier.py

Test classes:

1. **TestParseVotableCatalogs** — Parse sample VOTable XML, verify correct extraction of name/description/utype fields, handle empty descriptions, empty input.

2. **TestBuildDictionaryEntries** — Verify entity_type='dataset', source='vizier', external_id mapping, canonical_name from description, metadata with utype.

3. **TestQueryTapVizier** — Mock urllib.request.urlopen, verify retry behavior, verify POST parameters.

4. **TestRunHarvest** — Mock all externals, verify bulk_load called with correct entries, connection closed.

5. **TestLargeCatalog** — Simulate 30,000 entries to verify >25,000 threshold.

### 3. VOTable XML namespace handling

The VOTable namespace `http://www.ivoa.net/xml/VOTable/v1.3` must be used in all XPath queries. Use a namespace dict for ElementTree findall().

### 4. Pagination consideration

TAPVizieR may return all results in one query since TAP_SCHEMA.tables is a metadata table. No pagination needed — single query should return all catalogs. If MAXREC is needed, set it high (e.g. 100000).
