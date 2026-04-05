# Plan: harvest-aas-facilities

## Implementation Steps

### 1. scripts/harvest_aas_facilities.py

1. **download_aas_facilities(url)** - Download HTML page from https://journals.aas.org/facility-keywords/ with retry/backoff (matching harvest_ascl.py pattern)
2. **parse_aas_facilities(html)** - Parse the HTML table using html.parser (stdlib):
   - Extract table headers to identify wavelength regime columns
   - For each row, extract: full name, keyword, location, and boolean flags for each wavelength regime
   - Return list of dicts ready for bulk_load
   - Map: canonical_name=full_name, entity_type='instrument', source='aas'
   - aliases=[keyword] if keyword differs from full name
   - metadata={wavelength_regimes: [...active regimes...], location: "...", facility_flags: [...]}
3. **run_harvest(dsn)** - Orchestrate download -> parse -> bulk_load
4. **main()** - argparse CLI with --dsn, -v/--verbose

### 2. tests/test_harvest_aas.py

1. Create SAMPLE_HTML constant with realistic mock HTML (2 sections, ~10 facilities)
2. **test_parse_returns_list** - parse_aas_facilities returns non-empty list
3. **test_parse_entry_structure** - each entry has required keys
4. **test_parse_wavelength_regimes** - wavelength flags land in metadata
5. **test_parse_aliases** - keyword becomes alias
6. **test_parse_count** - expected count from sample HTML
7. **test_hst_lookup_pattern** - verify HST-like entry has wavelength flags
8. Integration test (marked) for actual DB load if available

### Design Decisions

- Use html.parser from stdlib (no BeautifulSoup dependency needed)
- Wavelength regime names normalized to lowercase keys: gamma_ray, x_ray, ultraviolet, optical, infrared, millimeter, radio, neutrinos_particles_gw
- Facility flags: solar, archive_database, computational_center
- All boolean flags stored as list of active regime names in metadata.wavelength_regimes
