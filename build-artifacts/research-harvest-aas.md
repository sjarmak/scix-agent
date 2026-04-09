# Research: harvest-aas-facilities

## Key Findings

### dictionary.py API

- `bulk_load(conn, entries)` accepts list of dicts with keys: canonical_name, entity_type, source, plus optional external_id, aliases, metadata
- `lookup(conn, name, entity_type=None)` does case-insensitive canonical_name match then alias fallback
- Uses psycopg (v3), ON CONFLICT upsert

### db.py

- `get_connection(dsn=None)` reads SCIX_DSN env var, returns psycopg.Connection

### CLI Pattern (harvest_ascl.py)

- sys.path.insert for src/scix
- argparse with --dsn, -v/--verbose
- logging.basicConfig
- download function with retry + exponential backoff (urllib.request)
- parse function returns list of dicts for bulk_load
- run_harvest orchestrates download -> parse -> bulk_load
- main() wires argparse

### AAS Facility Keywords Page

- URL: https://journals.aas.org/facility-keywords/
- HTML table with columns:
  1. Full Facility Name
  2. Keyword (abbreviation)
  3. Location
  4. Gamma-ray (> 120 keV)
  5. X-ray (0.1 - 100 Angstroms)
  6. Ultraviolet (100 - 3000 Angstroms)
  7. Optical (3000 - 10,000 Angstroms)
  8. Infrared (1 - 100 microns)
  9. Millimeter (0.1 - 10 mm)
  10. Radio (< 30 GHz)
  11. Neutrinos, particles, and gravitational waves
  12. Solar Facility
  13. Archive/Database
  14. Computational Center
- 500+ facility entries
- Each facility has a full name, keyword/abbreviation, location, and boolean flags for wavelength regimes
- Wavelength flags appear as text markers in the cells

### Mapping to entity_dictionary

- canonical_name: Full Facility Name
- entity_type: 'instrument'
- source: 'aas'
- external_id: None (no external IDs)
- aliases: [Keyword] (the abbreviation)
- metadata: {wavelength_regimes: [...], location: "...", facility_flags: [...]}

### Test Pattern

- Integration tests use db_conn fixture with DSN from helpers
- @pytest.mark.integration for DB-dependent tests
- Unit tests for parsing use mock/sample HTML
