# Research: harvest-pds4

## PDS Registry API

- Base URL: `https://pds.nasa.gov/api/search/1/products` (redirects to `pds.mcp.nasa.gov`)
- Query by logical_identifier prefix to filter context product types
- Returns JSON with `summary.hits` count and `data` array

## Query patterns (verified)

| Type          | Query                                                                                        | Hits |
| ------------- | -------------------------------------------------------------------------------------------- | ---- |
| Investigation | `pds:Identification_Area.pds:logical_identifier like "urn:nasa:pds:context:investigation:*"` | 137  |
| Instrument    | `pds:Identification_Area.pds:logical_identifier like "urn:nasa:pds:context:instrument:*"`    | 748  |
| Target        | `pds:Identification_Area.pds:logical_identifier like "urn:nasa:pds:context:target:*"`        | 1543 |

## Response structure

Each product in `data` array:

- `id`: Full versioned URN (e.g. `urn:nasa:pds:context:investigation:mission.cassini-huygens::1.0`)
- `title`: Human-readable name
- `properties`: Dict with dot-notation keys:
  - `pds:Identification_Area.pds:logical_identifier`: Base URN (no version)
  - `pds:Identification_Area.pds:title`: Title (array of 1)
  - `pds:Identification_Area.pds:alternate_title`: May be `["null"]` string
  - `pds:Investigation.pds:type` / `pds:Target.pds:type`: Classification
  - `pds:Investigation.pds:description` / `pds:Target.pds:description`: Description text

## Pagination

- Use `limit` and `start` query params
- Default limit appears to be 100
- Use `summary.hits` to know total count

## Alias extraction

- `pds:Identification_Area.pds:alternate_title` — often `"null"` string
- `pds:Alias_List.pds:Alias.pds:alternate_title` — sometimes present
- Can also extract abbreviations from title parentheses, e.g. "The Catalina Sky Survey (CSS)" -> alias "CSS"

## Entity type mapping

- Investigation -> entity_type='mission'
- Instrument -> entity_type='instrument'
- Target -> entity_type='target'

## Existing harvester pattern (from harvest_ascl.py)

- `download_*()` -> fetch raw data with retry
- `parse_*()` -> transform to dict records
- `run_harvest()` -> orchestrate download + parse + bulk_load
- `main()` -> argparse CLI
- Uses `scix.dictionary.bulk_load()` with `discipline` kwarg
