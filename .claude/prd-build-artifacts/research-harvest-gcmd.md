# Research: harvest-gcmd

## GCMD Data Sources

### Source 1: adiwg/gcmd-keywords GitHub repo

- URL pattern: `https://raw.githubusercontent.com/adiwg/gcmd-keywords/master/resources/json/{scheme}.json`
- Available schemes: `instruments`, `platforms`, `sciencekeywords`
- Format: nested JSON with `uuid`, `label`, `broader`, `children` fields
- Structure: single-element array with root node; hierarchy via nested `children` arrays
- Leaf nodes: no `children` key or empty array

### Source 2: CMR KMS API (for providers and projects)

- URL pattern: `https://cmr.earthdata.nasa.gov/kms/concepts/concept_scheme/{scheme}?format=json`
- Available schemes: `providers` (3708 entries), `projects`
- Format: flat `concepts` array with `uuid`, `prefLabel`, `scheme`, `definitions`
- Pagination: `page_size` (default 2000), `page_num`, `hits` fields
- No nested hierarchy — flat list

## JSON Structure: GitHub Schemes

```json
[{
  "uuid": "...",
  "label": "Instruments",
  "children": [
    {
      "broader": "parent-uuid",
      "uuid": "...",
      "label": "Solar/Space Observing Instruments",
      "children": [...]
    }
  ]
}]
```

Fields per node: `uuid`, `label`, `broader` (parent uuid), `children` (optional array), `hasDefinition` (optional 0/1)

## JSON Structure: KMS API

```json
{
  "hits": 3708,
  "page_num": 1,
  "page_size": 2000,
  "concepts": [
    {
      "uuid": "...",
      "prefLabel": "DOC/NOAA/NESDIS/STAR",
      "scheme": { "shortName": "providers", "longName": "Providers" },
      "definitions": [{ "text": "...", "reference": "..." }]
    }
  ]
}
```

## Existing Pattern (harvest_ascl.py, harvest_physh.py)

- urllib.request with retry + exponential backoff
- Parse into dicts with: canonical_name, entity_type, source, external_id, aliases, metadata
- bulk_load(conn, entries, discipline=...) for DB insert
- argparse CLI with --dsn, --verbose

## Mapping Plan

| GCMD Scheme              | entity_type | Source API |
| ------------------------ | ----------- | ---------- |
| instruments              | instrument  | GitHub     |
| platforms                | instrument  | GitHub     |
| sciencekeywords (leaves) | observable  | GitHub     |
| providers                | mission     | KMS API    |
| projects                 | mission     | KMS API    |

All entries: source='gcmd', discipline='earth_science'
