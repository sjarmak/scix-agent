# Plan: harvest-gcmd

## Implementation Steps

### 1. Define constants and scheme config

- GITHUB_BASE_URL for adiwg/gcmd-keywords raw JSON
- KMS_BASE_URL for CMR KMS API
- SCHEME_CONFIG dict mapping scheme name to: url, entity_type, source_type ('github'|'kms')

### 2. Download functions

- `download_github_json(url)` — fetch JSON from GitHub, retry with backoff, return parsed list
- `download_kms_json(scheme)` — fetch paginated JSON from KMS API, collect all pages, return concepts list

### 3. Parse GitHub hierarchy schemes

- `parse_github_hierarchy(root_nodes, scheme, entity_type)` — recursive walk of `children` arrays
  - Build breadcrumb path from root to each node
  - For sciencekeywords: only emit leaf nodes (no children)
  - For instruments/platforms: emit all nodes
  - Track (canonical_name, entity_type, source) tuples for duplicate detection
  - Disambiguate duplicates by prefixing with parent category

### 4. Parse KMS flat schemes

- `parse_kms_entries(concepts, scheme, entity_type)` — iterate flat concepts list
  - canonical_name from prefLabel
  - metadata includes gcmd_scheme, definitions
  - For providers: parse slash-separated hierarchy from prefLabel into gcmd_hierarchy

### 5. Unified harvest function

- `harvest_scheme(scheme_name)` — dispatch to correct download+parse based on config
- `harvest_all()` — loop over all 5 schemes

### 6. Duplicate disambiguation

- After collecting all entries for a scheme, find collisions on canonical_name
- For collisions: replace canonical_name with "parent > name" format
- Original short name goes into aliases

### 7. CLI with argparse

- `--help`, `--dsn`, `--verbose`, `--dry-run`, `--scheme`
- `--dry-run`: parse and print counts, skip DB write
- `--scheme`: harvest only one scheme (instruments, platforms, sciencekeywords, providers, projects)

### 8. Tests

- Mock urllib.request.urlopen for all downloads
- Synthetic fixtures mimicking each JSON structure
- Test hierarchy parsing, leaf extraction, duplicate disambiguation
- Test metadata structure (gcmd_scheme, gcmd_hierarchy, uuid)
- Test --dry-run and --scheme flags
- Test entry counts against acceptance criteria thresholds
