# PRD: Remaining Harvesters + Harvester Modernization

## Problem Statement

The External Data Integration PRD (Milestone 1+2) delivered the entity graph schema, shared HTTP client, harvest_runs tracking, and 3 updated harvesters (GCMD, PDS4, SPDF). Four new harvesters remain unbuilt (SsODNet, SBDB, CMR, SPASE update), and 9 existing harvesters still use raw urllib with no harvest_runs logging or entity graph writes.

## Prior Work (Already Built)

- `migrations/019-022`: schema_migrations, harvest_runs, entity graph tables, staging
- `src/scix/http_client.py`: ResilientClient with retry, rate-limit, circuit breaker, disk cache
- `src/scix/harvest_utils.py`: HarvestRunLog, upsert_entity/identifier/alias/relationship helpers
- `scripts/harvest_gcmd.py`: writes entities + entity_identifiers + harvest_runs (ResilientClient)
- `scripts/harvest_pds4.py`: writes entities + entity_identifiers + entity_relationships + harvest_runs (ResilientClient)
- `scripts/harvest_spdf.py`: writes datasets + entities + entity_relationships + harvest_runs (ResilientClient)

## Goals

1. Build SsODNet, SBDB, CMR harvesters (3 new external sources)
2. Update SPASE harvester to v2.7.1 + entity graph writes
3. Modernize existing harvesters: ResilientClient + harvest_runs logging
4. Validate extraction→resolution quality before scale linking

## Non-Goals

- Materialized views and MCP tools (separate PRD)
- Full granule-level data ingestion
- Embedding-based entity linking

## Blocking Gate: Extraction Quality Evaluation

**Before building SsODNet or running linking at scale**, run a 50-paper manual evaluation:

- Sample 50 papers with extractions from `extractions` table
- Run entity resolver against their mentions
- Measure precision (>70% required) and recall (>50% required) on a hand-labeled 200-mention sample
- If below threshold: add normalization layer (lowercase, abbreviation expansion, parenthetical stripping)

This gate determines SsODNet scope: if <10K unique small-body names in extractions, use API-based seed. If >10K, download full ssoBFT.

## Requirements

### Must-Have

- **SsODNet harvester** (`scripts/harvest_ssodnet.py`)
  - Download ssoBFT Parquet file (2.1GB, ~1.2M objects) OR API-based seed (scope decided by blocking gate)
  - Use ResilientClient for download with SHA-256 checksum and HTTP Range resume
  - Parse into entities (canonical_name from IAU-preferred designation, entity_type='target', source='ssodnet')
  - Store entity_identifiers: ssodnet name (id_scheme='ssodnet'), SPK-ID (id_scheme='sbdb_spkid')
  - Store entity_aliases: all known designations
  - Store physical properties (diameter, albedo, taxonomy) in entities.properties
  - Use staging schema (migration 022) for bulk load — staging.promote_entities() for atomic promote
  - Log harvest_run with satellite count validation (alias/identifier counts proportional to entity counts)
  - Acceptance: `SELECT count(*) FROM entities WHERE source='ssodnet'` returns >1,000,000 (bulk) or >5,000 (seed); `SELECT * FROM entity_aliases WHERE lower(alias)='1999 rq36'` resolves to Bennu

- **SBDB enrichment** (`scripts/harvest_sbdb.py`)
  - Query JPL SBDB API for objects already in entities table (from SsODNet) to add orbital elements
  - Use ResilientClient with rate_limit={'ssd.jpl.nasa.gov': 1} (1 concurrent per IP)
  - Update entities.properties with orbital_class, neo (boolean), pha (boolean), discovery_date, discovery_site
  - Resumable via harvest_runs.cursor_state (last processed entity_id)
  - Log harvest_run
  - Acceptance: NEO-flagged entities have `properties->>'neo' = 'true'`; rate limiting enforced (no 429 errors)

- **CMR collection harvester** (`scripts/harvest_cmr.py`)
  - Use ResilientClient to paginate NASA CMR collections via Search-After headers
  - Request format: UMM-JSON (`Accept: application/vnd.nasa.cmr.umm_results+json`)
  - Store as datasets (source='cmr', canonical_id=concept-id)
  - Extract instruments, platforms, science_keywords; cross-reference GCMD entities via entity_identifiers(id_scheme='gcmd_uuid')
  - Deduplicate on concept-id during upsert
  - Log harvest_run with page counts
  - Acceptance: `SELECT count(*) FROM datasets WHERE source='cmr'` returns >8,000

- **SPASE harvester update** (update `scripts/harvest_spase.py`)
  - Replace urllib with ResilientClient
  - Update pinned version from v2.7.0 to v2.7.1 (new enumeration values: Amplitude, Magnitude, etc.)
  - Write to entities table (in addition to entity_dictionary) using harvest_utils.upsert_entity
  - Store entity_identifiers with id_scheme='spase_resource_id' where applicable
  - Log harvest_run via HarvestRunLog
  - Acceptance: New v2.7.1 values present in entities table; harvest_runs has row with source='spase'

- **Extraction quality evaluation** (`scripts/eval_extraction_quality.py`)
  - Sample 50 random papers from extractions table
  - Run entity resolver on all mentions
  - Output: precision, recall, match_method distribution, unmatched mentions list
  - Write results to `build-artifacts/extraction-quality-eval.md`
  - Acceptance: Script runs without error; output includes precision/recall numbers and unmatched mention examples

### Should-Have

- **Modernize existing harvesters to ResilientClient + harvest_runs**
  - Update these scripts to use ResilientClient (replace urllib): harvest_spase.py, harvest_ascl.py, harvest_aas_facilities.py, harvest_vizier.py, harvest_pwc_methods.py, harvest_physh.py, harvest_astromlab.py
  - Add HarvestRunLog to each (start/complete/fail lifecycle)
  - Acceptance: Each harvester imports ResilientClient and HarvestRunLog; harvest_runs table gets rows after running any harvester

- **Wikidata cross-linking for new entities**
  - Extend `scripts/enrich_wikidata_multi.py` to query entities from new entities table (not just entity_dictionary)
  - Store QIDs in entity_identifiers with id_scheme='wikidata'
  - Acceptance: >50% of planetary science entities in entities table have Wikidata QIDs in entity_identifiers

### Nice-to-Have

- **SsODNet scope decision query** (`scripts/count_extraction_mentions.py`)
  - Run `SELECT DISTINCT` on extraction mention strings, cross-reference against known small body catalogs
  - Output: count of unique small-body names, recommendation for API vs bulk

## Design Considerations

### SsODNet Staging Pattern

SsODNet loads 1.2M entities. Must use staging schema (migration 022) to avoid the multi-table promote bug (R2 from parent PRD): promote entities + identifiers + aliases atomically via staging.promote_entities().

### SBDB Rate Limiting

JPL SBDB enforces 1 concurrent request per IP. ResilientClient's per-host rate limiting handles this: `ResilientClient(rate_limits={'ssd.jpl.nasa.gov': 1.0})`.

### CMR Pagination

CMR uses Search-After headers (not offset-based). Server-side changes during pagination can invalidate cursors. Deduplicate on concept-id and accept <1% record loss at collection level.

## Implementation Order

```
Layer 0 (parallel, no deps):
  ├── extraction-quality-eval (blocking gate)
  ├── spase-harvester-update (smallest, proves pattern)
  └── harvester-modernization (existing harvesters → ResilientClient + HarvestRunLog)

Layer 1 (depends on extraction-quality-eval):
  ├── ssodnet-harvester (scope decided by eval)
  └── cmr-harvester (independent of ssodnet)

Layer 2 (depends on ssodnet):
  └── sbdb-enrichment

Layer 3 (depends on ssodnet + wikidata infra):
  └── wikidata-cross-linking
```
