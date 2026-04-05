# PRD: Multi-Discipline Ontology Coverage for SciX

## Problem Statement

The current SciX entity dictionary covers only astrophysics (ASCL, PhySH, AAS Facilities, VizieR, PwC, AstroMLab). SciX serves 4 disciplines — astrophysics, earth science, heliophysics, and planetary science — but 3 of 4 have zero vocabulary coverage. This creates a blind spot for dictionary-based entity matching, alias resolution, and normalization for the majority of SciX's expanding user base.

## Goals & Non-Goals

### Goals

- Extend entity_dictionary to cover earth science, heliophysics, and planetary science vocabularies
- Maintain the same harvester pattern (download → parse → bulk_load) established in Phase 1
- Add new entity_types where existing types are insufficient
- Enable cross-discipline alias resolution via Wikidata QIDs as universal linker
- Preserve discipline provenance in metadata for disambiguation

### Non-Goals

- Building a unified ontology from scratch (use existing authoritative registries)
- Complete deduplication across all vocabularies (handle at query time)
- Real-time synchronization with upstream registries (batch harvesting is sufficient)
- Replacing ADS's existing keyword systems (supplement, not replace)

## Research Findings (5-Agent Divergent Research)

### Agent 1: GCMD Science Keywords (Earth Science)

**Source:** NASA Global Change Master Directory
**Scale:** ~14,000+ terms across 14 keyword schemes (6-level hierarchy)
**Formats:** JSON, CSV, XML, RDF/SKOS — freely downloadable
**Access:** REST API via CMR, GitHub mirror (adiwg/gcmd-keywords), static KMS directory
**Key schemes for entity_dictionary:**

- Instruments/Sensors → entity_type: instrument (4-level hierarchy with short/long names)
- Platforms/Sources → entity_type: instrument (satellites, aircraft, ground stations)
- Data Centers → entity_type: facility/organization
- Projects → entity_type: mission
- Earth Science Keywords ��� entity_type: method/observable (6-level variable hierarchy)
  **Coverage:** Excellent for earth science; limited heliophysics (Sun-Earth only); no planetary science
  **Update frequency:** Semi-regular, 3-6 month cycles (current v23.x)
  **GitHub:** https://github.com/adiwg/gcmd-keywords (JSON, npm package)

### Agent 2: SPASE (Heliophysics)

**Source:** Space Physics Archive Search and Extract consortium
**Scale:** 19 resource types, ~500+ measurement types, ~200+ observatories
**Formats:** XML Schema (XSD), tab-delimited vocabulary files
**Access:** GitHub (spase-group/spase-base-model), schema site (spase-group.org/data/schema/)
**Key vocabularies:**

- Observatory/Instrument resources → entity_type: instrument
- NumericalData/DisplayData → entity_type: dataset
- MeasurementType enumerations → entity_type: observable
- ObservedRegion enumerations → metadata enrichment
- InstrumentType enumerations → metadata for instruments
  **Registry:** Heliophysics Data Portal (heliophysicsdata.gsfc.nasa.gov) — 100+ observatories
  **Vocabulary source:** Tab-delimited files (dictionary.tab, list.tab, member.tab) in spase-base-model
  **Update:** Community-governed via SPASE Metadata Working Team

### Agent 3: PDS4 (Planetary Science)

**Source:** NASA Planetary Data System (4th generation)
**Scale:** 1 core dictionary + 22 discipline LDDs + 36+ mission-specific dictionaries
**Formats:** XML Schema (XSD), Schematron, JSON, CSV
**Access:** pds.nasa.gov/datastandards/dictionaries/, GitHub (pds-data-dictionaries)
**Key structures:**

- Investigation (mission) context products → entity_type: mission
- Instrument context products → entity_type: instrument
- Target context products → entity_type: target (new entity_type needed)
- 36 mission namespaces (Apollo through OSIRIS-REx) with instrument enumerations
  **IAU Nomenclature:** 15,361 approved planetary feature names across 41 bodies — machine-readable at planetarynames.wr.usgs.gov
  **PDS Registry API:** REST API for programmatic access (JSON, XML, CSV)
  **External IDs:** PDS URNs (urn:nasa:pds:...) for bidirectional ADS linking

### Agent 4: SWEET + CF Standard Names (Earth Science Deep)

**SWEET (Semantic Web for Earth and Environmental Terminology):**

- Scale: ~6,000 concepts across ~200 ontologies
- Format: OWL/Turtle (RDF), GitHub (ESIPFed/sweet), CC0 license
- Modules: Realms (atmosphere, ocean, earth), Phenomena, Processes, Properties, Substances
- Relevance: Processes/phenomena → entity_type: method; Properties → entity_type: observable
- Status: v3.5.0 (July 2022), actively maintained by ESIP

**CF Standard Names:**

- Scale: ~4,500 standardized variable names with units and definitions
- Format: XML (cf-standard-name-table.xml), GitHub
- Structure: Compound names (e.g., "mass_concentration_of_dust_dry_aerosol_particles_in_air")
- Relevance: Measurement variables → entity_type: observable (new type needed)
- Update: Every 1-2 months, actively maintained

**NASA STI Thesaurus:**

- Scale: 18,400+ terms, 4,300 definitions
- Format: RDF/SKOS, OWL, CSV — freely downloadable
- Coverage: Aerospace, earth science, biological science, planetary science
- Relevance: Broad subject indexing → overlaps with GCMD but broader aerospace coverage

### Agent 5: Cross-Cutting Architecture

**New entity_types needed:**

- `mission` / `project` — PDS missions, GCMD projects, SPASE observatories
- `observable` / `variable` — CF Standard Names, SPASE MeasurementType, SWEET properties
- `target` — Planetary bodies, earth regions, atmospheric layers
- `facility` / `organization` — Data centers, observatories, research stations

**Name collision risk:** Moderate-to-high. "Solar", "wind", "radiation" mean different things across disciplines. Mitigate with discipline metadata and Wikidata QID disambiguation.

**Wikidata as cross-linker:** IVOA already recommends Wikidata for facility vocabulary (NOTE-ObsFacilityWikidata-1.0). Store wikidata_qid for every entity where available.

**Schema extension:** Add `discipline` and `vocabulary_source` to metadata JSONB. Store parent_id and parent_path for hierarchical vocabularies. No table restructuring needed — metadata JSONB handles the extension.

**Scale estimate:** 5,000-9,000 new entries (after deduplication), bringing total dictionary from ~48K to ~55-57K entries.

**IVOA UCDs:** ~1,000 terms describing physical quantities. These are metadata enrichment (describing dataset columns), not entities. Store as metadata.ucd field, not as entity_dictionary rows.

## Convergence Decisions

Based on structured review by prioritization, architecture, and domain reviewers:

1. **SWEET cut.** OWL parsing complexity not justified given GCMD overlap. Cut entirely.
2. **GCMD unified harvester.** All GCMD schemes (Instruments, Platforms, Science Keywords, Data Centers, Projects) share the same JSON format from the same GitHub repo — implement as one harvester with multiple output modes.
3. **GCMD Data Centers & Projects promoted to Must-Have.** Zero marginal effort when combined with GCMD instruments harvester.
4. **CF Standard Names demoted to Nice-to-Have.** Compound names (e.g., "mass_concentration_of_dust_dry_aerosol_particles_in_air") produce near-zero recall in entity matching against paper text. Only useful after root-term decomposition step.
5. **IAU Nomenclature demoted to Nice-to-Have.** 15K features is too granular; papers reference ~200-300 prominent features. A curated shortlist delivers 90% of value at 3% of effort.
6. **`discipline` promoted to top-level column.** Convergence review determined JSONB burial is insufficient for the filtering patterns this field enables. Requires migration 014.
7. **`facility` entity_type deferred.** AAS Facility Keywords already stores observatories as entity_type='instrument'. Resolve overlap before adding a new type. Use mission, observable, target for now.
8. **bulk_load needs executemany().** Current per-row INSERT loop is wasteful at 60K scale. Code-only change.
9. **Add functional index on lower(canonical_name).** Current lookup() does case-insensitive search without supporting index. Migration 014.

## Requirements

### Must-Have (Layer 0)

- **Migration 014: schema hardening**
  - Add `discipline TEXT` column (nullable, indexed) to entity_dictionary
  - Add functional index `idx_entity_dict_canonical_lower ON entity_dictionary (lower(canonical_name))`
  - Backfill discipline from metadata->>'discipline' for existing rows
  - Acceptance: migration applies cleanly, lookup() uses index (EXPLAIN shows index scan)

- **Code hardening**
  - Add `ALLOWED_ENTITY_TYPES` frozenset in dictionary.py; validate in upsert_entry() and bulk_load()
  - Replace per-row INSERT loop in bulk_load() with executemany()
  - Acceptance: bulk_load of 10K entries completes in < 2s; invalid entity_type raises ValueError

### Must-Have (Layer 1)

- **GCMD unified harvester** (scripts/harvest_gcmd.py)
  - Download all GCMD keyword schemes as JSON from GitHub (adiwg/gcmd-keywords) or KMS static directory
  - Parse into entity_dictionary entries for 5 schemes:
    - Instruments/Sensors → entity_type='instrument', source='gcmd'
    - Platforms/Sources → entity_type='instrument', source='gcmd'
    - Earth Science Keywords (leaf nodes) → entity_type='observable', source='gcmd'
    - Data Centers → entity_type='mission', source='gcmd' (data providers as organizational missions)
    - Projects → entity_type='mission', source='gcmd'
  - metadata: {discipline: 'earth_science', gcmd_scheme, gcmd_hierarchy: 'Category>Class>Type>Subtype', short_name, long_name, uuid}
  - discipline column: 'earth_science'
  - Acceptance: > 1,000 instrument entries, > 500 platform entries, > 3,000 science keyword entries, > 200 data center entries, > 100 project entries

- **SPASE vocabulary harvester** (scripts/harvest_spase.py)
  - Download tab-delimited vocabulary files from spase-group/spase-base-model GitHub
  - Parse MeasurementType, InstrumentType, ObservedRegion enumerations
  - MeasurementType → entity_type='observable', source='spase'
  - InstrumentType → entity_type='instrument', source='spase' (as type categories, not instances)
  - ObservedRegion → metadata enrichment entries with entity_type='observable', source='spase'
  - discipline column: 'heliophysics'
  - Acceptance: > 50 measurement types, > 30 instrument types, > 40 observed regions

### Must-Have (Layer 2)

- **PDS4 context harvester** (scripts/harvest_pds4.py)
  - Download PDS4 context products (missions, instruments, targets) from PDS Registry API or GitHub
  - Parse Investigation → entity_type='mission', Instrument → entity_type='instrument', Target → entity_type='target'
  - external_id: PDS URN (urn:nasa:pds:...)
  - discipline column: 'planetary_science'
  - Acceptance: > 30 missions, > 100 instruments, > 50 targets

### Should-Have (Layer 2)

- **Wikidata cross-linking enrichment (multi-discipline)**
  - Extend existing enrich_wikidata_instruments.py pattern to cover new entity_types (mission, observable, target)
  - Query Wikidata for QIDs and aliases for GCMD instruments, PDS4 missions, SPASE observatories
  - Acceptance: > 50% of new instrument entries have wikidata_qid

### Nice-to-Have (Layer 3)

- **CF Standard Names harvester** — only with root-term decomposition (extract ~300 root physical quantities, not all 4,500 compound names)
- **IAU Planetary Nomenclature** — curated shortlist of ~300 high-profile features, not full 15,361
- **NASA STI Thesaurus harvester** — 18,400 terms (RDF/SKOS, broad but redundant with GCMD)
- **SPASE registry harvester** — Full observatory/instrument/dataset catalog from HPDE

## Design Considerations

### Entity Type Extension

Current types: instrument, dataset, software, method
New types: **mission, observable, target**

`facility` deferred — audit AAS Facility Keywords overlap first.

Add `ALLOWED_ENTITY_TYPES` validation in dictionary.py (code-only, no migration).

### Discipline Column + Metadata Convention

`discipline` is a **top-level TEXT column** (migration 014), not buried in JSONB. This enables efficient `WHERE discipline = 'earth_science'` filtering with a btree index.

Additional metadata JSONB fields per convention:

```json
{
  "vocabulary_source": "gcmd|spase|pds4|cf|iau|sti",
  "parent_path": "optional hierarchy breadcrumb",
  "wikidata_qid": "optional cross-reference",
  "gcmd_scheme": "instruments|platforms|science_keywords|data_centers|projects",
  "gcmd_hierarchy": "Category > Class > Type > Subtype"
}
```

### Disambiguation Strategy

For name collisions across disciplines:

1. Filter by `discipline` column (primary, indexed)
2. Use `source` field to identify vocabulary provenance
3. Wikidata QID as universal cross-reference disambiguator
4. Application-level context (user's active discipline) selects preferred match

### Hierarchy Storage

Store `parent_path` as a string in metadata JSONB for display and simple filtering. At 60K rows, `LIKE` queries on parent_path are fast enough (< 10ms). If hierarchical traversal becomes a hot path, upgrade to `ltree` column in a future migration. Document this as a known upgrade path.

### Update Frequency

| Source            | Cadence     | Method                           |
| ----------------- | ----------- | -------------------------------- |
| GCMD Keywords     | Quarterly   | Re-download JSON from GitHub/KMS |
| SPASE             | Semi-annual | Pull GitHub tab-delimited files  |
| PDS4              | Per-release | Query PDS Registry API           |
| CF Standard Names | Monthly     | Re-download XML                  |
| IAU Nomenclature  | Annual      | Re-scrape database               |

## Resolved Questions

1. **observable vs variable:** Unified as `observable`. Variables are a subset of observables.
2. **SWEET:** Cut. GCMD overlap too high for OWL complexity.
3. **facility entity_type:** Deferred. Audit AAS overlap first.
4. **GCMD hierarchy depth:** Store leaf nodes as canonical entries; full hierarchy path in metadata for context.
5. **discipline storage:** Promoted to top-level column (migration 014), not JSONB.

## Open Questions

1. How many GCMD/SPASE instruments overlap with existing AAS Facility Keywords?
2. What is the actual usage of PDS URNs in ADS bibliographic records?
3. What fraction of SciX papers are earth science vs heliophysics vs planetary science? (Drives priority allocation.)
4. Should CF Standard Names undergo root-term decomposition before ingestion?

## Estimated Scale (Must-Have Only)

| Source                                   | New Entries      | Entity Types                |
| ---------------------------------------- | ---------------- | --------------------------- |
| GCMD Instruments + Platforms             | ~1,500           | instrument                  |
| GCMD Earth Science Keywords (leaf nodes) | ~3,000-5,000     | observable                  |
| GCMD Data Centers + Projects             | ~300             | mission                     |
| SPASE vocabularies                       | ~200             | observable, instrument      |
| PDS4 context products                    | ~200             | mission, instrument, target |
| **Total Must-Have**                      | **~5,200-7,200** |                             |

Combined with existing ~48K entries, total dictionary: **~53-55K entries** (Must-Have).

With Nice-to-Have (CF root terms ~300, IAU shortlist ~300, STI ~18K): up to ~73K entries.

## Risk Register

### R1: GCMD common-word observables swamp entity extraction (CRITICAL)

- **Likelihood:** High | **Impact:** High
- **Failure mode:** GCMD Earth Science Keywords include "Temperature", "Pressure", "Wind Speed" as observable entries. These match 60%+ of earth science abstracts, destroying extraction precision.
- **Mitigation:** Filter GCMD observables: discard any term < 3 tokens or appearing in > 5% of corpus abstracts (TF-IDF gate). Prefer instrument/mission-specific leaf nodes over generic measurement terms. Consider a two-stage pipeline: NER candidate spans first, then dictionary reranking.

### R2: NULL discipline excludes 48K astrophysics entries (CRITICAL)

- **Likelihood:** High | **Impact:** High
- **Failure mode:** Migration 014 adds `discipline` column but leaves existing rows NULL. Any downstream query filtering `WHERE discipline = 'astrophysics'` silently returns zero results for the entire existing corpus.
- **Mitigation:** Backfill immediately in migration 014: `UPDATE entity_dictionary SET discipline = 'astrophysics' WHERE source IN ('ascl','aas','physh','pwc','astromlab','vizier','ads_data')`. Add integration test asserting zero NULL discipline rows post-migration.

### R3: GCMD UNIQUE constraint collisions on duplicate leaf names

- **Likelihood:** High | **Impact:** Medium
- **Failure mode:** "Precipitation" appears under both Atmosphere and Cryosphere in GCMD. UNIQUE(canonical_name, entity_type, source) treats these as duplicates — ON CONFLICT silently overwrites one.
- **Mitigation:** Use GCMD UUID as `external_id` and include hierarchy-qualified names (or use UUID in the unique key). Alternatively, prefix canonical_name with the parent topic for ambiguous leaf nodes.

### R4: SPASE/PDS4 vocabulary terms don't match natural language

- **Likelihood:** High | **Impact:** High
- **Failure mode:** SPASE uses PascalCase identifiers ("MagneticField"), PDS4 uses formal names ("Mars Hand Lens Imager (MAHLI)"). Papers write "magnetic field" and "MAHLI". Dictionary matching misses abbreviations and natural-language variants.
- **Mitigation:** Build synonym expansion layer from description text. Harvest abbreviation/alternate_id fields from PDS4. CamelCase-split SPASE terms into space-separated aliases. Require co-occurrence with instrument mention for observable matching.

### R5: External data source format changes break harvesters

- **Likelihood:** Medium | **Impact:** High
- **Failure mode:** GCMD GitHub repo archived or JSON format changed. SPASE tab files restructured between versions. PDS4 API adds OAuth2.
- **Mitigation:** Pin to specific commit SHAs/tagged releases. Mirror downloads to `data/snapshots/` with checksums. Parse SPASE by header name, not column index. Schema-validate all inputs with Pydantic models before writing. Cache full harvest output to JSONL for restart capability.

### R6: Wikidata SPARQL rate limiting at scale

- **Likelihood:** High | **Impact:** Medium (enrichment is additive, not critical path)
- **Failure mode:** 600+ facility SPARQL queries hit 60-second timeout and trigger 24-hour IP block.
- **Mitigation:** Batch queries (50 per request). Mandatory 2-second sleep between batches. Cache all SPARQL results to disk. Consider local WDQS mirror for large-scale enrichment.

### R7: Downstream code breaks on new entity_types

- **Likelihood:** High | **Impact:** Medium
- **Failure mode:** Code that switches on entity_type only handles instrument/dataset/software/method. New mission/observable/target values fall through to default no-op or raise.
- **Mitigation:** Audit all entity_type switch/if-chains before shipping. Add CI test asserting every entity_type in the table has a handler in each consuming service. Add `ALLOWED_ENTITY_TYPES` frozenset validation.
