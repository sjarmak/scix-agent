# SciX External Data Sources Reference

This document catalogs the authoritative external APIs for enriching the SciX literature database with structured, cross-domain metadata. It serves as the canonical reference for agents building ingestion pipelines, entity resolvers, and linking systems.

## Design Principles

1. **Authoritative IDs over names** - Always store external canonical identifiers
2. **Structured metadata over embeddings** - Prefer controlled vocabularies and typed relationships
3. **Object-centric systems over archives** - Prioritize APIs that model entities, not just files
4. **Provenance tracking** - Every enrichment traces back to source + timestamp + confidence

## Canonical ID Strategy

| Domain                | Source of Truth | ID Example                       |
| --------------------- | --------------- | -------------------------------- |
| Small bodies          | SsODNet         | `ssodnet:bennu`                  |
| Planetary data        | PDS Registry    | `urn:nasa:pds:orex.ocams`        |
| Earth datasets        | CMR             | `C1234567890-PROVIDER`           |
| Earth vocabularies    | GCMD            | `gcmd:uuid`                      |
| Heliophysics data     | SPASE           | `spase://NASA/NumericalData/...` |
| Heliophysics metadata | SPDF/CDAWeb     | `AC_H2_MFI`                      |

---

## Planetary Science Sources

### SsODNet (Solar System Open Database Network)

- **Role**: Canonical small-body identity, aliases, physical/dynamical properties
- **Discipline**: Planetary science
- **Base URL**: `https://ssp.imcce.fr/webservices/ssodnet/api/`
- **API type**: REST (JSON)
- **Auth**: None
- **Key endpoints**:
  - `/resolver/` - Name resolution: given a name/designation, returns canonical object + all known aliases
  - `/ssocard/` - Full metadata card for a named object (physical properties, taxonomy, dynamical family)
- **Record types**: object identity, physical properties, dynamical properties, taxonomy, aliases
- **Canonical ID field**: `name` (IAU-preferred designation from resolver)
- **Alias fields**: `aliases[]` from resolver response
- **Relationship fields**: `dynamical_family`, `taxonomy.class`
- **Update frequency**: Weekly
- **Priority**: 1 (highest for small-body resolution)
- **Caching**: Aggressive - high reuse across papers mentioning the same objects
- **Agent notes**: Use resolver endpoint first for any small-body name mentioned in papers. The resolver handles designations, provisional names, and common aliases (e.g., "1999 RQ36" -> "Bennu").

### NASA Planetary Data System (PDS)

- **Role**: Mission, instrument, and dataset structure for planetary science
- **Discipline**: Planetary science
- **Base URL**: `https://pds.nasa.gov/api/search/1/`
- **API type**: REST (JSON, Solr-backed)
- **Auth**: None
- **Key endpoints**:
  - `/products` - Search across all product types
  - `/classes/Product_Context` - Missions, instruments, targets, facilities
  - `/classes/Product_Bundle` - Dataset bundles (top-level collections)
  - `/classes/Product_Collection` - Dataset collections within bundles
- **Record types**: missions (Investigation), instruments (Instrument), targets (Target), datasets (Bundle/Collection)
- **Canonical ID field**: `lidvid` (Logical Identifier + Version ID, e.g., `urn:nasa:pds:orex.ocams::1.0`)
- **Alias fields**: `pds:Identification_Area.pds:alternate_id`, `pds:Alias_List.pds:Alias.pds:alternate_id`
- **Relationship fields**: `ref_lid_instrument`, `ref_lid_investigation`, `ref_lid_target`
- **Update frequency**: As missions deliver data (continuous)
- **Priority**: 1
- **Pagination**: Uses `search_after` parameter (not offset-based) - must follow cursor
- **Agent notes**: PDS uses a hierarchical LID system. A mission like OSIRIS-REx has LID `urn:nasa:pds:context:investigation:mission.orex`. Query Product_Context with `q=` parameter using Solr syntax.

### NASA JPL Small-Body Database (SBDB)

- **Role**: Orbital elements, physical parameters, close-approach data for small bodies
- **Discipline**: Planetary science
- **Base URL**: `https://ssd-api.jpl.nasa.gov/`
- **API type**: REST (JSON)
- **Auth**: None
- **Key endpoints**:
  - `/sbdb.api` - Single object lookup by name/designation/SPK-ID
  - `/sbdb_query.api` - Bulk query with filters (orbit class, size, etc.)
  - `/cad.api` - Close-approach data
- **Record types**: orbital elements, physical parameters (diameter, albedo, rotation period), discovery circumstances
- **Canonical ID field**: `spkid` (SPK-ID, unique integer), cross-ref with `des` (designation)
- **Alias fields**: `fullname`, `des` (primary designation), `name` (if named)
- **Relationship fields**: `orbit_class.code`, `neo` flag, `pha` flag
- **Update frequency**: Daily (orbital solutions updated continuously)
- **Priority**: 1
- **Agent notes**: Use `sbs=1` flag for small-body search mode. Physical parameters may have multiple published values - API returns preferred value. Cross-reference with SsODNet for alias resolution.

---

## Earth Science Sources

### NASA Earthdata / Common Metadata Repository (CMR)

- **Role**: Canonical dataset, variable, and service graph for Earth science
- **Discipline**: Earth science
- **Base URL**: `https://cmr.earthdata.nasa.gov/search/`
- **API type**: REST (JSON, XML, ECHO10, UMM-JSON)
- **Auth**: None for search; Earthdata Login for data access
- **Key endpoints**:
  - `/collections.json` - Dataset collections (primary entity)
  - `/granules.json` - Individual data files within collections
  - `/variables.json` - Measured variables within collections
  - `/services.json` - Data transformation services
- **Record types**: collections (datasets), granules, variables, services, tools
- **Canonical ID field**: `concept-id` (e.g., `C1234567890-PROVIDER`)
- **Alias fields**: `short_name`, `entry_id`, `doi`
- **Relationship fields**: `associations` (collection->variable, collection->service), `platforms[]`, `instruments[]`
- **Update frequency**: Continuous (providers push updates)
- **Priority**: 1
- **Query params**: `keyword=`, `short_name=`, `platform=`, `instrument=`, `science_keywords[]=`
- **Pagination**: `page_size` (max 2000) + `page_num` or `scroll_id` for large result sets
- **Caching**: Aggressive - collection metadata is relatively stable
- **Agent notes**: CMR is the single largest Earth science metadata catalog. Use UMM-JSON format (`Accept: application/vnd.nasa.cmr.umm_results+json`) for richest metadata. Science keywords use GCMD vocabulary - cross-reference with GCMD for hierarchy.

### Global Change Master Directory (GCMD)

- **Role**: Controlled vocabulary normalization for Earth science
- **Discipline**: Earth science
- **Base URL**: `https://gcmd.earthdata.nasa.gov/kms/`
- **API type**: REST (JSON, XML, CSV, RDF/SKOS)
- **Auth**: None
- **Key endpoints**:
  - `/concepts/concept_scheme/sciencekeywords` - 6-level science keyword hierarchy
  - `/concepts/concept_scheme/instruments` - Instrument vocabulary (4 levels)
  - `/concepts/concept_scheme/platforms` - Platform vocabulary (satellites, aircraft, etc.)
  - `/concepts/concept_scheme/projects` - Mission/project vocabulary
  - `/concepts/concept_scheme/providers` - Data center vocabulary
  - `/concepts/concept_scheme/locations` - Geographic location vocabulary
- **Record types**: hierarchical keyword concepts across 14 schemes
- **Canonical ID field**: `uuid` per concept
- **Alias fields**: Concepts have `prefLabel`, `altLabel` (SKOS)
- **Relationship fields**: `broader`, `narrower`, `related` (SKOS hierarchy)
- **Update frequency**: Semi-regular, 3-6 month cycles (current v23.x)
- **Priority**: 1
- **Scale**: ~14,000+ terms across all schemes
- **Also available**: GitHub mirror at `adiwg/gcmd-keywords` (JSON, npm package)
- **Agent notes**: GCMD keywords are the controlled vocabulary used by CMR collections. When a paper mentions an Earth science variable or instrument, normalize via GCMD first, then link to CMR collections that use that keyword. The hierarchy enables both exact and broader matching.

---

## Heliophysics Sources

### Space Physics Data Facility (SPDF) / CDAWeb

- **Role**: Dataset and instrument metadata for space physics
- **Discipline**: Heliophysics
- **Base URL**: `https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/`
- **API type**: REST (JSON, XML)
- **Auth**: None
- **Key endpoints**:
  - `/dataviews/sp_phys/datasets` - List all datasets
  - `/dataviews/sp_phys/datasets/{id}` - Dataset metadata
  - `/dataviews/sp_phys/datasets/{id}/variables` - Variables in a dataset
  - `/dataviews/sp_phys/observatories` - Observatory/mission list
  - `/dataviews/sp_phys/instruments` - Instrument list
  - `/dataviews/sp_phys/instrumenttypes` - Instrument type vocabulary
- **Record types**: datasets, variables, observatories, instruments, instrument types
- **Canonical ID field**: dataset `Id` (e.g., `AC_H2_MFI`)
- **Alias fields**: `Label`, `Notes`
- **Relationship fields**: `ObservatoryId`, `InstrumentId`, `InstrumentType`
- **Update frequency**: As new datasets are added (weekly-monthly)
- **Priority**: 1
- **Agent notes**: CDAWeb dataset IDs follow a convention: `MISSION_LEVEL_INSTRUMENT` (e.g., `AC_H2_MFI` = ACE, Level 2, Magnetic Field Instrument). The observatory/instrument endpoints provide the entity hierarchy. Use `Accept: application/json` header.

### SPASE (Space Physics Archive Search and Extract)

- **Role**: Ontology and canonical resource identifiers for heliophysics
- **Discipline**: Heliophysics
- **Schema URL**: `https://spase-group.org/data/model/`
- **Registry**: `https://hpde.io/`
- **API type**: XML Schema + static registry (browse/download)
- **Auth**: None
- **Key resources**:
  - `NumericalData` - Time-series datasets
  - `DisplayData` - Plots, images, movies
  - `Catalog` - Event lists, feature catalogs
  - `Observatory` - Spacecraft, ground stations
  - `Instrument` - Instrument descriptions
  - `Person` - Researcher identifiers
  - `Repository` - Data archive locations
- **Record types**: 19 resource types with standardized metadata
- **Canonical ID field**: `ResourceID` (SPASE URI, e.g., `spase://NASA/NumericalData/ACE/MAG/L2/PT16S`)
- **Alias fields**: `AlternateName` within resource descriptions
- **Relationship fields**: `InstrumentID`, `ObservatoryID`, `ProviderResourceName`
- **Vocabularies**: ~500+ measurement types, ~200+ observatories
- **Update frequency**: As community contributes (GitHub-based)
- **Priority**: 1
- **Agent notes**: SPASE is both a schema (defining how to describe heliophysics resources) and a registry (hpde.io contains actual resource descriptions). Treat SPASE as the ontology layer for heliophysics - use its vocabulary types and measurement classifications for normalization. Cross-reference SPASE ResourceIDs with SPDF dataset IDs.

---

## Cross-Domain (Optional, High Leverage)

### IVOA (International Virtual Observatory Alliance)

- **Role**: TAP (Table Access Protocol) standard for astronomical databases
- **Discipline**: Cross-domain (primarily astrophysics)
- **TAP standard**: `https://www.ivoa.net/documents/TAP/`
- **Registry**: `https://www.ivoa.net/documents/Registry/`
- **Agent notes**: TAP provides a uniform SQL-like query interface to many astronomical databases. Useful for querying VESPA (planetary) and ESA PSA (planetary) endpoints that implement EPN-TAP.

### VESPA (Virtual European Solar and Planetary Access)

- **Role**: Planetary science data discovery via EPN-TAP
- **Discipline**: Planetary science
- **TAP endpoint**: `https://vespa.obspm.fr/tap`
- **Agent notes**: Discover planetary science data services. Each service exposes an EPN-TAP table queryable via ADQL. Lower priority than PDS but covers European planetary data.

### ESA Planetary Science Archive (PSA)

- **Role**: European planetary mission data
- **Discipline**: Planetary science
- **TAP endpoint**: `https://archives.esac.esa.int/psa/#/tap`
- **Agent notes**: Covers ESA missions (Rosetta, Mars Express, BepiColombo). TAP interface for programmatic access. Supplements PDS for European missions.

---

## Ingestion Order

1. **SciX literature** -> `documents` (already done - papers table)
2. **SsODNet** -> entities + aliases (small bodies)
3. **PDS** -> missions, instruments, datasets (planetary)
4. **SBDB** -> physical/orbital enrichment (planetary)
5. **CMR** -> datasets + variables (Earth science)
6. **GCMD** -> controlled vocabulary normalization (Earth science)
7. **SPASE** -> heliophysics ontology + entities
8. **SPDF** -> dataset metadata (heliophysics)
9. **Linking passes** - document-entity, document-dataset, entity-entity

## API Preferences

- **Prefer REST**: CMR, SsODNet, SBDB, SPDF, PDS
- **Support TAP**: VESPA, PSA (secondary)
- **Avoid**: Scraping portals, non-programmatic archives
- **Cache aggressively**: SsODNet (small-body identity is stable), CMR collections, GCMD vocabularies

## Constraints

- Do NOT rely on embeddings for primary linking (structural/ID-based first)
- Do NOT flatten all entities into one type (preserve domain-specific typing)
- Do NOT lose source provenance (every record traces to source + sync run)
- Do NOT ingest full granule-level data initially (collection/bundle level first)
