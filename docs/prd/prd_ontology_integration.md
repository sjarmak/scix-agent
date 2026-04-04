# PRD: Scientific Ontology Integration for Entity Normalization

## Problem Statement

The SciX entity extraction pipeline needs structured vocabularies for dictionary matching, alias resolution, and normalization across 4 entity types (instruments, datasets, software, methods). The UAT (already loaded, 2,275 concepts) covers only topics — not the entity types we extract. Divergent research across 5 domains identified ~48,000 dictionary entries available from freely accessible registries, downloadable in 2-3 days of integration work.

## Goals & Non-Goals

### Goals

- Build a composite seed dictionary covering all 4 entity types from authoritative registries
- Enable dictionary-based entity matching as the primary extraction method for closed-set types
- Provide alias resolution (e.g., "HST" = "Hubble Space Telescope" = Wikidata Q2513)
- Establish a maintainable update pipeline for vocabulary freshness

### Non-Goals

- Building our own ontology from scratch (use existing registries)
- Complete coverage of all possible entity mentions (dictionary is the foundation, NER fills gaps)
- Integrating every registry discovered (prioritize by coverage-per-effort)

## Requirements

### Must-Have

- **ASCL software dictionary**
  - Download `ascl.net/code/json`, parse into canonical_name + aliases + bibcode
  - Acceptance: `SELECT count(*) FROM dictionary_software` returns > 3,500; lookup("astropy") returns ASCL ID and ADS bibcode

- **ADS `data` field vocabulary extraction**
  - Extract all unique data source labels from the 32M paper corpus
  - Acceptance: Produces a deduplicated list of 150+ archive-level dataset names with paper counts

- **PhySH Techniques dictionary**
  - Download via REST API or GitHub; parse SKOS hierarchy into flat + hierarchical tables
  - Acceptance: `SELECT count(*) FROM dictionary_methods WHERE source='physh'` returns > 200; lookup("Monte Carlo") returns parent concept and child variants

- **AAS Facility Keywords dictionary**
  - Scrape ~690 entries with wavelength regime and location metadata
  - Acceptance: `SELECT count(*) FROM dictionary_instruments WHERE source='aas'` returns > 600; lookup("HST") returns facility record with wavelength flags

### Should-Have

- **Wikidata alias enrichment for instruments**
  - SPARQL queries to harvest aliases, instrument sub-components, and cross-identifiers for each AAS facility
  - Acceptance: Average alias count per facility > 3; "Hubble Space Telescope" entry has aliases including "HST", "Hubble"

- **Papers With Code methods dictionary**
  - Download ML methods JSON from GitHub
  - Acceptance: `SELECT count(*) FROM dictionary_methods WHERE source='pwc'` returns > 1,000

- **AstroMLab 5 concept vocabulary**
  - Download 9,999 concepts with embeddings from GitHub
  - Acceptance: Concepts loadable; "Instrumental Design" category contains > 1,000 entries

- **VizieR catalog list via TAP**
  - Query TAPVizieR for full catalog metadata
  - Acceptance: > 25,000 catalog entries with identifiers and titles

### Nice-to-Have

- **Softalias-KG SPARQL integration** for software alias disambiguation
- **SIMBAD/NED object type hierarchies** as negative dictionaries (filter celestial objects from dataset mentions)
- **OntoPortal-Astro facility vocabulary** when it stabilizes (IVOA community effort)
- **NASA STI Thesaurus** (18,400 terms) for supplementary method/technique coverage

## Design Considerations

### Dictionary Schema

```sql
CREATE TABLE entity_dictionary (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- instrument, dataset, software, method
    source TEXT NOT NULL,       -- ascl, aas, physh, pwc, vizier, ads_data, wikidata, astromlb
    external_id TEXT,           -- ASCL ID, Wikidata QID, VizieR catalog ID, etc.
    aliases TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}', -- wavelength, hierarchy, bibcode, etc.
    UNIQUE(canonical_name, entity_type, source)
);
CREATE INDEX idx_dict_type ON entity_dictionary(entity_type);
CREATE INDEX idx_dict_aliases ON entity_dictionary USING GIN(aliases);
```

### Methods: Accept Semi-Open-Set Nature

The DEAL shared task (ADS team) deliberately excluded methods as an NER entity type. PhySH provides ~209 structured terms. Papers With Code adds ~1,064 ML methods. Together they cover the "head" of methods vocabulary. The long tail requires LLM extraction + corpus-driven discovery (BERTopic/c-TF-IDF on astro-ph.IM). The dictionary serves as a normalization target, not a completeness guarantee.

### Update Frequency

| Source                | Update Cadence | Method                |
| --------------------- | -------------- | --------------------- |
| ASCL                  | Quarterly      | Re-download JSON      |
| PhySH                 | Annual         | Pull GitHub           |
| Papers With Code      | Monthly        | Pull GitHub JSON      |
| AAS Facility Keywords | Annual         | Re-scrape             |
| Wikidata              | Quarterly      | Re-run SPARQL         |
| VizieR                | Annual         | Re-query TAP          |
| ADS `data` field      | Per-ingest     | Extract from new data |

## Open Questions

- What is the actual overlap between ADS `facility[]` field and AAS Facility Keywords?
- How many of the 27K VizieR catalogs appear in paper text? (likely 2-5K effective vocabulary)
- Is the AstroMLab concept quality sufficient for direct use, or does it need manual curation?
- Should methods be decomposed into sub-types (family → algorithm → implementation)?
- When will OntoPortal-Astro publish production-ready facility vocabulary?

## Research Provenance

### Divergent Research (5 independent agents)

1. **Instruments**: AAS Facility Keywords (~690), Wikidata SPARQL (~5K w/aliases), OntoPortal-Astro (2025), IAU-MPC observatory codes (~2,693)
2. **Software**: ASCL (~3,958, JSON API), ADS doctype:software (50-100K), JOSS (~260 astro), Softalias-KG SPARQL
3. **Datasets**: ADS `data` field (~200 archive-level), VizieR (~27K catalogs via TAP), HEASARC (~940 tables), MAST (~135 missions)
4. **Knowledge Graphs**: AstroMLab 5 (9,999 concepts w/embeddings), OpenAlex (4,500 topics), SemOpenAlex (26B triples), IVOA UCD, Wikidata P4466 (UAT linkage)
5. **Methods**: PhySH Techniques (~209, REST API + SKOS), Papers With Code (~1,064 ML methods), NASA STI Thesaurus (18,400 terms), MSC 2020

### Key Convergence

All agents agreed ADS metadata is the best starting point. Methods are confirmed as semi-open-set by multiple sources including the ADS team's own DEAL shared task exclusion.

### Key Surprise

AstroMLab 5 provides 9,999 concepts with embeddings — 4x denser than UAT and directly applicable to our entity types, yet was unknown before this research round.
