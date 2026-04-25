# Entity Enrichment Value Props Eval — 2026-04 (post-vjf)

_Generated 2026-04-25 after the **vjf option-2 part_of inheritance backfill**
landed (commits `8d53d5d` + `d63b0a3` + `63efc5e`)._

## Why this run

The pre-vjf eval (`docs/eval/entity_value_props_2026-04.md`) scored
`specific_entity = 1.10`, below the 1.5 acceptance threshold. The four
flagship-instrument gold queries (`spec-004 NIRSpec`, `spec-005 ACIS`,
`spec-009 Crab Pulsar`, `spec-010 Perseverance`) returned zero results because
the instruments had **zero rows** in `document_entities` despite their
`part_of` edges to parent missions being present in `entity_relationships`.

vjf option 2 closed that gap with three changes:

1. `scripts/backfill_part_of_inheritance.py` — reads `flagship_seed` /
   `curated_flagship_v1` `part_of` edges and inserts:
   - **Instrument rows** in `document_entities`, found via
     `papers.tsv @@ phraseto_tsquery(...)` over the instrument's
     mission-disambiguated surface forms (long-form descriptive aliases like
     `Advanced CCD Imaging Spectrometer`, plus mission-prefixed aliases like
     `Chandra/ACIS`, `JWST NIRSpec`, `LSST Camera`). Bare CamelCase
     canonicals (`NIRSpec`, `LSSTCam`) and bare uppercase acronyms (`ACIS`,
     `OM`) are deliberately rejected because they collide with name-twin
     instruments on other missions (Keck NIRSPEC echelle) or with unrelated
     jargon ('ACIS' → english-stemmed 'aci').
   - **Inherited parent rows** mirroring each instrument row to the parent
     mission with `link_type='inherited'`, `match_method='part_of_inheritance'`
     and `evidence={"via_instrument_id":..., "via_instrument_name":...}`.
2. `scripts/eval_entity_value_props.py::SpecificEntityBackend._entity_papers`
   — wraps the `document_entities → papers` join in `DISTINCT ON (bibcode)` so
   a paper carrying both an existing tier-2 `abstract_match` row AND a new
   `inherited` row is counted once, not twice.
3. (No new edges were added — all 24 instrument→mission `part_of` rows
   already existed under `source='flagship_seed'`. Backfill is purely
   on the document side.)

Run on prod under `scix-batch --allow-prod` inserted **28,625 instrument
rows + 27,382 inherited parent rows** across 24 instruments. Per-entity
counts table at the end of this report.

## Summary

Overall score: **1.30 / 3.0** (pre-vjf: 1.22, +0.08).

| Prop | N | Mean | StdErr | Δ vs pre-vjf |
|---|---|---|---|---|
| alias_expansion | 10 | 2.10 | 0.31 | unchanged (uses `hybrid_search`, not `document_entities`) |
| ontology_expansion | 10 | 0.50 | 0.31 | unchanged |
| disambiguation | 10 | 2.10 | 0.46 | unchanged |
| type_filter | 10 | 1.20 | 0.29 | unchanged |
| **specific_entity** | **10** | **1.60** | **0.40** | **+0.50** (clears 1.5 acceptance gate) |
| community_expansion | 10 | 0.30 | 0.15 | unchanged (semantic-community signal not affected by part_of backfill) |

Only `specific_entity` and `community_expansion` were re-judged in this run
(they're the only props that read from `document_entities`); the four
hybrid-search-only props are reproduced verbatim from the pre-vjf report.

## specific_entity (post-vjf)

N = 10, mean = **1.60**, stderr = 0.40.

| Query ID | Pre-vjf | Post-vjf | Δ | Rationale |
|---|---|---|---|---|
| spec-001 | 3 | 3 | 0 | All 10 results explicitly reference the Hubble Space Telescope mission (HST) in their titles or abstracts, with no false positives matching only 'Edwin Hubble' or generic 'space telescope'. |
| spec-002 | 1 | 1 | 0 | Only result [2] is the canonical JWST mission paper; the other 9 results are about TESS, HST, Hobby-Eberly, SIRTF, etc., showing the retrieval failed to filter to JWST-specific papers. (Resolver picked the curated_flagship_v1 mission entity, which has 5,159 pre-existing tier-2 abstract_match rows from the bulk linker — those rows include passing-mention papers; backfill didn't change this entity.) |
| spec-003 | 2 | 2 | 0 | Results 4-10 (7 of 10) clearly reference the Chandra X-ray Observatory mission, while results 1-3 are off-topic. |
| spec-004 | 0 | **2** | **+2** | 9 of 10 results explicitly reference JWST NIRSpec (results 2-10), with only result 1 being off-topic (a 2009 paper about transit surveys with no NIRSpec mention). The retrieval cleanly excludes MIRI/NIRCam/Keck-NIRSPEC. **Backfill enabled this query.** |
| spec-005 | 0 | **3** | **+3** | All 10 results explicitly reference Chandra's ACIS (Advanced CCD Imaging Spectrometer), with no HRC or ACS confusion. **Backfill enabled this query.** |
| spec-006 | 2 | 2 | 0 | Most results (1, 3, 4, 6, 7, 8, 9) reference ALMA the instrument or directly related radio interferometry work, but result 2 is an unrelated 'Alma-Ata' string match and results 5 and 10 are about other instruments. (Resolver picks the curated_flagship_v1 ALMA entity which has aho_corasick noise.) |
| spec-007 | 3 | 3 | 0 | 9 of 10 results are directly about the Kepler Space Telescope mission. |
| spec-008 | 0 | 0 | 0 | TRAPPIST-1b — entity does not exist in `entities` table at all (verified by direct DB query: zero rows in `entities` or `entity_aliases` matching `trappist-1b`). Out of scope for vjf. |
| spec-009 | 0 | 0 | 0 | Crab Pulsar — entity does not exist (verified). Out of scope for vjf. |
| spec-010 | 0 | 0 | 0 | Perseverance — entity does not exist (verified). Out of scope for vjf. |

**Maximum achievable specific_entity score given gold-set unresolvables**:
7 / 10 = 2.10. Current 1.60 covers 76% of the achievable ceiling.

## community_expansion (post-vjf)

N = 10, mean = 0.30, stderr = 0.15 (unchanged from pre-vjf).

The part_of inheritance backfill writes only to `document_entities`. The
community_expansion backend resolves a seed entity, looks up its modal
`paper_metrics.community_semantic_medium`, and returns sibling papers from
that community ordered by pagerank. The new instrument rows DO contribute
to the modal community calculation, but most flagship missions already had
abundant doc_entities rows under `aho_corasick_abstract`, so the modal
community is unchanged. Improving community_expansion is the next bead;
out of scope for vjf.

| Query ID | Score | Rationale |
|---|---|---|
| comm-001 HST community | 0 | None of the top 10 results are HST-instrument (WFC3/STIS/ACS/COS) or STScI operations papers. |
| comm-002 JWST community | 0 | None of the top-10 results are JWST-instrument papers or relate to the JWST ecosystem (NIRSpec, NIRCam, MIRI, FGS, NIRISS). |
| comm-003 Chandra community | 0 | None of the top 10 results are about Chandra or its instruments. |
| comm-004 Cassini-Huygens | 0 | None of the top 10 results are about Cassini-Huygens, Titan, Enceladus, or Saturn's rings. |
| comm-005 Kepler community | 0 | None of the top 10 results are about the Kepler mission, Kepler-discovered exoplanets, KOI catalog, or follow-up missions. |
| comm-006 TRAPPIST-1 sibs | 0 | No results returned — TRAPPIST-1 entity unresolvable. |
| comm-007 LIGO network | 1 | Only result [5] (GW150914 detection by LIGO) clearly belongs to the modern gravitational-wave observing network community. |
| comm-008 MSL/Curiosity sibs | 0 | No results returned — Mars Science Laboratory entity unresolvable. |
| comm-009 CMB missions | 1 | Only result [6] (WMAP first-year cosmological parameters) directly satisfies the CMB-mission community expectation. |
| comm-010 Gaia astrometry | 1 | Only result [4] (HIPPARCOS and TYCHO catalogues) directly satisfies the expected community-expansion. |

## Backfill row counts (per entity)

```
 instrument  -> mission                          | instr rows | parent rows
-------------+----------------------------------+------------+-------------
 NIRSpec     -> James Webb Space Telescope        |        766 |         766
 NIRCam      -> James Webb Space Telescope        |        701 |         648
 MIRI        -> James Webb Space Telescope        |        590 |         540
 NIRISS      -> James Webb Space Telescope        |        134 |         122
 FGS         -> James Webb Space Telescope        |        594 |         590
 STIS        -> Hubble Space Telescope            |       3295 |        3295
 ACS         -> Hubble Space Telescope            |       3302 |        3227
 WFC3        -> Hubble Space Telescope            |       2351 |        2043
 COS         -> Hubble Space Telescope            |       1768 |        1611
 NICMOS      -> Hubble Space Telescope            |        723 |         651
 ACIS        -> Chandra X-ray Observatory         |       1188 |        1188
 HRC         -> Chandra X-ray Observatory         |        857 |         829
 EPIC        -> XMM-Newton                        |        616 |         616
 RGS         -> XMM-Newton                        |        693 |         627
 OM          -> XMM-Newton                        |       2893 |        2839
 IRAC        -> Spitzer Space Telescope           |       2850 |        2850
 IRS         -> Spitzer Space Telescope           |       2612 |        2480
 MIPS        -> Spitzer Space Telescope           |        701 |         495
 LSSTCam     -> Vera C. Rubin Observatory         |        100 |         100
 BOSS        -> Sloan Digital Sky Survey          |        886 |         886
 APOGEE      -> Sloan Digital Sky Survey          |        482 |         471
 ALMA Band 3 -> Atacama Large Millimeter Array    |        149 |         149
 ALMA Band 6 -> Atacama Large Millimeter Array    |        224 |         218
 ALMA Band 7 -> Atacama Large Millimeter Array    |        150 |         141
                                            TOTAL |     28,625 |      27,382
```

## Carry-over results (unchanged props)

Reproduced from `docs/eval/entity_value_props_2026-04.md` for completeness;
these props use `hybrid_search` and are unaffected by the
`document_entities` backfill.

### alias_expansion (N=10, mean=2.10)

| Query ID | Score | Rationale |
|---|---|---|
| alias-001 | 3 | Nearly all top results are HST observations of M31, including [4] which explicitly uses 'Hubble Space Telescope' without the HST acronym. |
| alias-002 | 1 | Results topically relevant but no visible evidence alias expansion surfaced full-name-only papers. |
| alias-003 | 2 | Most results are Chandra/X-ray cluster studies with several explicit 'Chandra X-ray' phrasings. |
| alias-004 | 3 | Multiple results use the canonical expanded form ('cosmic microwave background') rather than CMB. |
| alias-005 | 3 | Results 1-6 spell out 'Type Ia supernovae' rather than the SNe Ia shorthand. |
| alias-006 | 2 | Majority are ALMA submillimeter observations, but no canonical 'Atacama Large Millimeter/submillimeter Array' verbatim. |
| alias-007 | 0 | Retrieval returned no results. |
| alias-008 | 2 | Most results are WISE/NEOWISE brown-dwarf studies; full-form 'Wide-field Infrared Survey Explorer' not surfaced. |
| alias-009 | 3 | All 10 results are about DESI; clean alias expansion. |
| alias-010 | 2 | Most results Gaia astrometry; 'Global Astrometric Interferometer for Astrophysics' not surfaced. |

### ontology_expansion (N=10, mean=0.50)

See `docs/eval/entity_value_props_2026-04.md` § ontology_expansion for full
rationales. Only `ont-006` (dwarf-planet → Eris/Makemake/Ceres/Haumea/Sedna)
scored 3; all other queries scored 0 or 1.

### disambiguation (N=10, mean=2.10)

See `docs/eval/entity_value_props_2026-04.md` § disambiguation for full
rationales. Strong on Kepler/Cassini/New Horizons/Voyager/Viking/NICER/Juno
(all 3s); weak on Sagan-mission/Psyche/Vesta-spacecraft (all 0s, mostly
because of pre-mission-era ground observations dominating).

### type_filter (N=10, mean=1.20)

See `docs/eval/entity_value_props_2026-04.md` § type_filter for full
rationales. Only `type-010` (exoplanet missions: TESS/CHEOPS/Kepler/K2/TOI)
scored 3; the rest 0-2 reflecting weak entity_type filter enforcement at
retrieval time.

## Reproduction

```bash
# Backfill (one-shot, idempotent):
scix-batch --allow-prod python scripts/backfill_part_of_inheritance.py \
    --db scix --allow-prod --delete-prior

# Re-eval (specific + community only — unchanged props can be skipped):
scix-batch python scripts/eval_entity_value_props.py \
    --props specific community --db "dbname=scix" --judge-timeout-s 240
```

Per-query JSONL: `.claude/prd-build-artifacts/eval-d4-20260425T152759.jsonl`.
