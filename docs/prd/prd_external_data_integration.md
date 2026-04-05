# PRD: External Data Integration for Cross-Domain Entity Resolution

## Problem Statement

The SciX literature database contains ~5M papers across astrophysics, earth science, heliophysics, and planetary science, but entity knowledge is siloed: the `entity_dictionary` table stores ~55K vocabulary terms with no join path to papers, no multi-scheme external IDs, no entity-entity relationships, and no provenance tracking. The `extractions` table stores NER-extracted mentions as raw JSONB with no link to canonical dictionary entries. A paper mentioning "Bennu", "OSIRIS-REx", and "spectral reflectance data" cannot currently be resolved to canonical entities, missions, or datasets -- an agent must make 4-5 tool calls and manually stitch results.

Seven authoritative external APIs (SsODNet, PDS, SBDB, CMR, GCMD, SPDF, SPASE) provide structured, canonical metadata that can transform the database into an agent-navigable knowledge layer. The existing harvesters (GCMD, SPASE, PDS4, PhySH) prove the pattern works but load into a flat dictionary table that lacks the relational infrastructure for cross-domain linking.

## Goals & Non-Goals

### Goals

- Extend PostgreSQL schema with entities, aliases, external IDs, datasets, relationships, and provenance tables
- Build ingestion pipelines for all 7 priority sources with shared HTTP client infrastructure
- Create entity resolution layer: exact canonical match -> alias lookup -> fuzzy fallback
- Pre-compute document-entity and document-dataset links with evidence and confidence
- Provide agent-optimized materialized views for single-query full-context retrieval (<100ms)
- Implement incremental sync with harvest_runs tracking and resumable cursors

### Non-Goals

- Building a unified ontology from scratch (use existing authoritative registries)
- Embedding-based primary linking (NASA SMD KG found cross-domain embedding linking produces unacceptable false positives)
- Full granule-level data ingestion (collection/bundle level first)
- Real-time synchronization (batch harvesting on configurable cadence)
- Replacing the existing entity_dictionary (maintain compatibility via view during transition)

## Research & Convergence Provenance

### Phase 1: Divergent Research (5 independent agents)

1. **Prior Art & Industry Patterns** -- analyzed OpenAlex, Semantic Scholar, Wikidata, NASA SMD KG, OpenSanctions
2. **First-Principles Schema Design** -- analyzed all 15 existing migrations, proposed relational schema extension
3. **API Integration Realities** -- tested/documented all 7 APIs, identified pagination quirks and bulk alternatives
4. **Agent-Facing Query Design** -- analyzed MCP server, proposed bridge tables and composite views
5. **Failure Modes & Operational Risks** -- identified migration collisions, missing provenance, bulk_load limitations

### Phase 2: Structured Convergence (3-position debate, 2 rounds)

Three positions debated the PRD's key tensions:

- **Pragmatist (Incrementalist)**: Extend entity_dictionary, load only mentioned objects, ship MVP in 2-3 weeks
- **Architect (Clean Schema)**: New entities table, load full ssoBFT, all 4 phases as Must-Have
- **Operator (Risk Minimizer)**: Infrastructure first as hard gate, staging mandatory for large loads

### Resolved Decisions

**D1: New `entities` table, not entity_dictionary extension (UNANIMOUS after Round 2)**

- Pragmatist conceded: TEXT[] aliases cannot be JOINed, single external_id cannot hold multi-scheme IDs, `lookup()` returns `LIMIT 1` with no ranking. These are structural limitations unfixable by adding columns.
- entity_dictionary remains untouched; 11 existing harvesters continue working unmodified.
- Compatibility view (`entity_dictionary_compat`) provides backward compatibility during transition.
- Seed migration: `INSERT INTO entities SELECT ... FROM entity_dictionary` gives 55K entities on day one.

**D2: Phase 0 is a hard gate (UNANIMOUS)**

- The 013 migration numbering collision (two files with same prefix) is a live bug proving infrastructure isn't ready.
- `schema_migrations` table + proper migration runner must land before any new tables.
- `harvest_runs` table must exist before any new harvester writes data.
- No new migration is applied until Phase 0 is verified.

**D3: Phased delivery in 3 milestones (CONSENSUS)**

- Architect moved from "all 4 phases as one unit" to 3-milestone plan.
- Pragmatist accepted clean schema design in exchange for reduced harvester scope.
- Operator accepted that not all sources need staging (vocabulary-scale loads write directly to public tables).

**D4: Staging schema is Must-Have for loads >100K rows (CONSENSUS)**

- Promoted from Should-Have. SsODNet (1.2M rows) must use staging pattern from migration 015.
- Vocabulary-scale sources (GCMD ~15K, SPASE ~200, PDS4 ~500) write directly to public tables.

**D5: Shared HTTP client for new harvesters only initially (MAJORITY)**

- Build the client module in Phase 0. New harvesters (SsODNet, CMR, SBDB, SPDF) must use it.
- Existing harvesters (GCMD, SPASE, PDS4) refactored incrementally as they're updated, not blocked.
- Operator's dissent: would prefer all harvesters on shared client before any new data flows. Noted but overruled -- existing harvesters work and refactoring them is separate from new functionality.

### Unresolved Tension (to revisit with data)

**SsODNet scope: full ssoBFT (1.2M) vs. mention-seeded subset (~5K-50K)**

- Architect + Operator favor full load: eliminates circular dependency (need names to resolve, but resolution requires names), enables discovery queries, is a single download not 50K API calls.
- Pragmatist favors seeded subset: 95%+ of 1.2M objects may never appear in any paper, 2.1GB download + staging pipeline is a project in itself.
- **Resolution**: Defer to data. Run `SELECT DISTINCT` on extraction mentions to measure the seed set size. If <10K unique small-body names, start with API-based seed. If >10K, the ssoBFT download is more practical than 10K sequential API calls. Either way, the schema supports both approaches -- this is a harvester implementation decision, not a schema decision.

### Debate Highlights

- **Pragmatist's strongest contribution**: Identifying that the PRD is 3-4 PRDs in one, and that Phase 1 (DDL) is not the hard part -- it's just SQL. The real work is harvesters and linking.
- **Architect's strongest contribution**: Proving entity_dictionary is structurally incapable of knowledge-graph operations by tracing the actual `lookup()` code path through `ANY(aliases) LIMIT 1`.
- **Operator's strongest contribution**: Concrete failure scenario for SsODNet partial load without staging (800K orphaned rows, no rollback path), which promoted staging from Should-Have to Must-Have.

## Requirements

### Must-Have (Phase 0: Infrastructure)

- **Migration tracker**
  - Add `schema_migrations` table: `(version INT PK, applied_at TIMESTAMPTZ, filename TEXT)`
  - Update `setup_db.sh` to check before applying each migration
  - Fix existing 013 numbering collision
  - Acceptance: `SELECT count(*) FROM schema_migrations` returns count equal to number of applied migrations; re-running `setup_db.sh` is idempotent with no errors

- **Shared HTTP client module** (`src/scix/http_client.py`)
  - Configurable retry count with exponential backoff
  - Per-host rate limiting (critical for SBDB's 1-concurrent-request limit)
  - Circuit breaker (fail-open after N consecutive failures)
  - Response caching to disk for restart capability
  - User-Agent header
  - Acceptance: `harvest_gcmd.py` and `harvest_spase.py` refactored to use shared client; retry test with mock 503 shows exponential backoff; SBDB sequential enforcement test passes

- **Harvest runs table** (`harvest_runs`)
  - Schema: `id SERIAL PK, source TEXT NOT NULL, started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, status TEXT, records_fetched INT, records_upserted INT, cursor_state JSONB, error_message TEXT, config JSONB`
  - All existing harvesters updated to log runs
  - Acceptance: After running any harvester, `SELECT * FROM harvest_runs WHERE source='gcmd'` returns row with status='completed' and accurate counts

### Must-Have (Phase 1: Schema Extension)

- **Entities table** (`entities`)
  - Schema: `id SERIAL PK, canonical_name TEXT NOT NULL, entity_type TEXT NOT NULL, discipline TEXT, source TEXT NOT NULL, harvest_run_id INT REFERENCES harvest_runs(id), properties JSONB DEFAULT '{}', created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()`
  - UNIQUE on `(canonical_name, entity_type, source)`
  - Indexes on `entity_type`, `discipline`, `lower(canonical_name)`
  - GIN index on `properties` with `jsonb_path_ops`
  - Acceptance: `\d entities` shows all columns, constraints, and indexes as specified

- **Entity identifiers table** (`entity_identifiers`)
  - Schema: `entity_id INT REFERENCES entities(id), id_scheme TEXT NOT NULL, external_id TEXT NOT NULL, is_primary BOOLEAN DEFAULT false`
  - PK on `(id_scheme, external_id)` -- enforces uniqueness per scheme
  - Index on `entity_id` for reverse lookup
  - Acceptance: A single entity can hold Wikidata QID + PDS URN + GCMD UUID simultaneously; `SELECT * FROM entity_identifiers WHERE id_scheme='wikidata' AND external_id='Q11091' ` returns the Bennu entity

- **Entity aliases table** (`entity_aliases`)
  - Schema: `entity_id INT REFERENCES entities(id), alias TEXT NOT NULL, alias_source TEXT`
  - PK on `(entity_id, alias)`
  - Functional index on `lower(alias)` for case-insensitive lookup
  - Acceptance: `SELECT e.canonical_name FROM entities e JOIN entity_aliases ea ON ea.entity_id = e.id WHERE lower(ea.alias) = '1999 rq36'` returns 'Bennu' in <10ms

- **Entity relationships table** (`entity_relationships`)
  - Schema: `id SERIAL PK, subject_entity_id INT REFERENCES entities(id), predicate TEXT NOT NULL, object_entity_id INT REFERENCES entities(id), source TEXT, harvest_run_id INT REFERENCES harvest_runs(id), confidence REAL DEFAULT 1.0`
  - UNIQUE on `(subject_entity_id, predicate, object_entity_id)`
  - Index on `object_entity_id` for reverse traversal
  - Predicate vocabulary: `instruments`, `observes_target`, `part_of_mission`, `broader_than`, `narrower_than`, `same_as`, `derived_from`
  - Acceptance: Query "what instruments are part of OSIRIS-REx mission" returns OCAMS, OLA, OTES, OVIRS, REXIS via single JOIN

- **Datasets table** (`datasets`)
  - Schema: `id SERIAL PK, name TEXT NOT NULL, discipline TEXT, source TEXT NOT NULL, canonical_id TEXT NOT NULL, description TEXT, temporal_start DATE, temporal_end DATE, properties JSONB DEFAULT '{}', harvest_run_id INT REFERENCES harvest_runs(id), created_at TIMESTAMPTZ DEFAULT NOW()`
  - UNIQUE on `(source, canonical_id)`
  - Acceptance: PDS bundles, CMR collections, and SPDF datasets all stored with their native IDs

- **Dataset-entity links** (`dataset_entities`)
  - Schema: `dataset_id INT REFERENCES datasets(id), entity_id INT REFERENCES entities(id), relationship TEXT NOT NULL`
  - PK on `(dataset_id, entity_id, relationship)`
  - Acceptance: OSIRIS-REx OCAMS dataset linked to both instrument entity and Bennu target entity

- **Document-entity links** (`document_entities`)
  - Schema: `bibcode TEXT NOT NULL, entity_id INT REFERENCES entities(id), link_type TEXT NOT NULL, confidence REAL, match_method TEXT, evidence JSONB, harvest_run_id INT REFERENCES harvest_runs(id)`
  - PK on `(bibcode, entity_id, link_type)`
  - No FK on bibcode (matches citation_edges pattern -- entities may reference papers not yet ingested)
  - link_type vocabulary: `mentions`, `uses_instrument`, `observes_target`, `analyzes_dataset`, `uses_data_from`, `derived_from`
  - Acceptance: After linking pass, `SELECT count(*) FROM document_entities WHERE bibcode='2019Natur.568...55L'` returns >0 with entity links to Bennu and OSIRIS-REx

- **Document-dataset links** (`document_datasets`)
  - Schema: `bibcode TEXT NOT NULL, dataset_id INT REFERENCES datasets(id), link_type TEXT NOT NULL, confidence REAL, match_method TEXT, harvest_run_id INT REFERENCES harvest_runs(id)`
  - PK on `(bibcode, dataset_id, link_type)`
  - Acceptance: Papers citing PDS datasets linked with provenance

- **Entity dictionary compatibility view**
  - `CREATE VIEW entity_dictionary_compat AS SELECT ... FROM entities JOIN entity_aliases ...` that exposes the same column shape as the existing entity_dictionary for backward compatibility
  - Acceptance: Existing `dictionary.lookup()` works against the view without code changes

### Must-Have (Phase 2: Ingestion Pipelines)

- **SsODNet harvester** (`scripts/harvest_ssodnet.py`)
  - Download ssoBFT Parquet file (2.1GB, ~1.2M objects)
  - Parse into entities (canonical_name from IAU-preferred designation) + entity_identifiers (ssodnet name, SPK-ID) + entity_aliases (all known designations)
  - Store physical properties (diameter, albedo, taxonomy) in properties JSONB
  - Log harvest_run
  - Acceptance: `SELECT count(*) FROM entities WHERE source='ssodnet'` returns >1,000,000; `SELECT * FROM entity_aliases WHERE lower(alias)='1999 rq36'` resolves to Bennu

- **PDS context harvester** (update existing `scripts/harvest_pds4.py`)
  - Harvest Investigation (missions), Instrument, Target context products
  - Store as entities with PDS URN in entity_identifiers
  - Create entity_relationships: instrument -> part_of_mission, mission -> observes_target
  - Acceptance: >30 missions, >100 instruments, >50 targets; OSIRIS-REx has relationships to its 5 instruments and Bennu target

- **SBDB enrichment** (`scripts/harvest_sbdb.py`)
  - Query SBDB for objects already in entities table (from SsODNet) to add orbital elements
  - Sequential requests only (1 concurrent per IP)
  - Update entities.properties with orbital class, NEO/PHA flags, discovery info
  - Acceptance: NEO-flagged entities have `properties->>'neo' = 'true'`; no concurrent request errors

- **CMR collection harvester** (`scripts/harvest_cmr.py`)
  - Paginate all NASA collections using CMR-Search-After headers (UMM-JSON format)
  - Store as datasets with CMR concept-id as canonical_id
  - Extract instruments, platforms, science_keywords into entity_identifiers (cross-ref GCMD)
  - Acceptance: >8,000 NASA collections harvested; `SELECT count(*) FROM datasets WHERE source='cmr'` returns >8,000

- **GCMD harvester update** (update existing `scripts/harvest_gcmd.py`)
  - Refactor to use shared HTTP client
  - Write to new entities table (in addition to or instead of entity_dictionary)
  - Store GCMD UUID in entity_identifiers
  - Acceptance: All existing GCMD entities have corresponding rows in entities + entity_identifiers tables

- **SPASE harvester update** (update existing `scripts/harvest_spase.py`)
  - Update to SPASE v2.7.1 (from pinned v2.7.0)
  - Write to new entities table
  - Acceptance: New v2.7.1 enumeration values (Amplitude, Magnitude, etc.) present in entities table

- **SPDF/CDAWeb harvester** (`scripts/harvest_spdf.py`)
  - Single REST call to `/dataviews/sp_phys/datasets` for all datasets
  - Harvest observatories and instruments from dedicated endpoints
  - Store datasets with CDAWeb ID; cross-reference SPASE ResourceIDs in entity_identifiers
  - Create entity_relationships: dataset -> from_instrument, instrument -> at_observatory
  - Acceptance: >2,000 datasets; observatory-instrument-dataset hierarchy navigable via entity_relationships

### Must-Have (Phase 3: Resolution & Linking)

- **Entity resolver** (`src/scix/entity_resolver.py`)
  - Resolution order: (1) exact canonical match, (2) alias lookup via entity_aliases, (3) entity_identifiers lookup, (4) discipline-aware ranking when multiple matches, (5) optional fuzzy fallback via pg_trgm
  - Returns all candidates with confidence scores (not LIMIT 1)
  - Discipline context accepted as optional parameter for disambiguation
  - Acceptance: `resolve("1999 RQ36")` returns Bennu with confidence 1.0; `resolve("Mars")` returns multiple candidates ranked by discipline; `resolve("MAHLI")` returns Mars Hand Lens Imager via alias

- **Document-entity linking pipeline** (`src/scix/link_entities.py`)
  - Batch job: for each paper's extracted mentions (from extractions table), resolve against entities using entity_resolver
  - Write results to document_entities with match_method, confidence, and evidence (text span, section)
  - Chunked commits (not single transaction for millions of rows)
  - Resumable from last processed bibcode
  - Acceptance: After running on test corpus of 1000 papers, document_entities has rows with varied match_methods and confidence scores; can resume after interruption without duplicates

- **Document-dataset linking pipeline** (`src/scix/link_datasets.py`)
  - Match dataset references in papers (ADS `data` field, citations to data DOIs, explicit mentions)
  - Write results to document_datasets
  - Acceptance: Papers with ADS `data` field entries linked to corresponding CMR/PDS/SPDF datasets

### Must-Have (Phase 4: Agent Views & MCP)

- **Agent document context view** (materialized)
  - Joins papers + document_entities + entities + entity_identifiers + entity_aliases
  - Single query by bibcode returns: paper metadata, all linked entities with canonical names, aliases, external IDs, types, disciplines
  - `UNIQUE INDEX` on bibcode for `REFRESH MATERIALIZED VIEW CONCURRENTLY`
  - Acceptance: `SELECT * FROM agent_document_context WHERE bibcode='2019Natur.568...55L'` returns full context including Bennu (small body), OSIRIS-REx (mission), linked instruments in <100ms

- **Agent entity context view** (materialized)
  - Joins entities + entity_identifiers + entity_aliases + entity_relationships + document_entities
  - Single query by entity_id returns: canonical name, all IDs, all aliases, related entities, supporting papers
  - Acceptance: `SELECT * FROM agent_entity_context WHERE canonical_name='Bennu'` returns SsODNet ID, SPK-ID, Wikidata QID, aliases (1999 RQ36, 101955), related mission (OSIRIS-REx), paper count

- **Agent dataset context view** (materialized)
  - Joins datasets + dataset_entities + entities + document_datasets
  - Single query by dataset_id returns: dataset metadata, linked entities (instrument, mission, target), citing papers
  - Acceptance: Query for OSIRIS-REx OCAMS bundle returns instrument, mission, target, and paper links

- **MCP server update** (`src/scix/mcp_server.py`)
  - New tool: `resolve_entity` -- wraps entity_resolver with full entity card return
  - New tool: `entity_context` -- queries agent_entity_context view
  - New tool: `document_context` -- queries agent_document_context view (replaces separate entity_search + get_paper workflow)
  - Acceptance: Agent can resolve "Bennu" -> full entity card -> related papers in 2 MCP calls (down from 4-5)

### Should-Have

- **Incremental sync manager** (`src/scix/sync_manager.py`)
  - Tracks last sync timestamp per source via harvest_runs
  - Supports per-source refresh cadence configuration
  - Detects changed/deleted records (where API supports it)
  - Acceptance: Re-running GCMD harvest after 0 upstream changes results in 0 upserts; running after upstream addition results in only new records

- **Wikidata cross-linking for new entities**
  - Extend existing Wikidata enrichment to cover entities in new tables
  - Batch SPARQL queries (50 per request, 2-second delays)
  - Store QIDs in entity_identifiers with `id_scheme='wikidata'`
  - Acceptance: >50% of planetary science entities have Wikidata QIDs

- **Staging schema for new tables**
  - Create staging counterparts for entities, entity_identifiers, entity_aliases
  - Batch-promote to public schema (matching existing staging pattern from migration 015)
  - Acceptance: Harvester writes hit staging tables; promote function atomically moves to public

### Nice-to-Have

- **Entity merge/split audit log**
  - Track when resolution decisions change (entity A merged with entity B, or split apart)
  - Acceptance: After a merge, audit log shows source entities, merge reason, and timestamp

- **Materialized view refresh scheduling**
  - Auto-refresh agent views after harvest_runs complete
  - Acceptance: After running a harvester, views are refreshed within 5 minutes

- **Cross-walk table for SPDF-SPASE ID mapping**
  - Explicit mapping between CDAWeb dataset IDs and SPASE ResourceIDs
  - Acceptance: Given CDAWeb ID `AC_H2_MFI`, returns SPASE ResourceID `spase://NASA/NumericalData/ACE/MAG/L2/PT16S`

## Design Considerations

### Schema Evolution: New Tables vs. Extending entity_dictionary

The entity_dictionary serves vocabulary-level operations (alias resolution, type validation). The new entity graph serves knowledge-graph operations (traversal, linking, provenance). Mixing them forces every dictionary query to pay the cost of graph columns, and every graph query to filter out vocabulary-only entries. A clean separation with a compatibility view is the right trade-off.

The UAT schema (migration 007) with its separate `uat_relationships` and `paper_uat_mappings` tables is architecturally closer to what we need than entity_dictionary's flat pattern.

### Two-Tier Harvest Strategy

Sources split into two categories:

- **Tier 1 (Vocabulary/Catalog)**: GCMD, SPASE, PDS4, CDAWeb -- bounded catalogs, download-parse-load pattern
- **Tier 2 (Entity Enrichment)**: SsODNet (ssoBFT Parquet download), SBDB (targeted API queries for entities already in DB)

CMR is medium-scale: ~55K collections via paginated REST, not a full catalog dump.

### Canonical ID Strategy

| Domain                | Source of Truth | ID Scheme           |
| --------------------- | --------------- | ------------------- |
| Small bodies          | SsODNet         | `ssodnet`           |
| Planetary data        | PDS             | `pds_urn`           |
| Planetary orbits      | SBDB            | `sbdb_spkid`        |
| Earth datasets        | CMR             | `cmr_concept_id`    |
| Earth vocabularies    | GCMD            | `gcmd_uuid`         |
| Heliophysics ontology | SPASE           | `spase_resource_id` |
| Heliophysics data     | SPDF            | `spdf_dataset_id`   |
| Universal cross-link  | Wikidata        | `wikidata`          |

Never merge entities across sources without explicit `same_as` relationship with provenance.

### Performance Budget

- Agent context queries: <100ms (materialized views with PK indexes)
- Entity resolution: <50ms (B-tree on lower(alias), btree on entity_identifiers)
- Full document context: <200ms (includes entity aggregation JSONB build)
- Materialized view refresh: <5 minutes (concurrent refresh, non-blocking)

### No Foreign Key on document_entities.bibcode

Matching the established pattern from citation_edges and citation_contexts: entities may reference papers not yet ingested, and FK constraints would block incremental loading. This is a deliberate pattern shared by every large-scale scientific metadata system (including OpenAlex).

## Open Questions

1. **How many unique small-body names appear in the SciX corpus?** Determines SsODNet strategy: API-based seed (<10K mentions) vs. full ssoBFT download (>10K). Run `SELECT DISTINCT` on extraction mentions before Milestone 2. **[BLOCKING for SsODNet harvester scope]**
2. ~~**Should entity_dictionary be deprecated or maintained long-term?**~~ **RESOLVED (D1)**: entity_dictionary stays untouched. New `entities` table operates in parallel. Compatibility view bridges existing callers. Deprecation is a future decision after all harvesters migrate.
3. **How should confidence propagate through multi-hop relationships?** Paper mentions "MAHLI" (confidence 0.9), MAHLI is instrument on MSL (confidence 1.0) -- what confidence for paper-to-MSL link? **[Deferred to Milestone 3 resolver refinement]**
4. **What is the CDAWeb-to-SPASE ID crosswalk?** Convention exists (`AC_H2_MFI` <-> `spase://NASA/NumericalData/ACE/MAG/L2/PT16S`) but is not trivially parseable. Need to verify if a published mapping exists. **[Deferred to Milestone 3 SPDF harvester]**
5. **What fraction of SciX papers are earth science vs. heliophysics vs. planetary science?** Drives priority allocation for linking passes. **[Run before Milestone 2 linking pass]**

## Risk Register (Updated with Premortem Findings)

Full premortem analysis: [docs/premortem_external_data_integration.md](../premortem_external_data_integration.md)

### R0: Extraction payload is not resolution-ready (CRITICAL) [NEW - Premortem]

- **Likelihood:** High | **Impact:** Critical | **Score:** 12
- **Failure mode:** `extractions.payload` contains coarse, unnormalized strings ("nested sampling", "silicate dust") in 4 flat buckets with no discipline context, no subtype, and no span offsets. Entity resolver achieves only ~40% match rate against 1.2M-entity graph. 8 new tables sit empty of document links.
- **Mitigation:** **BLOCKING GATE**: Before Milestone 2, run 50-paper manual evaluation of extraction→resolution quality. Add acceptance criterion: resolver achieves >70% precision, >50% recall on labeled 200-mention sample. If below threshold, add thin normalization layer (lowercase, abbreviation expansion, parenthetical stripping) between extractions and resolver.
- **Source:** Premortem Lens 4 (Scope & Requirements)

### R1: Materialized view refresh infeasible at projected scale (CRITICAL) [UPGRADED - Premortem]

- **Likelihood:** High | **Impact:** Critical | **Score:** 12
- **Failure mode:** `agent_document_context` joins 5M papers x 18M document_entities x 1.2M entities with JSONB aggregation. `REFRESH MATERIALIZED VIEW CONCURRENTLY` takes 23+ minutes. Concurrent refreshes deadlock. Autovacuum falls behind on document_entities (3.6M dead tuple threshold), causing index bloat from 3GB to 11GB.
- **Mitigation:** **BLOCKING GATE**: Before committing view definitions, benchmark at projected volumes (synthetic 1M entities, 10M document_entities). If refresh >30 min, redesign: partition document_entities by discipline, add pre-aggregated `entity_summary` intermediate matview, serve document context via live query instead of monolithic matview. Set `autovacuum_vacuum_scale_factor = 0.01` on document_entities.
- **Source:** Premortem Lenses 1 + 5 (Technical Architecture + Scale)

### R2: Multi-table staging promote silently discards satellite data (CRITICAL) [NEW - Premortem]

- **Likelihood:** High | **Impact:** Critical | **Score:** 12
- **Failure mode:** Staging promote copies single-table pattern (TRUNCATE after promote). SsODNet loads 1.2M entities in 10K-row chunks; each promote truncates unpromoted aliases/identifiers from other chunks. Result: 1.2M entities with only ~120K aliases (last chunk's worth). harvest_runs records "completed" with no satellite count validation.
- **Mitigation:** Redesign promote for multi-table atomicity: promote entities + identifiers + aliases in single transaction. Add `counts JSONB` to harvest_runs for satellite table validation. Validate alias/identifier counts proportional to entity counts before marking completed.
- **Source:** Premortem Lens 3 (Operational)

### R3: Migration runner fails on first production deployment (CRITICAL) [NEW - Premortem]

- **Likelihood:** Certain | **Impact:** High | **Score:** 12
- **Failure mode:** Migration runner deployed against DB with 18 manually-applied migrations and no tracking table. Runner sees 0 rows in schema_migrations, attempts re-apply all. 013 collision causes silent skip. Seed migration view creation fails, rolling back entities table.
- **Mitigation:** Migration runner must introspect existing tables and backfill schema_migrations for already-applied migrations. Acceptance criterion: "Running updated setup_db.sh against a database with all existing migrations already applied results in N rows in schema_migrations and zero errors."
- **Source:** Premortem Lens 3 (Operational)

### R4: Cross-domain name collisions degrade resolution quality (CRITICAL)

- **Likelihood:** High | **Impact:** High
- **Failure mode:** "Mars" appears as PDS4 target, SPASE ObservedRegion, SBDB object, and CMR platform. `resolve("Mars")` without discipline context returns non-deterministic results.
- **Mitigation:** Entity resolver returns all candidates ranked by discipline. Callers must provide discipline context when available. Never use `LIMIT 1` without ranking. Store all matches with confidence scores.

### R5: Harvester failures are silent (HIGH) [UPGRADED - Premortem]

- **Likelihood:** High | **Impact:** High
- **Failure mode:** SPASE GitHub URL returns 404; harvester logs "0 entities" without error. PDS cursor format changes; harvester enters infinite loop consuming 40GB memory. GCMD GitHub mirror drifts 8 months behind live KMS API. No harvester validates response schema or compares output counts to previous runs.
- **Mitigation:** Add response-schema validation to every harvester (fail loudly on mismatch). Add harvest-count comparison (alert CRITICAL if any source drops below 80% of previous run's count). Add pagination safety bounds (max_pages circuit breaker). Reverse decision D5: all harvesters on shared HTTP client with validation before Phase 2.
- **Source:** Premortem Lenses 2 + 3 (Integration + Operational)

### R6: SsODNet ssoBFT download fails with no resume (HIGH)

- **Likelihood:** Medium | **Impact:** High
- **Failure mode:** 2.1GB download from French academic server fails at ~1.5GB on connection reset. No partial-download resume. Each failure restarts from zero.
- **Mitigation:** Cache locally with SHA-256 checksum. Implement HTTP Range headers for resume. Schema-validate Parquet before ingestion. Fall back to per-object API for critical entities.

### R7: CMR Search-After pagination breaks on server-side changes (MEDIUM)

- **Likelihood:** Medium | **Impact:** Medium
- **Failure mode:** Server-side collection deletions during 3-hour pagination window invalidate cursor.
- **Mitigation:** Deduplicate on concept-id during upsert. Log page counts. Accept <1% record loss at collection level.

### R8: bulk_load single-transaction for 1.2M rows holds write lock (HIGH)

- **Likelihood:** High | **Impact:** Medium
- **Failure mode:** Single executemany() for 1.2M rows holds lock for minutes.
- **Mitigation:** Chunk commits at 10,000 rows. Use staging schema with multi-table atomic promote. Log progress to harvest_runs.cursor_state.

### Premortem-Derived Blocking Gates

| Gate                                  | Must pass before             | Validation                                                     |
| ------------------------------------- | ---------------------------- | -------------------------------------------------------------- |
| Extraction quality evaluation (R0)    | Milestone 2 resolver/linking | >70% precision, >50% recall on 200-mention labeled sample      |
| Matview scale benchmark (R1)          | Phase 4 view definitions     | REFRESH CONCURRENTLY <30 min at 1M entities / 10M doc_entities |
| Migration runner production test (R3) | Any new migration            | Run against DB with all existing migrations; zero errors       |
| Multi-table promote validation (R2)   | SsODNet harvester            | Alias/identifier counts match entity counts after promote      |

## Implementation Order (Revised per Convergence)

### Milestone 1: Infrastructure + Schema (~1 week)

**Hard gate: nothing else starts until Phase 0 is verified.**

```
Phase 0: Infrastructure (HARD GATE)
  ├── schema_migrations table + setup_db.sh migration runner
  ├── Fix 013 migration numbering collision
  ├── harvest_runs table
  ├── Shared HTTP client module (new harvesters must use; existing refactored later)
  └── Staging schema extensions (staging.entities, staging.entity_identifiers, staging.entity_aliases)

Phase 1: Schema Extension (single migration, just DDL)
  ├── entities + entity_identifiers + entity_aliases + entity_relationships
  ├── document_entities bridge table
  ├── entity_dictionary_compat view (~10 lines SQL)
  └── Seed migration: INSERT INTO entities SELECT ... FROM entity_dictionary (55K rows day one)
```

Note: `datasets`, `dataset_entities`, `document_datasets` tables deferred to Milestone 3 (coupled to CMR/SPDF maturity).

### Milestone 2: First Harvesters + Resolver (~2-3 weeks)

```
Harvester Updates (3 sources, not 7):
  ├── Update harvest_gcmd.py → write to entities + entity_identifiers (low risk, proves pattern)
  ├── Update harvest_pds4.py → write entities + entity_identifiers + entity_relationships
  └── New harvest_spdf.py (single REST call, smallest new pipeline, high heliophysics value)

Resolution & Linking:
  ├── Entity resolver: exact canonical → alias lookup → entity_identifiers → discipline ranking
  ├── Batch linking pass: extractions → document_entities (chunked, resumable)
  └── Staging schema for linking pass output (large batch writes)
```

SsODNet scope decided by data: run `SELECT DISTINCT` on extraction mentions. If <10K unique small-body names, use API-based seed. If >10K, download ssoBFT via staging.

### Milestone 3: Remaining Sources + Agent Layer (subsequent PRD)

```
Additional Harvesters:
  ├── SsODNet ssoBFT bulk load (if not done in M2) via staging
  ├── CMR collection harvester (55K collections, Search-After pagination)
  ├── SBDB enrichment (sequential API, depends on SsODNet)
  └── SPASE harvester update (v2.7.1 + new tables)

Dataset Infrastructure:
  ├── datasets + dataset_entities + document_datasets tables
  └── Document-dataset linking pipeline

Agent Layer:
  ├── Materialized views (document, entity, dataset context)
  ├── MCP server new tools (resolve_entity, entity_context, document_context)
  └── Existing harvester refactor to shared HTTP client
```

### Gate Criteria

| Gate                      | Condition                                                                          | Verified By                                                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Phase 0 → Phase 1         | schema_migrations has row for every applied migration; 013 collision resolved      | `SELECT count(*) FROM schema_migrations` matches file count                                                           |
| Phase 1 → Milestone 2     | entities table exists; seed from entity_dictionary complete                        | `SELECT count(*) FROM entities` returns ~55K                                                                          |
| Milestone 2 → Milestone 3 | >=2 harvesters writing to new tables; document_entities has rows from linking pass | `SELECT count(DISTINCT source) FROM entities WHERE source NOT IN (SELECT DISTINCT source FROM entity_dictionary)` > 0 |

## Definition of Done

### Milestone 2 (MVP)

A paper mentioning "OSIRIS-REx" can be resolved to:

- **Mission**: OSIRIS-REx (PDS), with PDS URN, linked instruments, and Bennu target via entity_relationships
- **Cross-references**: GCMD keywords normalized, SPDF heliophysics datasets linked
- **Document bridge**: document_entities connects paper bibcode to resolved entities with confidence scores

Queryable via direct JOINs in <100ms with proper indexes.

### Milestone 3 (Full Vision)

A paper mentioning "Bennu", "OSIRIS-REx", and "spectral reflectance data" can be resolved to:

- **Canonical small body**: Bennu (SsODNet), with aliases [1999 RQ36, 101955], SPK-ID, Wikidata QID
- **Mission**: OSIRIS-REx (PDS), with PDS URN, linked instruments [OCAMS, OLA, OTES, OVIRS, REXIS]
- **Dataset**: OSIRIS-REx OCAMS bundle (PDS), CMR collection (if cross-listed)
- **Related vocabulary**: spectral reflectance -> GCMD science keyword or SPASE measurement type
- **Heliophysics context**: solar wind measurements from ACE (if paper references)

All queryable in <100ms from materialized views with:

- Canonical IDs from authoritative sources
- All known aliases
- Entity-entity relationships with provenance
- Confidence scores on all links
- Harvest run provenance (when was this data last refreshed)
