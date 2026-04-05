# Premortem: External Data Integration for Cross-Domain Entity Resolution

## Risk Registry

| #   | Failure Lens             | Severity     | Likelihood | Score  | Root Cause                                                                                                 | Top Mitigation                                                                                     |
| --- | ------------------------ | ------------ | ---------- | ------ | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1   | Technical Architecture   | Critical (4) | High (3)   | **12** | Schema designed for 55K scale, never load-tested at 1.2M entities / 18M document_entities                  | Benchmark matview refresh at projected volumes before committing view definitions                  |
| 2   | Scope & Requirements     | Critical (4) | High (3)   | **12** | Assumed extractions.payload is resolution-ready; never validated match quality                             | Run 50-paper manual evaluation of extraction→resolution quality before Milestone 2                 |
| 3   | Operational              | Critical (4) | High (3)   | **12** | Migration runner untested against existing untracked DB; staging promote not adapted for multi-table model | Test migration runner against production-state DB; redesign promote for parent+satellite atomicity |
| 4   | Integration & Dependency | Critical (4) | High (3)   | **12** | 7 independent HTTP/parsing layers with no shared validation, no format drift detection                     | Require all harvesters on shared HTTP client with response-schema validation                       |
| 5   | Scale & Evolution        | Critical (4) | High (3)   | **12** | Monolithic materialized views with O(n\*m) refresh cost on 18M+ row joins                                  | Partition document_entities; use incremental refresh instead of full rebuild                       |

## Cross-Cutting Themes

### Theme 1: The Extraction→Resolution Bridge Is Unvalidated (Lenses 1, 4, 5)

Three independent failure narratives converge on the same structural gap: the system assumes NER-extracted mention strings in `extractions.payload` can be reliably resolved against 1.2M+ canonical entities, but this has never been tested. The extraction pipeline produces coarse, unnormalized strings ("nested sampling", "silicate dust") into 4 flat buckets with no discipline context, no subtype, and no span offsets. The entity resolver is designed for clean inputs (canonical names, known aliases) but will receive messy free-text. This is the **highest-confidence risk** because it's a data quality problem, not a code bug -- no amount of schema elegance fixes it.

### Theme 2: Scale Was Projected But Never Benchmarked (Lenses 1, 5)

Both the Technical Architecture and Scale narratives independently identified the same failure: the materialized view `agent_document_context` joining 5M papers x 18M document_entities x 1.2M entities with JSONB aggregation is computationally infeasible to refresh in <5 minutes on a single PostgreSQL instance. The PRD specifies <100ms query and <5 minute refresh targets but contains zero benchmarks or cardinality projections through the JOIN fan-out. SsODNet alone (1.2M x 4.3 aliases = 5.2M alias rows) makes the normalized schema's JOIN cost visible.

### Theme 3: Multi-Table Staging Promote Is a Latent Data-Loss Bug (Lenses 3, 1)

The Operational and Technical Architecture narratives both flag that the staging promote pattern from migration 015 (`TRUNCATE` after single-table promote) will silently discard satellite table data when applied to the entity model's parent-children structure. The SsODNet load (1.2M entities + 5.2M aliases + 2.5M identifiers) is the first operation that will trigger this, and the harvest_runs table has no satellite-table count validation to catch it.

### Theme 4: Migration Runner Will Fail on First Production Deployment (Lens 3)

The migration runner is designed for fresh databases but will be deployed against a database with 18 manually-applied migrations and no tracking table. This is not a risk -- it is a certainty. The runner must introspect existing state and backfill `schema_migrations` before applying anything new.

### Theme 5: Harvester Failures Are Silent (Lenses 2, 3)

The Integration narrative showed SPASE returning 404 and the harvester logging "0 entities harvested" without raising. The Operational narrative showed harvest_runs recording `status='completed'` for a partially-loaded SsODNet dataset. No harvester validates that output counts match expectations, and no harvest has a `--force` re-run capability.

## Mitigation Priority List

Ranked by: failure modes addressed x severity x implementation cost.

| Priority | Mitigation                                                                               | Addresses       | Cost   | Impact                                                               |
| -------- | ---------------------------------------------------------------------------------------- | --------------- | ------ | -------------------------------------------------------------------- |
| **P0**   | Run 50-paper manual evaluation of extraction→resolution quality                          | Themes 1        | Low    | Surfaces the #1 blocker before any infrastructure is built           |
| **P1**   | Test migration runner against production-state DB (backfill schema_migrations)           | Theme 4, Lens 3 | Low    | Prevents guaranteed first-deployment failure                         |
| **P2**   | Benchmark matview refresh at projected volumes (synthetic 1M entities, 10M doc_entities) | Theme 2         | Low    | Validates or invalidates the entire view strategy before building it |
| **P3**   | Redesign staging promote for multi-table atomicity                                       | Theme 3         | Medium | Prevents data loss on first large harvester run                      |
| **P4**   | Add response-schema validation + harvest count comparison to all harvesters              | Theme 5         | Medium | Prevents silent failures across all 7 sources                        |
| **P5**   | Shared HTTP client for ALL harvesters (reverse D5)                                       | Lens 2          | Medium | Eliminates 7 independent failure modes                               |
| **P6**   | Partition document_entities by discipline or year                                        | Theme 2         | Medium | Enables scoped matview refresh, reduces vacuum pressure              |
| **P7**   | Add pre-aggregated entity_summary matview as intermediate layer                          | Lens 1          | Medium | Reduces agent view JOIN from 6-table to 3-table                      |
| **P8**   | Promote frequently-queried JSONB properties to typed columns                             | Lens 5          | Medium | Prevents GIN index bloat on heterogeneous keys                       |
| **P9**   | Add thin normalization layer between extractions and resolver                            | Lens 4          | Low    | Improves match rate without re-extraction                            |

## Design Modification Recommendations

### 1. Add Extraction Quality Gate Before Milestone 2 (P0)

**What to change**: Before building the entity resolver or linking pipeline, run a manual evaluation: take 50 papers with known entity associations, attempt resolution of their `extractions.payload` mentions against entity_dictionary, measure precision/recall. Add an acceptance criterion: "resolver achieves >70% precision, >50% recall on labeled 200-mention sample."

**Failure modes addressed**: Scope & Requirements (Lens 4), Technical Architecture (indirect)

**Effort**: 1-2 days. Zero infrastructure required.

### 2. Benchmark Materialized Views at Scale Before Committing (P2)

**What to change**: Generate synthetic data at projected volumes (1M entities, 5M aliases, 10M document_entities). Time `REFRESH MATERIALIZED VIEW CONCURRENTLY` on the proposed `agent_document_context` definition. If >30 minutes, redesign: use incremental summary tables (`entity_summary` matview as intermediate) or serve document context via live query with proper indexes instead of monolithic matview.

**Failure modes addressed**: Technical Architecture (Lens 1), Scale & Evolution (Lens 5)

**Effort**: 2-3 days. Synthetic data generation + benchmark script.

### 3. Redesign Staging Promote for Multi-Table Entity Model (P3)

**What to change**: Replace the single-table `TRUNCATE`-after-promote pattern with atomic multi-table promotion: promote entities, identifiers, and aliases in a single transaction, validate counts match expectations, then truncate all staging tables together. Add `counts JSONB` to harvest_runs for satellite table validation.

**Failure modes addressed**: Operational (Lens 3), Technical Architecture (indirect)

**Effort**: 1 day. ~50 lines of SQL in the migration.

### 4. Reverse Decision D5: All Harvesters on Shared HTTP Client (P5)

**What to change**: Require existing harvesters (GCMD, SPASE, PDS4) to use the shared HTTP client before Phase 2 starts. The refactor is mechanical (~150 lines of duplicated retry code across 3 files). Add response-schema validation and harvest-count comparison (alert if any source drops below 80% of previous run).

**Failure modes addressed**: Integration & Dependency (Lens 2), Operational (Theme 5)

**Effort**: 3-4 days. Mostly mechanical extraction of existing retry logic.

### 5. Partition document_entities and Use Incremental Refresh (P6 + P7)

**What to change**: Partition `document_entities` by discipline (4 partitions: astrophysics, earth_science, heliophysics, planetary_science). Create `entity_summary` as intermediate matview (entity_id -> aliases array, identifiers JSONB, relationship count). Agent-facing views JOIN against entity_summary instead of raw satellite tables. Refresh only affected partitions after each harvester run.

**Failure modes addressed**: Scale & Evolution (Lens 5), Technical Architecture (Lens 1)

**Effort**: 3-5 days. Schema change + view redesign.

## Full Failure Narratives

### Lens 1: Technical Architecture Failure (Severity: Critical, Likelihood: High)

Schema designed for correctness at vocabulary scale (55K entities) never load-tested at actual volumes (1.2M entities, 5.8M aliases, 18M document_entities). Entity resolver performing 3 sequential queries per resolution across normalized tables made the 16M-call linking pass take 11 days instead of hours. Materialized view refresh consumed 140GB temp space and took 9+ hours. Team denormalized in panic, creating three competing alias representations with no single source of truth.

**Root cause**: No cardinality projection through the normalized schema before writing DDL.

### Lens 2: Integration & Dependency Failure (Severity: Critical, Likelihood: High)

SPASE GitHub URL returned 404 silently (0 entities, no error). PDS cursor format changed, causing infinite pagination loop consuming 40GB memory. GCMD GitHub mirror drifted 8 months behind live KMS API. SsODNet 2.1GB download failed at 1.5GB three times with no resume capability. SBDB blocked waiting on SsODNet. CMR cursor invalidated by server-side deletes.

**Root cause**: 7 independent HTTP/parsing implementations with no shared validation or format drift detection.

### Lens 3: Operational Failure (Severity: Critical, Likelihood: High)

Migration runner deployed against untracked production DB attempted to re-apply all 18 migrations. 013 collision caused silent skip. Seed migration's view creation failed, rolling back entities table. SsODNet staging promote truncated aliases between chunks, leaving 90% of entities with zero aliases. harvest_runs showed "completed" with no satellite count validation. Recovery required manual DELETE of 1.2M rows, locking entities table for 40 minutes.

**Root cause**: Migration runner tested only against fresh databases; staging promote copied from single-table pattern without multi-table adaptation.

### Lens 4: Scope & Requirements Failure (Severity: Critical, Likelihood: High)

NER extractions produce coarse, unnormalized strings with no discipline context or subtype distinction. Entity resolver achieved only ~40% match rate against 1.2M-entity graph. 8 new tables sat mostly empty of document links. Re-extraction required multi-day, multi-dollar batch API operation not budgeted for. Agent experience degraded: new tools returned sparse data while old tools weren't integrated.

**Root cause**: Assumed extractions.payload was resolution-ready without validating match quality on real data.

### Lens 5: Scale & Evolution Failure (Severity: Critical, Likelihood: High)

At 18M document_entities rows, matview refresh took 23+ minutes. Concurrent refreshes deadlocked. GIN index on heterogeneous JSONB properties bloated to 1.4GB. Autovacuum fell behind (3.6M dead tuples threshold), causing index bloat from 3GB to 11GB. WAL generation hit 40GB/day during linking passes. <100ms SLA missed 35% of the time by month five.

**Root cause**: Monolithic materialized views with O(n\*m) refresh cost, not designed for incremental maintenance.
