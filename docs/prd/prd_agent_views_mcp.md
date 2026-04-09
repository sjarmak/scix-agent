# PRD: Agent Views & MCP Tools for Entity Graph

## Problem Statement

The entity graph schema (migration 021) and harvesters are in place, but agents still cannot query entity context in a single call. An agent asking "what do we know about Bennu?" must make 4-5 MCP tool calls and manually stitch results from entity_dictionary, extractions, and papers tables. Three materialized views and three new MCP tools will collapse this to 1-2 calls.

## Prior Work (Already Built)

- `migrations/021_entity_graph.sql`: entities, entity_identifiers, entity_aliases, entity_relationships, document_entities, datasets, dataset_entities, document_datasets
- `src/scix/entity_resolver.py`: EntityResolver with exact/alias/identifier/fuzzy cascade
- `src/scix/link_entities.py`: batch document-entity linking pipeline
- `src/scix/mcp_server.py`: existing MCP server with paper/search/entity tools
- Harvesters writing to entity graph: GCMD, PDS4, SPDF (+ more from remaining harvesters PRD)

## Blocking Gate: Materialized View Scale Benchmark

**Before committing view definitions**, benchmark at projected volumes:

- Generate synthetic data: 1M entities, 10M document_entities, 100K entity_aliases
- Run `CREATE MATERIALIZED VIEW` and `REFRESH MATERIALIZED VIEW CONCURRENTLY`
- If refresh >30 minutes: redesign with partitioned document_entities, pre-aggregated intermediate matview, or live query fallback
- Set `autovacuum_vacuum_scale_factor = 0.01` on document_entities

This gate must pass before any view migration is applied.

## Goals

1. Provide single-query agent context for documents, entities, and datasets (<100ms)
2. Add 3 MCP tools that replace the current 4-5 call workflow
3. Ensure views refresh concurrently without blocking reads

## Non-Goals

- Real-time view updates (batch refresh on harvest completion is sufficient)
- Embedding-based search in views (use existing search infrastructure)
- Auto-refresh scheduling (nice-to-have, not MVP)

## Requirements

### Must-Have

- **Scale benchmark script** (`scripts/bench_matviews.py`)
  - Generate synthetic entities (1M), document_entities (10M), entity_aliases (500K) in a test schema
  - Create each materialized view definition
  - Measure: CREATE time, REFRESH CONCURRENTLY time, single-row query time
  - Output results to stdout and `build-artifacts/matview-benchmark.md`
  - Acceptance: Script runs to completion; results include timing for all 3 views; recommendations printed if any view exceeds 30-minute refresh

- **Agent document context view** (materialized, migration 023 or later)
  - Definition: papers LEFT JOIN document_entities JOIN entities, with JSONB aggregation of linked entities including canonical_names, aliases, external_ids, types, disciplines
  - `UNIQUE INDEX` on bibcode for `REFRESH MATERIALIZED VIEW CONCURRENTLY`
  - Single query by bibcode returns: paper metadata + all linked entities with full context
  - Acceptance: `SELECT * FROM agent_document_context WHERE bibcode='2019Natur.568...55L'` returns full context in <100ms; `REFRESH MATERIALIZED VIEW CONCURRENTLY agent_document_context` completes in <30 minutes at projected scale

- **Agent entity context view** (materialized)
  - Definition: entities LEFT JOIN entity_identifiers, entity_aliases, entity_relationships, document_entities, with JSONB aggregation
  - Single query by entity_id or canonical_name returns: canonical name, all IDs, all aliases, related entities (with predicates), supporting paper count
  - Acceptance: `SELECT * FROM agent_entity_context WHERE canonical_name='Bennu'` returns SsODNet ID, SPK-ID, aliases, related missions, paper count in <100ms

- **Agent dataset context view** (materialized)
  - Definition: datasets LEFT JOIN dataset_entities JOIN entities, LEFT JOIN document_datasets
  - Single query by dataset_id returns: dataset metadata, linked entities (instrument, mission, target), citing paper count
  - Acceptance: Query for SPDF dataset returns instrument, observatory, and paper links in <100ms

- **MCP tool: `resolve_entity`**
  - Wraps EntityResolver.resolve() with full entity card return
  - Input: mention text, optional discipline
  - Output: list of candidates with canonical_name, entity_type, source, discipline, confidence, all identifiers, all aliases
  - Acceptance: MCP call `resolve_entity("Bennu")` returns entity card with SsODNet source, aliases, identifiers

- **MCP tool: `entity_context`**
  - Queries agent_entity_context view
  - Input: entity_id or canonical_name
  - Output: full entity context (identifiers, aliases, relationships, paper count)
  - Acceptance: `entity_context("OSIRIS-REx")` returns mission with linked instruments, target (Bennu), paper count

- **MCP tool: `document_context`**
  - Queries agent_document_context view
  - Input: bibcode
  - Output: paper metadata + all linked entities with types and confidence scores
  - Replaces the current separate `entity_search` + `get_paper` workflow
  - Acceptance: `document_context("2019Natur.568...55L")` returns paper + entity links in single call

- **View refresh helper** (`src/scix/views.py`)
  - `refresh_view(conn, view_name)` — wraps `REFRESH MATERIALIZED VIEW CONCURRENTLY`
  - `refresh_all_views(conn)` — refreshes all 3 agent views in sequence
  - Logs refresh duration to harvest_runs or a dedicated refresh_log
  - Acceptance: `refresh_all_views()` completes without error; logged durations available

### Should-Have

- **Document-dataset linking pipeline** (`src/scix/link_datasets.py`)
  - Match dataset references in papers: ADS `data` field entries, citations to data DOIs, explicit dataset mentions
  - Write results to document_datasets with confidence and match_method
  - Chunked, resumable (same pattern as link_entities)
  - Acceptance: Papers with ADS `data` field entries linked to corresponding CMR/PDS/SPDF datasets

- **Incremental sync manager** (`src/scix/sync_manager.py`)
  - Track last sync timestamp per source via harvest_runs
  - `needs_refresh(source)` checks cadence config
  - `run_sync(source)` triggers harvester + view refresh
  - Acceptance: Re-running GCMD harvest after 0 upstream changes results in 0 upserts

### Nice-to-Have

- **Materialized view refresh scheduling**
  - Auto-refresh agent views after any harvest_run completes
  - Could be a PostToolUse hook or a cron job
  - Acceptance: After running a harvester, views are refreshed within 5 minutes

- **Entity merge/split audit log**
  - Track when resolution decisions change
  - Acceptance: After a merge, audit log shows source entities, merge reason, timestamp

- **SPDF-SPASE ID crosswalk table**
  - Explicit mapping between CDAWeb dataset IDs and SPASE ResourceIDs
  - Acceptance: Given CDAWeb ID `AC_H2_MFI`, returns SPASE ResourceID

## Design Considerations

### View Refresh Strategy

All 3 views use `REFRESH MATERIALIZED VIEW CONCURRENTLY` which requires a UNIQUE index and does not block reads. The trade-off is that data is stale between refreshes. For a batch-harvesting system this is acceptable — views are refreshed after each harvest run.

If the scale benchmark shows >30 minute refresh for agent_document_context (the largest), fallback options:

1. Partition document_entities by discipline, refresh per-partition
2. Add pre-aggregated `entity_summary` intermediate matview
3. Serve document context via live query with proper indexes (no matview)

### MCP Tool Design

The 3 new tools follow the existing MCP server pattern in `src/scix/mcp_server.py`. Each tool:

- Accepts a single primary identifier (bibcode, entity_id/name, dataset_id)
- Returns a JSON object with all context pre-joined
- Uses the materialized views for <100ms response time

### Performance Budget

| Operation                    | Target  | Fallback if exceeded                 |
| ---------------------------- | ------- | ------------------------------------ |
| agent_document_context query | <100ms  | Add covering index on bibcode        |
| agent_entity_context query   | <100ms  | Add covering index on canonical_name |
| agent_dataset_context query  | <100ms  | Live query (dataset count is small)  |
| Full view refresh (all 3)    | <30 min | Partition + intermediate matview     |
| entity_resolver.resolve()    | <50ms   | Already met via B-tree indexes       |

## Implementation Order

```
Layer 0 (blocking gate):
  └── scale-benchmark (must pass before any view definition)

Layer 1 (depends on benchmark passing):
  ├── document-context-view (migration + view definition)
  ├── entity-context-view
  ├── dataset-context-view
  └── view-refresh-helper

Layer 2 (depends on views):
  ├── mcp-resolve-entity
  ├── mcp-entity-context
  └── mcp-document-context

Layer 3 (should-have, independent):
  ├── link-datasets-pipeline
  └── sync-manager
```
