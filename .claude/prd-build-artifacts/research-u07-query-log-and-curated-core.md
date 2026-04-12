# Research — u07 query_log + curated entity core

## Schema snapshot

### `query_log` (post-migration 031)

- id SERIAL PK
- tool_name TEXT NOT NULL (legacy, from 016)
- params_json JSONB (legacy)
- latency_ms REAL (legacy)
- success BOOLEAN NOT NULL (legacy)
- error_msg TEXT (legacy)
- created_at TIMESTAMPTZ NOT NULL DEFAULT now()
- **ts TIMESTAMPTZ NOT NULL DEFAULT now()** (new)
- **tool TEXT** (new)
- **query TEXT** (new)
- **result_count INT** (new)
- **session_id TEXT** (new)
- **is_test BOOLEAN NOT NULL DEFAULT false** (new)

Constraint: `tool_name` and `success` are NOT NULL and have no defaults. The
new `log_query()` wrapper must fill these; we populate `tool_name <- tool`
and `success <- (error_msg is NULL)`.

### `entities` (post migration 028)

- id SERIAL PK
- canonical_name TEXT NOT NULL
- source TEXT NOT NULL
- ambiguity_class entity_ambiguity_class (nullable; populated by u05)
- ...

### `document_entities` (post 028)

- PK (bibcode, entity_id, link_type, tier)
- tier=1 rows written by u06 pipeline

## Existing `_log_query` in mcp_server.py

`src/scix/mcp_server.py:169` already writes to query_log but only to the
legacy columns. The new `log_query()` wrapper is used by the backfill
script and by new MCP instrumentation; it populates the new columns while
keeping the legacy NOT NULL columns satisfied.

## Ambiguity classifier

`src/scix/ambiguity.py` is the pure classifier. We do NOT call it here; we
read `entities.ambiguity_class` values directly from SQL.

## Artifacts to produce

- `build-artifacts/curated_core.csv`
- `build-artifacts/curated_core_stratification.md`

## Migration 032

Next unused migration number. Must create:

- `curated_entity_core` (entity_id PK, query_hits_14d, promoted_at)
- `core_promotion_log` (event log of promote/demote operations)
