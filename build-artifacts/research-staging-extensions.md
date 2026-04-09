# Research: Staging Extensions for Entity Graph

## Existing Staging Pattern (015_staging_schema.sql)

- Creates `staging` schema with `staging.extractions` mirroring `public.extractions`
- No FK constraints in staging (allows writes referencing non-existent public rows)
- `staging.promote_extractions()` function: upserts from staging to public via ON CONFLICT, then TRUNCATEs staging
- Returns promoted row count via CTE with RETURNING 1
- Wrapped in BEGIN/COMMIT transaction

## Entity Dictionary (013_entity_dictionary.sql)

- `entity_dictionary` table: id SERIAL PK, canonical_name TEXT, entity_type TEXT, source TEXT, external_id TEXT, aliases TEXT[], metadata JSONB
- UNIQUE(canonical_name, entity_type, source)
- GIN index on aliases array

## Public Entity Tables

- No `public.entities`, `public.entity_identifiers`, or `public.entity_aliases` tables exist yet
- Migration 022 must create both public and staging versions
- `harvest_runs` table exists (020) but spec says staging tables should NOT have FK to it

## Key Design Decisions

- Promote function must handle entity_id remapping: staging entity IDs won't match public entity IDs
- Must join through (canonical_name, entity_type, source) natural key to resolve correct public entity_id
- All 3 tables promoted atomically in single transaction
- TRUNCATE all 3 staging tables after promote
