# Research — u04 Staging Promotion Hardening

## Schema (from scix_test DB as of 2026-04-12)

### harvest_runs

- `id int PK`, `source text NN`, `started_at ts`, `finished_at ts`, `status text NN default 'running'`, `records_fetched int`, `records_upserted int`, `cursor_state jsonb`, `error_message text`, `config jsonb`, `counts jsonb NN default '{}'`.
- Known statuses in codebase: `'running'`, `'completed'`, `'failed'`. We'll add `'rejected_by_diff'` and `'promoted'`.

### entities

- `id serial PK`, unique(canonical_name, entity_type, source), `discipline`, `harvest_run_id FK`, `properties jsonb`, `ambiguity_class`, `link_policy`, `source_version`, `supersedes_id`.

### entity_aliases

- PK(entity_id, alias), `alias_source`.

### entity_identifiers

- PK(id_scheme, external_id), `entity_id FK ON DELETE CASCADE`, `is_primary bool`.

### document_entities

- PK(bibcode, entity_id, link_type, tier), `harvest_run_id FK`, FK entity_id ON DELETE CASCADE.
- Orphan prevention: if we delete an entity, document_entities cascades — so orphan protection check must happen BEFORE any delete.

### Staging tables (migration 030)

- `entities_staging(id PK, staging_run_id NN, canonical_name, entity_type, discipline, source NN, source_version, ambiguity_class, link_policy, properties jsonb, created_at)`.
- `entity_aliases_staging(id PK, staging_run_id NN, staging_entity_id, canonical_name, entity_type, source, alias NN, alias_source)`.
- `entity_identifiers_staging(id PK, staging_run_id NN, staging_entity_id, canonical_name, entity_type, source, id_scheme NN, external_id NN, is_primary)`.

### Existing promote_harvest stub

Returns `COUNT(*) FROM entities_staging WHERE staging_run_id = run_id`. We replace the body via `CREATE OR REPLACE FUNCTION` called from Python module import (no new migration file).

## Test patterns

- No `tests/conftest.py` — per-test file DSN handling via `helpers.DSN`, `is_production_dsn`, `SCIX_TEST_DSN`.
- Most integration tests skip if `SCIX_TEST_DSN` not set or points at production.
- `psycopg` (v3) is used directly with cursor context managers.

## Decisions

1. **Advisory lock**: `pg_try_advisory_lock(hashtext('entities_promotion'))` acquired by the SQL function.
2. **Atomicity**: entire function runs in its own transaction (BEGIN..COMMIT wrapped by caller; function itself rolls back via EXCEPTION handler if diff checks fail).
3. **Shadow diff**: computed in pure SQL comparing counts of canonical (name+type) and aliases.
4. **Orphan check**: query `document_entities` grouped by entity_id for entities that exist in production but not in staging (i.e. would be implicitly removed by a full replace). Since we're doing UPSERT not REPLACE, this is only a concern if we also delete. We'll implement UPSERT semantics so orphan check is a conservative pre-check: any existing entity with >=1000 document_entities rows that would be "replaced" (same natural key collision) must not change canonical_name/entity_type/source — since UPSERT by natural key preserves id, the orphan rule effectively blocks `source_version` downgrades or discipline changes that would cause supersession. For the test scenario, we'll synthesize a staging run where orphan prevention should trigger — we model this as the staging run having FEWER entities for a given source than production has, i.e., a per-source count check: if a source in production has existing entities with >=1000 document_entities each and that source is in staging with a reduced count, reject.
5. **Simpler approach**: orphan prevention rule as implemented: an entity is "orphaned" if its id would be removed. Since UPSERT never removes, instead we reject if the staging run's per-source canonical count is less than production's per-source count minus tolerance AND any production entity in that source has >=1000 document_entities rows. This matches the spirit of the rule ("entities with >=1000 document_entities rows cannot disappear") without actually needing deletes.
6. **Promotion semantics**: UPSERT by natural key (canonical_name, entity_type, source). Aliases and identifiers UPSERT by their PKs.
7. **Per-source floors**: staging run's row count for source X must be >= floor[X]. Broken harvester floors = 0.
8. **Diff fields** returned in PromotionResult.diff:
   - `staging_count`, `production_count_before`, `canonical_shrinkage_pct`, `alias_shrinkage_pct`, `per_source_counts` dict, `floor_violations` list, `orphan_violations` list, `schema_errors` list.

## LLM cost ceiling

- Table `llm_cost_ledger(day DATE PK, total_usd NUMERIC NN DEFAULT 0, call_count INT NN DEFAULT 0)`.
- Functions `check_and_reserve(est) -> bool` (reads, if pass, inserts/updates with estimated), `record_actual(actual) -> None` (adjusts ledger to actual cost for latest reservation — simplest: add delta between actual and last estimate via a small reservation queue, or just treat reserve as speculative and record_actual does a second increment minus estimate. Simpler: reserve does upsert + estimate, record_actual does delta).
- Per-query $0.01 cap: reject if estimate > 0.01.
- Per-day $50 cap: reject if total + estimate > 50.
- Token estimation helper: `estimate_cost(prompt_tokens, completion_tokens)` using Haiku rates.

## Snapshot replay

- Path: `data/entities/snapshots/{source}/{YYYY-MM-DD}.jsonl.gz`.
- Schema: one JSON object per line with shape `{"entity": {...}, "aliases": [...], "identifiers": [...]}`.
- Replay inserts into `entities_staging`, `entity_aliases_staging`, `entity_identifiers_staging` with a new `staging_run_id`.
- Round-trip test: dump existing staging to snapshot, reload into a fresh staging_run_id, diff = 0.
