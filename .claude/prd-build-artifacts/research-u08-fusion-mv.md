# Research — u08-fusion-mv

## Schema inputs

- `document_entities (bibcode TEXT, entity_id INT, link_type TEXT, confidence REAL, match_method TEXT, evidence JSONB, harvest_run_id INT, tier SMALLINT, tier_version INT)`, PK `(bibcode, entity_id, link_type, tier)`.
- `entities (id INT PK, canonical_name, entity_type, ...)`.
- `tier` is SMALLINT; `tier_weight(SMALLINT)` arg type must match.

## Existing MV prior art

- `migrations/024_agent_context_views.sql` already uses `CREATE MATERIALIZED VIEW IF NOT EXISTS` + `CREATE UNIQUE INDEX` (needed for REFRESH CONCURRENTLY).
- No SQL function / calibration log in repo yet.

## AST lint (`scripts/ast_lint_resolver.py`)

- Bans `SELECT ... FROM document_entities_canonical` in Python source files outside `src/scix/resolve_entities.py`.
- `# noqa: resolver-lint` comment on the offending line exempts it (works for multiline string literals too, per tests).
- Our `src/scix/fusion_mv.py` must not read from the MV OR must use the noqa tag. Plan: helper file issues only `REFRESH MATERIALIZED VIEW ...` + writes to `fusion_mv_state` (no SELECT FROM canonical at all — lint safe). The Python test uses raw psql SELECT wrapped in `# noqa: resolver-lint`.

## Test DSN pattern

- `tests/helpers.py` exposes `get_test_dsn()` — returns SCIX_TEST_DSN if set, None otherwise; skip destructive tests when None.
- `scix_test` db has the full schema through migration 031.

## Noisy-OR formula and clamp

- `fused = 1 - exp(sum(ln(1 - c_t * w_t)))`
- If `c_t * w_t == 1.0`, ln(0) = -inf. Clamp via `LEAST(0.9999, c_t * w_t)`.
- For typical placeholder weights (max 0.98) and confidences ≤ 1.0, max product is 0.98 — well below clamp, so closed-form equivalence preserved.

## Placeholder weights

- t1 = 0.98, t2 = 0.85, t3 = 0.92 (deprecated), t4 = 0.50, t5 = 0.88 (JIT)
- tier 0 (legacy) weight: leave unset via CASE → default small weight (e.g. 0.50) or NULL → skip. Spec lists t1..t5 only. Emit default 0.50 for anything outside 1..5 for safety.

## Refresh rate limiter design

- `fusion_mv_state(id INT PK DEFAULT 1 CHECK id=1, dirty BOOL DEFAULT true, last_refresh_at TIMESTAMPTZ)`.
- `mark_dirty()`: UPSERT dirty=true.
- `refresh_if_due(min_interval)`: SELECT dirty, last_refresh_at; if dirty and (last_refresh_at IS NULL OR now()-last > interval) → REFRESH CONCURRENTLY then set dirty=false, last_refresh_at=now(); return True. Else False.
- REFRESH CONCURRENTLY must run outside a transaction block, so use `conn.autocommit = True` during refresh.
