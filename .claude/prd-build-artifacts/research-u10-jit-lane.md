# Research — u10-jit-lane

## u03 resolve_entities.py (mock lane host)

- Public API: `resolve_entities(bibcode, context: EntityResolveContext) -> EntityLinkSet`
- Four lane helpers (static/jit_cache/live_jit/local_ner), each mock-backed.
- `EntityLinkSet` requires the `_RESOLVER_INTERNAL` token — cannot be built
  outside `scix.resolve_entities`. **Implication:** our jit modules cannot
  synthesise `EntityLinkSet` directly. Tests for local_ner/router will use a
  light shim dataclass (`LocalNERResult`) OR monkeypatch a factory that
  resolve_entities can inject. Per instructions "Option A": expose public
  functions (`call_live_jit`, `get_cached`, `run_local_ner`, `route_jit`) that
  return a domain type the resolver will wrap. We'll return a small
  dataclass `LocalNERResult(bibcode, entity_ids, confidences, lane,
model_version, candidate_set_hash)` and let resolve_entities wrap it.

## entity_link_set.py

- `EntityLink(entity_id, confidence, link_type, tier, lane)` is a plain
  frozen dataclass — accessible from jit modules for internal use.

## Migration pattern (028/032/033)

- Use `BEGIN; ... COMMIT;` wrapper, `CREATE TABLE IF NOT EXISTS`, idempotent.
- document_entities has: bibcode TEXT, entity_id INT, link_type TEXT,
  confidence REAL, match_method TEXT, evidence JSONB, harvest_run_id INT,
  tier SMALLINT, tier_version INT.
- 034 will add candidate_set_hash, model_version, expires_at, and partition
  by RANGE(expires_at) with tier default 5.

## pyproject / pytest

- pytest 8 + pytest-asyncio 1.3.0 installed but NOT configured (no asyncio_mode).
- Safest path: use `asyncio.run()` directly in sync test functions (no asyncio
  plugin dependency). That's what we'll do for portability.

## ast_lint_resolver

- Forbidden patterns in jit modules unless the literal line has
  `# noqa: resolver-lint`.
- `document_entities_jit_cache` INSERT/UPDATE/DELETE all forbidden outside
  `src/scix/resolve_entities.py`.
- cache.py writes will use `# noqa: resolver-lint` as the instructions permit.
- Reads from `document_entities_jit_cache` are NOT in the forbidden pattern
  list (the lint only bans writes to the jit cache and the canonical MV SELECT).
  So `SELECT FROM document_entities_jit_cache` in cache.py is lint-clean.

## Test DB

- `dbname=scix_test` is reachable. We apply migration 034 once inside the
  cache test module (idempotent).
