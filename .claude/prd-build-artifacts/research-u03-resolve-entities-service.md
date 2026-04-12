# Research — u03-resolve-entities-service

## Existing codebase patterns

- `src/scix/link_entities.py` uses `@dataclass(frozen=True)` for `ResolverMatch` with `entity_id`, `confidence`, `match_method` — this is the natural shape for a single entity link.
- `src/scix/db.py` exposes `get_connection()` and `DEFAULT_DSN`. NOTE: the prompt claimed `is_production_dsn()` exists, but grep finds no such symbol. u03 is mock-only, so we do NOT depend on DB connectivity; safe to skip.
- `src/scix/field_mapping.py` uses `from __future__ import annotations`, type hints, and simple module layout — we follow that style.
- No `tests/conftest.py` currently. pytest configured in `pyproject.toml` with `pythonpath = ["src", "tests"]` so `from scix.resolve_entities import ...` works in tests.
- Existing tests use plain pytest functions, no fixtures beyond ad-hoc ones. No hypothesis usage found before this work unit.

## Dependency status

- `libcst` — NOT previously installed. Installed via `.venv/bin/python -m pip install libcst` (v1.8.6). Needs to be added to `pyproject.toml` dev extras.
- `hypothesis` — NOT previously installed. Installed (v6.151.12). Needs to be added to `pyproject.toml` dev extras.
- `psycopg` already a core dep; not used by u03 since lanes are mocked.

## M13 design constraints (from PRD)

- Single entry point: `resolve_entities(bibcode, context) -> EntityLinkSet`.
- Four internal lanes: static-core, jit_cache_hit, live_jit, local_ner.
- EntityLinkSet must be impossible to construct from outside the module — enforced via sentinel token in `_resolver_token.py`.
- AST lint must block direct writes/reads to `document_entities*` / `document_entities_canonical` outside `resolve_entities.py`.
- Property test: fix (bibcode, candidate_set_hash, model_version); all four lanes return equal entity-id sets; confidences within 0.01.
- Benchmark: per-lane p95 latency; static ≤5ms, jit_cache ≤25ms; live_jit/local_ner budgets also asserted.

## Mock strategy

- Static lane: in-module dict `_MockStaticStore[bibcode] -> frozenset[int]`.
- JIT cache: in-module dict `_MockJitCache[(bibcode, cset_hash, model_version)] -> frozenset[int]`.
- Live JIT: deterministic stub generator keyed on (bibcode, cset_hash) with 300ms sleep (reduced for benchmarks via injectable latency).
- Local NER: deterministic stub keyed on bibcode with 200ms sleep.

Property-test invariant: seed all four lanes with SAME entity-id set for a given key; add per-lane confidence noise ≤0.01 to show that the invariant detects intentional allowed drift.

## AST lint strategy

Use libcst `CSTVisitor` to walk every `.py` under `src/`. For each `Call` node look at:

1. Function is an Attribute ending in `.execute` (or `.executemany` / `.copy`).
2. First positional/named arg is a `SimpleString`, `ConcatenatedString`, or `FormattedString`.
3. Extract the literal text and match against forbidden patterns:
   - `INSERT INTO document_entities\b` (not `document_entities_canonical`)
   - `UPDATE document_entities\b`
   - `DELETE FROM document_entities\b`
   - `INSERT INTO document_entities_jit_cache`
   - `FROM document_entities_canonical` (read ban)
4. Exempt `src/scix/resolve_entities.py`.
5. Honor `# noqa: resolver-lint` trailing comment on the same line.

## Files to create

- `src/scix/_resolver_token.py` — sentinel class and module-private instance.
- `src/scix/entity_link_set.py` — `EntityLink` + `EntityLinkSet` dataclasses with token guard.
- `src/scix/resolve_entities.py` — the single entry point + four mock lanes + `EntityResolveContext`.
- `scripts/ast_lint_resolver.py` — libcst walker; CLI entry; returns 0/1.
- `tests/test_resolve_entities.py` — per-lane unit tests.
- `tests/test_resolve_entities_type_guard.py` — EntityLinkSet constructor TypeError.
- `tests/test_resolve_entities_invariant.py` — Hypothesis property test.
- `tests/test_ast_lint_resolver.py` — Planted-violation lint test.
- `tests/bench_resolve_entities.py` — Benchmark writing to build-artifacts/m13_latency.md.
