# Research: u11-eval-harness

## Prior art in repo

- `scripts/audit_tier1.py` already has a working `wilson_95_ci(successes, total)` that matches spec (95/100 → [0.887, 0.978], tol ±0.005). Migrate / reuse.
- `migrations/028_entity_schema_hardening.sql` adds `document_entities.tier SMALLINT NOT NULL DEFAULT 0`. PK is (bibcode, entity_id, link_type, tier).
- `migrations/033_fusion_mv.sql` defines `document_entities_canonical` MV (M8). Reads forbidden by AST lint unless `# noqa: resolver-lint`. Per unit instructions, we must NOT read from it — read base `document_entities` directly.
- `tests/helpers.py` → `get_test_dsn()` enforces SCIX_TEST_DSN and blocks production.
- `scripts/ast_lint_resolver.py` bans INSERT/UPDATE/DELETE on document_entities and SELECT FROM document_entities_canonical outside `src/scix/resolve_entities.py`. Reads on base `document_entities` are allowed.

## Dependencies

- `pyproject.toml` does NOT have `sklearn` or `anthropic`. Adding either is out of scope — implement Cohen's kappa manually (trivial formula) and gate the Anthropic client behind a stub. Tests never touch the API.
- psycopg / pytest / libcst are present.

## Key design decisions

1. **Migration 035** matches spec primary key exactly: `(tier, bibcode, entity_id, annotator)`.
2. **Wilson CI** lives in `src/scix/eval/wilson.py` and is re-exported / imported by `scripts/audit_tier1.py`-style callers; keep the exact existing function semantics.
3. **Cohen's kappa**: standard formula `κ = (po - pe) / (1 - pe)` over a shared label set. Handles degenerate `pe = 1` case by returning `1.0` when labels are identical, otherwise `0.0`.
4. **Sampler**: SELECT tier, bibcode, entity_id, (optional confidence) FROM document_entities ORDER BY random() per tier with LIMIT n_per_tier. Simpler and correct for our analytics read path. No canonical MV read.
5. **LLM judge**: Real Anthropic call only if `ANTHROPIC_API_KEY` is set AND `use_real=True` passed. Default stub returns deterministic labels (cycle over 'correct'/'incorrect'/'ambiguous') for reproducibility.
6. **run_audit.py --fixture**: Creates a temporary (papers + entities + document_entities) fixture in scix_test, samples, writes `build-artifacts/eval_report.md` with Wilson 95% CIs per tier, cleans up. Keeps the same pattern as `scripts/audit_tier1.py`.

## Files in scope

1. `migrations/035_entity_link_audits.sql`
2. `src/scix/eval/__init__.py`
3. `src/scix/eval/wilson.py`
4. `src/scix/eval/audit.py`
5. `src/scix/eval/llm_judge.py`
6. `scripts/run_audit.py`
7. `tests/test_audit.py`
8. `tests/test_llm_judge.py`
