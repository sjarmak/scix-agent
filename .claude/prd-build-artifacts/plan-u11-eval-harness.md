# Plan: u11-eval-harness

## Goal

Ship M9 evaluation harness: audit table, stratified sampler, LLM-judge stub, Cohen's κ, Wilson CIs.

## Steps

1. **Migration 035**: create `entity_link_audits` table with exact PK `(tier, bibcode, entity_id, annotator)` and CHECK label constraint.
2. **src/scix/eval/**init**.py**: public exports (`sample_stratified`, `judge`, `cohens_kappa`, `wilson_95_ci`).
3. **src/scix/eval/wilson.py**: port `wilson_95_ci` from `scripts/audit_tier1.py` verbatim (tested 95/100 → [0.887, 0.978]).
4. **src/scix/eval/audit.py**:
   - `@dataclass(frozen=True) class AuditCandidate(tier, bibcode, entity_id, confidence)`.
   - `sample_stratified(conn, n_per_tier=125, seed=42)` — loop over DISTINCT tier, SELECT n_per_tier with ORDER BY random() per tier.
   - `write_audit_report(out_path, tiers_counts, judged_labels)` — markdown with per-tier Wilson CI.
5. **src/scix/eval/llm_judge.py**:
   - `@dataclass(frozen=True) class LinkRow(tier, bibcode, entity_id)`.
   - `@dataclass(frozen=True) class JudgeLabel(bibcode, entity_id, label, rationale)`.
   - `judge(links, *, use_real=False)` — deterministic stub cycling labels if no API key or use_real=False.
   - `cohens_kappa(human, judge)` — hand-rolled formula.
6. **scripts/run_audit.py**: `--fixture` path seeds a small fixture in scix_test (using noqa for writes to document_entities), runs sampler, judges with stub, writes `build-artifacts/eval_report.md`, cleans up. Gracefully skips if SCIX_TEST_DSN not set.
7. **tests/test_audit.py**:
   - Wilson CI unit tests (95/100 anchor, edge cases).
   - `sample_stratified` integration test against SCIX_TEST_DSN (skips if absent) — seeds tiers, samples, asserts per-tier count bounds.
   - `write_audit_report` renders expected markdown snippets.
8. **tests/test_llm_judge.py**:
   - Stub `judge()` returns one label per input link, all in allowed set.
   - `cohens_kappa` on known synthetic labels: perfect agreement → 1.0; total disagreement (2 labels) → -1.0; no agreement (mixed) → known value.
   - `cohens_kappa` on the classic 2x2 example (a=10, b=5, c=15, d=20, total=50, pa=0.6, pe=0.52, κ≈0.167).

## Constraints

- NEVER read from `document_entities_canonical` — base table only. AST lint enforces.
- All writes to document_entities in the fixture helper use `# noqa: resolver-lint`.
- No new PyPI deps.
- Must pass `python scripts/ast_lint_resolver.py src` exit 0.
