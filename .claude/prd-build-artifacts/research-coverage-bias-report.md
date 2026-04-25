# Research: coverage-bias-report (M1)

## PRD spec (docs/prd/prd_full_text_applications_v2.md M1)

- Compares full-text cohort (~14.9M) vs abstract-only (~17.5M).
- Facets: arxiv_class, year, citation count buckets, journal (bibstem), and
  community_semantic_medium (gracefully skipped if absent).
- Outputs: docs/full_text_coverage_analysis.md (extend, do not recreate)
  and results/full_text_coverage_bias.json.
- Acceptance: KL-divergence per facet + narrative on safe vs unsafe agent
  queries.

## Existing assets

- scripts/coverage_bias_analysis.py — produces counts/percents per facet
  via DB queries (year, arxiv_class, citation bucket, pub, doctype,
  database, field completeness, citation network completeness). Writes a
  full markdown report. Lacks: KL-divergence, JSON output, bibstem facet,
  community_semantic_medium facet, agent guidance section.
- docs/full_text_coverage_analysis.md — counts-only report, last
  generated 2026-04-20. Contains corpus summary, field completeness,
  citation completeness, and per-facet tables.
- tests/test_coverage_bias.py — mocks DB cursor for the existing script;
  pattern is reusable.

## Schema findings

- `papers` table: bibstem is ARRAY (TEXT[]). pub is the user-facing
  journal name string. arxiv_class is ARRAY. No community_semantic_*
  columns on papers.
- `paper_metrics` table HAS community_semantic_coarse / medium / fine
  (per migration 051). Need to JOIN papers to paper_metrics for the
  semantic facet.
- bibstem facet wants the unnested array (similar to arxiv_class).

## KL-divergence approach

- D_KL(P||Q) = sum_i P(i) log(P(i)/Q(i)).
- P = full-text distribution, Q = corpus prior (full corpus distribution).
- Use scipy.stats.entropy(p, q) for the production path; provide a pure
  arithmetic implementation in the script's helper for testability and
  for environments without scipy. Smooth zeros via Laplace add-epsilon
  (epsilon=1e-12) so log() never sees zero.

## scix-batch wrapping

- Read-only script (SELECT only, no writes). Production DB is fine to
  query from a normal shell, but on a 32M-paper table several of these
  GROUP BYs (especially LATERAL unnest) can sit at multi-GB working
  memory. Doc the scix-batch wrapper for the prod re-run path:
  `scix-batch python scripts/report_full_text_coverage_bias.py
   --json-out results/full_text_coverage_bias.json`.
- This unit does NOT need --allow-prod (no writes).

## Test infrastructure

- No tests/conftest.py — tests can stand alone.
- Pattern from tests/test_coverage_bias.py: use sys.path.insert to import
  the script module, mock matplotlib if missing.
- For this unit's tests we don't touch matplotlib (no figure generation
  in the new script — figures are already produced by the existing
  script). Pure unit tests on the KL helper + JSON-shape test using
  synthetic distributions, no DB required.

## Decision: extend by wrapping, not by rewriting

The new script `scripts/report_full_text_coverage_bias.py` will:

1. Reuse the dataclasses and DB query functions from
   `coverage_bias_analysis.py` (import them — DRY).
2. Add new facet collectors for bibstem (unnested) and
   community_semantic_medium (joined via paper_metrics).
3. Add a pure-Python KL-divergence helper.
4. Convert the collected DistributionRow lists into the required JSON
   shape and write to --json-out.
5. Append the "Agent guidance" narrative section to
   docs/full_text_coverage_analysis.md (idempotent — replace the
   section if it already exists; do not duplicate the existing
   counts-based content).
6. Provide a --dry-run mode that uses synthetic distributions (no DB
   touch) so AC2 can be verified offline.
