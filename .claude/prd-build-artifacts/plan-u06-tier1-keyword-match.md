# Plan — u06 tier1 keyword match

## 1. `scripts/link_tier1.py`

- argparse: `--db-url`, `--dry-run`, `--verbose`.
- Single SQL pass wrapped in a CTE-based `INSERT ... ON CONFLICT DO NOTHING`.
- Source matches come from two subqueries UNION'd:
  1. `papers p, unnest(p.keywords) k JOIN entities e ON lower(e.canonical_name) = lower(k)`
  2. `papers p, unnest(p.keywords) k JOIN entity_aliases ea ON lower(ea.alias) = lower(k) JOIN entities e ON e.id = ea.entity_id`
- Insert columns: `bibcode, entity_id, link_type='keyword_match',
tier=1, tier_version=1, confidence=1.0, match_method='keyword_exact_lower',
evidence=jsonb_build_object('keyword', k, 'match_source', 'canonical'|'alias')`.
- ON CONFLICT target: `(bibcode, entity_id, link_type, tier) DO NOTHING`.
- Return count of new rows (from `cur.rowcount`).
- Add `# noqa: resolver-lint` comment on the INSERT line.
- `main()` opens connection, runs SQL, prints summary, commits (unless dry-run).

## 2. `scripts/audit_tier1.py`

- argparse: `--db-url`, `--sample-size` (default 200), `--output`
  (default `build-artifacts/tier1_audit.md`).
- Pull all tier=1 rows joined with `papers` (bibcode, arxiv_class[1]) and
  `entities` (canonical_name, source).
- Compute stratification key:
  - Try `(source, arxiv_class_first)`.
  - If only one source in the data, fall back to `source` alone.
  - If no buckets, single-bucket "all".
- Proportional allocation of `sample_size` across buckets; each bucket gets
  `ceil(n_in_bucket / total * sample_size)` capped at rows in bucket.
  Then shuffle with a fixed seed and truncate to sample_size.
- Fill `label_placeholder = "unlabeled"` for each sample row.
- Write markdown table with columns:
  `bibcode | entity_id | canonical_name | source | arxiv_class | label_placeholder`.
- Wilson CI helper: `wilson_95_ci(successes: int, total: int) -> tuple[float,float]`.
- Compute CI over the current sample using `successes=0` as a placeholder
  (no labels yet). Also include a worked example in the output:
  `wilson_95_ci(95, 100)` to demonstrate the function.
- Write output to an mkdir'd path.

## 3. `tests/test_tier1.py`

- Fixture `dsn` — uses `tests.helpers.get_test_dsn()`, skips if not set.
- Fixture `seeded_db` — opens connection, TRUNCATEs
  `document_entities, entity_aliases, entities, papers` in dep order,
  inserts:
  - 12 papers with varied arxiv_class + keywords (some matching, some not).
  - 25 entities across two sources (`unit_test_a`, `unit_test_b`) with
    canonical_names matching some keywords; a couple of aliases in
    `entity_aliases`.
  - Commits, yields connection, then TRUNCATEs again after test.
- Test `test_wilson_95_ci_known_input` — `wilson_95_ci(95, 100)` within
  tolerance of `[0.887, 0.978]`.
- Test `test_link_tier1_end_to_end` — invokes the link script
  (via subprocess or direct function import) with the test DSN, asserts
  ≥5 rows written with `tier=1, link_type='keyword_match', confidence=1.0`.
- Test `test_audit_tier1_generates_markdown` — runs the audit on the
  seeded + linked data, asserts output file exists, contains header,
  has ≤200 rows, columns present.
- Refactor: expose a `run_tier1_link(conn) -> int` function in
  `scripts/link_tier1.py` and a `run_audit(conn, sample_size, output_path)`
  function in `scripts/audit_tier1.py` so tests can call directly.

## 4. Acceptance check

- AC1 ✓ via `run_tier1_link`.
- AC2 ✓ via seeded 12-paper / 25-entity fixture.
- AC3 ✓ via `run_audit` + wilson helper.
- AC4 ✓ via `pytest tests/test_tier1.py -v` with `SCIX_TEST_DSN`.
