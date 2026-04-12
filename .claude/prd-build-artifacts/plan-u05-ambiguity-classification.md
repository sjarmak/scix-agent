# Plan — u05 Ambiguity Classification

## 1. Dependency

- Add `wordfreq>=3.1` to `[project].dependencies` in `pyproject.toml`.

## 2. `src/scix/ambiguity.py` (pure module)

Expose:

- `AmbiguityClass = Literal["banned", "homograph", "domain_safe", "unique"]`
- Constants: `ZIPF_BANNED_THRESHOLD = 3.0`, `SHORT_NAME_MAX_LEN = 2`, `DOMAIN_SAFE_MIN_LEN = 6`.
- `def is_banned_name(name: str) -> bool` — True if `len(name.strip()) <= 2` OR `zipf_frequency(name.lower(), "en") >= 3.0`.
- `def classify(canonical_name, aliases, source_count, collision_count) -> AmbiguityClass`:
  1. If `is_banned_name(canonical)` or any alias is banned → `"banned"`.
  2. If `collision_count > 0` → `"homograph"`.
  3. If `len(canonical) >= 6` and `collision_count == 0` and `source_count == 1` → `"domain_safe"`.
  4. Else → `"unique"`.

No DB, no psycopg, no I/O. Fully unit-testable.

## 3. `scripts/classify_entity_ambiguity.py`

CLI with argparse:

- `--dsn` (default `SCIX_TEST_DSN` env or `dbname=scix_test`)
- `--batch-size` (default 1000)
- `--audit-out` (default `build-artifacts/ambiguity_audit.md`)
- `--dry-run` flag

Flow:

1. Connect (NOT autocommit for the UPDATE; autocommit-safe for reads).
2. **Pass 1 — index**:
   - Query `SELECT id, canonical_name, source FROM entities` → build `id → (canonical_lower, source)`.
   - Query `SELECT entity_id, alias FROM entity_aliases` → build `id → [alias_lower, ...]`.
   - Build `name_to_ids: dict[str, set[int]]` covering all lowercased canonical names and aliases.
   - Progress: print counts every 10K rows.
3. **Pass 2 — classify**:
   - For each entity, compute:
     - `names_for_entity = {canonical_lower} | set(aliases_lower)`
     - `collision_ids = union(name_to_ids[n] for n in names_for_entity) - {self.id}`
     - `collision_count = len(collision_ids)`
     - `collision_sources = {entities[i].source for i in collision_ids} | {self.source}`
     - `source_count = len(collision_sources)`
   - Call `classify(canonical, aliases, source_count, collision_count)`.
   - Accumulate `(ambiguity_class, id)` tuples.
   - Every `batch_size` rows, run `cur.executemany("UPDATE entities SET ambiguity_class = %s::entity_ambiguity_class WHERE id = %s", batch)` and print progress.
4. **Final COMMIT**.
5. **Summary query**: `SELECT ambiguity_class, count(*) FROM entities GROUP BY 1 ORDER BY 1` → print to stdout.
6. **Audit report**:
   - Ensure `build-artifacts/` dir exists.
   - For each of the 4 classes: `SELECT id, canonical_name, entity_type, source FROM entities WHERE ambiguity_class = %s ORDER BY random() LIMIT 50`.
   - Write markdown: 4 sections, each with a table (id, canonical_name, entity_type, source). Top-of-file summary (total counts per class).

## 4. `tests/test_ambiguity.py`

**Unit tests (no DB)**:

- `test_banned_short` — 'a' (1 char), 'Hi' (2 chars) → banned.
- `test_banned_common_word` — 'the', 'The', 'CAT' → banned (Zipf ≥ 3).
- `test_homograph` — non-banned name, collision_count=1 → homograph.
- `test_domain_safe` — 'GALFA-HI' (8 chars), collision_count=0, source_count=1 → domain_safe.
- `test_unique_short_non_banned` — 'XYZ' (3 chars, Zipf ≈ 0), collision_count=0, source_count=1 → unique (fails ≥6 rule).
- `test_unique_multi_source` — 8-char unique name but source_count=2 → unique (fails single-source rule).
- `test_banned_precedes_homograph` — 'the' with collision_count=5 → banned (order of precedence).
- `test_alias_triggers_banned` — canonical='Foo123', aliases=['the'] → banned.

**Integration test** (`@pytest.mark.integration`, gated on `get_test_dsn()`):

- Fixture: create 5–6 seeded entities in `scix_test`:
  - `('the', 'concept', 'uat')` — banned
  - `('HST', 'facility', 'ads')` and `('HST', 'facility', 'wikidata')` — HST collision → both homograph
  - `('GALFA-HI', 'survey', 'uat')` — domain_safe
  - `('Mars', 'body', 'wikidata')` — banned (common word, Zipf ≥ 3)
  - `('XYZ', 'unknown', 'uat')` — unique (3 chars, non-banned, no collision)
- Seed an alias 'HST' on one of them to exercise alias collision path.
- Run the classifier via `subprocess` or by importing and calling `main(dsn=...)`.
- Assert `SELECT ambiguity_class, count(*) ... GROUP BY 1` returns 4 non-null buckets.
- Assert each seeded entity has the expected class.
- Clean up rows in a `try/finally`.

## 5. Run + audit

- `SCIX_TEST_DSN=dbname=scix_test pytest tests/test_ambiguity.py -v`
- The integration test itself populates the 4 buckets and invokes the classifier which writes `build-artifacts/ambiguity_audit.md` — that satisfies AC5.

## 6. Risks / mitigations

- **`scix_test` already has `build-artifacts/` in repo root?** — check / create.
- **Large collision groups** — the union-of-sets approach is O(total_names) once, O(k) per entity. Fine for scix_test (~zero rows today) and safe for full 100K+ entity prod case.
- **UPDATE perf** — `executemany` at 1000-row batches is sufficient; if we ever need more, we can switch to the `UPDATE entities SET ambiguity_class = v.cls FROM (VALUES (1, 'unique'), ...) AS v(id, cls) WHERE entities.id = v.id` form. Not needed now.
- **Enum casting** — must cast text → `entity_ambiguity_class` in the UPDATE statement.
