# Research — u06 tier1 keyword match

## Schemas (verified via psql against `scix_test`)

- `papers.keywords` is `text[]` (array of keyword strings). Unnest required.
- `papers.arxiv_class` is also `text[]`. For stratification we will use the
  first element (or `NULL`) as the bucket key — scale-safe arbitrary choice.
- `entities`: `id int PK`, `canonical_name text`, `entity_type`, `source`,
  `discipline`, `ambiguity_class`, `link_policy`. Index
  `idx_entities_canonical_lower btree (lower(canonical_name))` — matches our
  lower(lower(...)) join.
- `entity_aliases`: `(entity_id, alias)` PK, `alias_source`. Index
  `idx_entity_aliases_lower btree (lower(alias))`.
- `document_entities`: PK `(bibcode, entity_id, link_type, tier)`, has
  `tier smallint not null default 0`, `tier_version int not null default 1`,
  `confidence real`, `match_method text`, `evidence jsonb`.

## Existing code

- `src/scix/db.py::get_connection(dsn)` — used for connection acquisition.
- `src/scix/link_entities.py` — existing LLM-mention-based linker;
  writes with `link_type='extracted_mention'`. It is NOT tier=1 keyword and
  is separate from u06.
- `scripts/link_entities.py` — CLI entrypoint for the existing linker.
- `tests/helpers.py::get_test_dsn()` — safety helper that refuses production DSN.
- `tests/test_migrations.py` — pattern for module-scoped DSN fixture and
  autouse schema setup.

## Empty scix_test

`scix_test` currently has 0 papers / entities / aliases / document_entities,
so the test must seed its own fixture data (within a transaction or
cleaned up afterward).

## Acceptance-criteria tie-backs

- AC1: single `INSERT ... SELECT ... ON CONFLICT DO NOTHING` SQL pass over
  `papers JOIN unnest(keywords) JOIN entities/entity_aliases` writing
  `tier=1, link_type='keyword_match', confidence=1.0`.
- AC2: seeded fixture ≥10 papers and ≥20 entities producing ≥5 tier=1 rows.
- AC3: `scripts/audit_tier1.py` — stratified 200 sample across
  `(source, arxiv_class_first)` buckets, columns
  `bibcode, entity_id, canonical_name, source, arxiv_class, label_placeholder`,
  Wilson 95% CI helper.
- AC4: `tests/test_tier1.py` — end-to-end against fixture DB, writes rows,
  generates audit markdown, Wilson CI function unit test
  (`95/100 → [0.887, 0.978]`).

## Wilson score interval (reference)

For `p̂ = x/n`, `z = 1.96`:

```
center = (p̂ + z²/(2n)) / (1 + z²/n)
half   = z * sqrt((p̂(1-p̂) + z²/(4n))/n) / (1 + z²/n)
[lo, hi] = [center - half, center + half]
```

For 95/100: center ≈ 0.938, lo ≈ 0.887, hi ≈ 0.970–0.978 depending on
rounding. The spec's target is `[0.887, 0.978]`; verify precision with a
tolerance of ~0.005 in the assertion.

## Build-artifacts target

`build-artifacts/` exists at repo root; `tier1_audit.md` will be written
into that directory. Mkdir defensively in the audit script.
