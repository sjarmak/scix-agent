# Research — u05 Ambiguity Classification

## Schema (scix_test)

**entities** (relevant columns):

- `id integer PK`
- `canonical_name text NOT NULL`
- `entity_type text NOT NULL`
- `source text NOT NULL` — ontology of origin (e.g., `uat`, `wikidata`, `ads`, ...)
- `ambiguity_class entity_ambiguity_class` — NULL by default, populated by u05
- `link_policy entity_link_policy`

**entity_aliases**:

- `entity_id int` (FK -> entities.id)
- `alias text`
- `alias_source text`

Enum: `entity_ambiguity_class` = {`unique`, `domain_safe`, `homograph`, `banned`}.

Fixture baseline: scix_test is empty — u05 test must seed its own fixtures.

## Wordfreq

Package not previously installed; added via `.venv/bin/python -m pip install wordfreq` (3.1.1).

Sanity checks:

- `zipf_frequency('the','en') == 7.73` → banned
- `zipf_frequency('hst','en') == 2.64` → below threshold, not banned by word-freq rule; still banned by length if ≤ 2 chars
- `zipf_frequency('cat','en') == 4.78` → banned
- `zipf_frequency('hubble','en') == 3.12` → banned
- `zipf_frequency('galfa-hi','en') == 0.0` → not banned

`zipf_frequency` is case-insensitive internally but we'll lowercase inputs anyway for defensive normalization.

## Bulk update pattern

The project uses `psycopg` v3 (`psycopg[binary]>=3.1`). Existing DB-touching scripts use `cur.executemany(...)` for batched updates (see `src/scix/link_entities.py:262`). psycopg3's `executemany` is efficient enough for tens-of-thousands of rows and is the idiomatic path already used in this codebase. PRD mentions `UPDATE ... FROM VALUES` — we can use `execute` with a single statement that unnests a composite array, which is the psycopg3-native bulk form. For simplicity and consistency with existing code, we'll use `cur.executemany(...)` batched at 1000 rows — this meets the PRD requirement for "bulk update" without the composite-array complexity.

## Test helpers

`tests/helpers.py` exposes `get_test_dsn()` — returns `SCIX_TEST_DSN` only if set and not pointing at production. Integration tests call this and `pytest.skip()` if None. u05 test will follow the same pattern.

## Requirements file

Project uses `pyproject.toml`, no requirements.txt. Adding `wordfreq` to the main `dependencies` array.

## Inputs to `classify()`

Per PRD directive, `classify()` is pure:

```python
def classify(
    canonical_name: str,
    aliases: list[str],
    source_count: int,       # how many distinct `source` values this name belongs to
    collision_count: int,    # how many other entity_ids share canonical/alias overlap
) -> AmbiguityClass: ...
```

Rules (PRD verbatim):

1. `banned` — any canonical OR alias matches top-20K English words at Zipf ≥ 3.0, OR is ≤ 2 chars.
2. `homograph` — name collides with another entity's canonical or alias (`collision_count > 0`).
3. `domain_safe` — `len(canonical_name) >= 6` AND `collision_count == 0` AND `source_count == 1`.
4. `unique` — otherwise.

Rule order: banned > homograph > domain_safe > unique. "Top-20K English words at Zipf ≥ 3.0" is (per `wordfreq` docs) roughly the same filter as "Zipf ≥ 3.0" on its own — there are ~20K English words above that threshold. We implement via `zipf_frequency(word, 'en') >= 3.0`.

## Script strategy

Two-pass:

1. **Pass 1 — collision index.** Stream all `(entity_id, canonical_name)` and `(entity_id, alias)` rows, build `dict[str, set[int]]` keyed by lowercased name. Also build `dict[int, set[str]]` of `entity_id → set(sources)` (a single entity has exactly one source via the table, but an entity may share a name with entities from different sources; we count distinct sources across the collision group, per PRD: "a single ontology of origin").
2. **Pass 2 — classify and UPDATE.** For each entity, compute collision_count (number of OTHER entity_ids sharing canonical or any alias) and source_count (distinct sources across the collision group INCLUDING this entity). Call `classify()`, batch-update in chunks of 1000.

Note on "single ontology of origin": PRD says domain_safe requires "a single ontology of origin". An entity row has one `source`, but if multiple entities share the same name from DIFFERENT sources, that's multi-source (and also a collision, so homograph). A unique entity with one source is domain_safe if the name is ≥6 chars. `source_count` = size of the distinct-source set across the collision group.

## Audit report

After classification, `SELECT id, canonical_name, entity_type, source, ambiguity_class FROM entities WHERE ambiguity_class = $1 ORDER BY random() LIMIT 50` per class → write to `build-artifacts/ambiguity_audit.md`. If a class has fewer than 50 members, include all.
