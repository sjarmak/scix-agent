# Plan: entity-resolver

## Approach

Create `src/scix/entity_resolver.py` with a class-based resolver that queries the normalized entity graph tables (migration 021). Tests will mock `psycopg.Connection` and cursor to avoid requiring a live database.

## Data structures

1. **EntityCandidate** — frozen dataclass:
   - entity_id: int, canonical_name: str, entity_type: str, source: str
   - discipline: str | None, confidence: float, match_method: str

## EntityResolver class

- Constructor: takes `psycopg.Connection`
- `resolve(mention, *, discipline=None, fuzzy=False, fuzzy_threshold=0.3) -> list[EntityCandidate]`
- `resolve_batch(mentions, ...) -> dict[str, list[EntityCandidate]]`

## Resolution cascade

1. Exact canonical: `WHERE lower(canonical_name) = lower(mention)` -> confidence 1.0
2. Alias: `WHERE lower(alias) = lower(mention)` JOIN entities -> confidence 0.9
3. Identifier: `WHERE external_id = mention` JOIN entities -> confidence 0.85
4. Fuzzy (optional): pg_trgm `similarity() > threshold` -> confidence = similarity score

## Post-processing

- Discipline boost: +0.05 confidence for matching discipline, applied before dedup
- Dedup by entity_id: keep highest confidence per entity_id
- Sort: confidence DESC, canonical_name ASC

## Testing strategy

Mock psycopg connection/cursor. Define helper that configures mock to return predefined rows for specific SQL patterns. Test each match method, discipline ranking, dedup, batch, and case-insensitivity.
