# Section Schema Contract

**Status:** Stable (PRD R14).
**Owner:** full-text ingestion working group.
**Related:** `papers_fulltext` table, `src/scix/sources/ar5iv.py`, `src/scix/sources/s2orc.py`, `src/scix/sources/ads_body_regex.py`, `docs/ADR/006_arxiv_licensing.md`.

## Problem

SciX ingests paper body text from three independent parsers:

1. **ar5iv** — XML from `ar5iv.labs.arxiv.org` (LaTeXML-derived HTML/XML).
2. **s2orc** — Semantic Scholar Open Research Corpus JSON with typed section annotations.
3. **ads_body_regex** — ADS raw body text with no structural markup; sections inferred from heading numbering.

Each parser has its own native notion of "section depth." Without a shared contract, downstream consumers (section-aware retrieval, table-of-contents reconstruction, section-scoped BM25) cannot treat the three sources uniformly. This document pins the canonical section schema that every parser MUST produce when writing `papers_fulltext.sections`.

## Canonical shape

`papers_fulltext.sections` is a PostgreSQL `JSONB` column holding a JSON **array** of section objects. Each element is a JSON **object** with exactly these four keys:

| Key       | JSON type | Semantics                                                                                |
| --------- | --------- | ---------------------------------------------------------------------------------------- |
| `heading` | string    | Section heading text, trimmed of leading/trailing whitespace. **Never empty.**           |
| `level`   | integer   | Section depth. `1` = top-level, `2` = subsection, `3` = subsubsection. Bounded to `1..3`. |
| `text`    | string    | Body text of the section (heading excluded). May be empty for heading-only sections.     |
| `offset`  | integer   | Zero-based character index of the section's start in the original source text. `>= 0`.   |

No additional keys are permitted. Unknown keys MUST be dropped by the parser before write.

### Level semantics

- `level = 1` — a top-level section. Canonical examples: `"1. Introduction"`, `"Methods"`, `"Conclusion"`, `"Abstract"` (when emitted as a section), `"References"`.
- `level = 2` — a subsection under a top-level section. Canonical examples: `"1.1. Background"`, `"2.3 Data Processing"`, `"Methods — Sample Selection"` when the parent is a level-1 `"Methods"`.
- `level = 3` — a subsubsection. Canonical examples: `"1.1.1 Instrumentation"`, `"2.3.4 Calibration Pipeline"`.

Anything deeper than three levels MUST be flattened to `level = 3`. Parsers that encounter `h4`/`h5`/`h6` or four-numeric-segment headings clip to `3` rather than emit `4+`. This matches empirical distribution in astronomy/NASA-SMD papers (>99% of observed section depth is `≤ 3`) and keeps the CHECK constraint small.

### Offset semantics

`offset` is a character index, not a byte offset, into **the original source text** passed to the parser — the same string used to populate `papers_fulltext.body`. It is the index of the first character of the section (conventionally, the first character of the heading line). It is **not** an offset into `text`, and it is **not** a BibTeX-style line number.

Consumers that want to reconstruct section boundaries for highlight-rendering can compute `[offset, next_offset)` slices against `body`. The final section's implicit end is `len(body)`.

### Heading semantics

`heading` is trimmed (no leading/trailing whitespace, no trailing numbering punctuation like a lone `.`), preserves internal whitespace as a single space, and is never the empty string. Parsers MUST drop sections whose heading would be empty after trimming; such fragments are not emitted. Numbering prefixes (`"1."`, `"2.3"`) MAY be retained as part of `heading` when present in the source — the contract does not require stripping them.

## Per-parser mapping

| Parser            | Source signal                              | Level mapping                                                                                                              | Offset computation                                                                                                | Notes                                                                                                                  |
| ----------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ar5iv`           | XML element tag names (`h1`, `h2`, `h3`+)  | `h1 → 1`, `h2 → 2`, `h3 → 3`, `h4+ → 3` (clipped)                                                                          | Character index of the heading element in the serialized source HTML                                              | Preserves ar5iv's LaTeXML numbering in `heading` (e.g. `"1. Introduction"`) when present                              |
| `s2orc`           | S2ORC `section_type` + nesting from `sections` array | Top-level entry in `sections` → `1`; nested `subsections` → `2`; `subsubsections` (or any deeper nesting) → `3` | Derived from S2ORC character span of the section header paragraph in the source body text                         | Fallback when `section_type` is absent: infer level from numeric heading prefix (same rule as `ads_body_regex`)        |
| `ads_body_regex`  | Heading-line regex + numbering depth       | Count of `.`-separated numeric groups in the heading prefix: `1` group → `1`, `2` → `2`, `3+` → `3`; unnumbered → `1` | Character index of the matched heading line in the raw ADS body text                                              | Unnumbered headings that match known top-level names (`Abstract`, `Introduction`, `Methods`, `Results`, `Discussion`, `Conclusion`, `References`, `Acknowledgements`) are emitted at `level = 1` |

All three parsers MUST produce sections in ascending `offset` order. Consumers rely on the array being monotonic in `offset` for O(n) section-boundary reconstruction.

## Assertion sketch (CHECK-compatible validator)

The contract is enforceable at the database layer with a single CHECK constraint backed by a pure PL/pgSQL or SQL function. The sketch below uses `jsonb_typeof` and `jsonb_array_elements` and is safe to add to migration scripts. It is intentionally deterministic and side-effect-free so it can live inside a table CHECK.

```sql
-- Validator: returns TRUE iff every element of `sections` conforms to
-- the R14 section schema contract. NULL sections are accepted (NULL
-- means "no structured sections were parsed"); an empty array is
-- accepted (means "parser ran, found no sections").
CREATE OR REPLACE FUNCTION is_valid_sections(sections jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT
    sections IS NULL
    OR (
      jsonb_typeof(sections) = 'array'
      AND NOT EXISTS (
        SELECT 1
        FROM jsonb_array_elements(sections) AS elem
        WHERE
          -- Each element must be an object
          jsonb_typeof(elem) <> 'object'
          -- heading: string, non-empty after trim
          OR jsonb_typeof(elem -> 'heading') <> 'string'
          OR btrim(elem ->> 'heading') = ''
          -- level: integer in 1..3
          OR jsonb_typeof(elem -> 'level') <> 'number'
          OR (elem ->> 'level')::int NOT BETWEEN 1 AND 3
          OR (elem ->> 'level')::numeric <> (elem ->> 'level')::int
          -- text: string (may be empty)
          OR jsonb_typeof(elem -> 'text') <> 'string'
          -- offset: non-negative integer
          OR jsonb_typeof(elem -> 'offset') <> 'number'
          OR (elem ->> 'offset')::int < 0
          OR (elem ->> 'offset')::numeric <> (elem ->> 'offset')::int
      )
    );
$$;

ALTER TABLE papers_fulltext
  ADD CONSTRAINT papers_fulltext_sections_valid
  CHECK (is_valid_sections(sections));
```

### What the validator enforces

- `sections` is either `NULL` or a JSON array (`jsonb_typeof(sections) = 'array'`).
- Every array element is a JSON object (`jsonb_typeof(elem) = 'object'`).
- `heading` is present, is a JSON string, and is non-empty after `btrim`.
- `level` is present, is a JSON number, is an integer (not a fractional value), and lies in `[1, 3]`.
- `text` is present and is a JSON string (empty string allowed).
- `offset` is present, is a non-negative integer.

### What the validator does **not** enforce

- **Monotonic `offset` ordering** across the array — the contract requires ascending `offset`, but a CHECK over `jsonb_array_elements` is unordered. Order is enforced at write time by the parsers and validated by the test suite.
- **Absence of extra keys** — a CHECK that forbids extra keys needs `jsonb_object_keys` per element and a subquery; omitted for performance. Parsers are responsible for dropping unknown keys; this is audited by the parser test suites.
- **`offset < length(body)`** — cross-column constraints would require a trigger; out of scope for the CHECK. Enforced by ingest-side assertions in `src/scix/sources/`.

These residual invariants are covered by application-level test fixtures (`tests/test_section_schema.py`) and the parser integration tests.

## Change management

Widening the contract (e.g. permitting `level = 4`, adding a `kind` key) requires:

1. A PRD amendment referencing R14.
2. Updating all three parsers in the same release.
3. Updating `is_valid_sections` in a forward-only migration.
4. Updating this document's "Canonical shape" table and the per-parser mapping.

Narrowing the contract (removing a key, tightening `level` bounds further) is a breaking change for stored data and requires a data-migration plan in addition to the above.
