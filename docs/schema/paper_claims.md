# `paper_claims` — nanopub-inspired claim provenance

Created by [migration 062](../../migrations/062_paper_claims.sql) as part of
the **nanopub-claim-extraction** PRD build.

## Purpose

`paper_claims` stores one row per scientific claim extracted from a paper's
full text. The table is modeled after the
[nanopublication](https://nanopub.net/) idea: each claim carries an explicit
provenance record (where it came from in the source paper, which model and
prompt produced it, when), so claims stay auditable and the extraction can be
re-run with newer models without losing history.

The table is the substrate for downstream MCP tools that surface claims, agree
/ disagree links between claims, and entity-grounded claim search.

## Provenance contract

The five-tuple

```
(bibcode, section_index, paragraph_index, char_span_start, char_span_end)
```

uniquely identifies the source span of every claim. The indices and offsets
are interpreted relative to
[`papers_fulltext`](../../migrations/041_papers_fulltext.sql):

- `bibcode` — FK to `papers(bibcode)`.
- `section_index` — zero-based index into the JSONB array
  `papers_fulltext.sections[]`. The element at this index is the section the
  claim was extracted from.
- `paragraph_index` — zero-based paragraph offset within
  `papers_fulltext.sections[section_index].text`. Paragraph splitting is
  whatever the extractor uses; the extractor must be deterministic so the
  contract holds across re-runs.
- `char_span_start` — inclusive character offset of the claim within
  `papers_fulltext.sections[section_index].text`.
- `char_span_end` — exclusive character offset of the claim within
  `papers_fulltext.sections[section_index].text`.

This contract is the integration point with the section retrieval MCP tool
(see [`mcp_tool_contracts.md`](../mcp_tool_contracts.md)). A consumer can take
a `paper_claims` row and recover the exact span of source text it came from
by indexing into `papers_fulltext.sections[section_index].text` with the
character span.

> **Re-extraction policy.** New extractions INSERT new rows (with a new
> `claim_id` and a new `extraction_model` / `extraction_prompt_version`).
> They do **not** UPDATE existing rows. This keeps the audit trail intact —
> if you want only the latest extraction, filter by
> `extraction_model + extraction_prompt_version`.

## Columns

| Column | Type | Nullable | Meaning |
|---|---|---|---|
| `claim_id` | `uuid` | no | Primary key. Defaults to `gen_random_uuid()`. |
| `bibcode` | `text` | no | Source paper. FK to `papers(bibcode)`. |
| `section_index` | `int` | no | Zero-based index into `papers_fulltext.sections[]`. |
| `paragraph_index` | `int` | no | Zero-based paragraph offset within the section text. |
| `char_span_start` | `int` | no | Inclusive char offset within the section text. |
| `char_span_end` | `int` | no | Exclusive char offset within the section text. |
| `claim_text` | `text` | no | Verbatim claim text from the paper. |
| `claim_type` | `text` | no | One of `factual`, `methodological`, `comparative`, `speculative`, `cited_from_other`. CHECK-constrained. |
| `subject` | `text` | yes | Optional structured-claim subject. |
| `predicate` | `text` | yes | Optional structured-claim predicate. |
| `object` | `text` | yes | Optional structured-claim object. |
| `confidence` | `real` | yes | Optional extractor confidence in `[0, 1]`. Semantics defined by `extraction_model` + `extraction_prompt_version`. |
| `extraction_model` | `text` | no | Model name + version that produced this claim (e.g. `claude-opus-4-7`). |
| `extraction_prompt_version` | `text` | no | Prompt template version (e.g. `v1`, `v2.1`). |
| `extracted_at` | `timestamptz` | no | Defaults to `now()`. |
| `linked_entity_subject_id` | `bigint` | yes | Optional link to `entities.id` for the subject. See note below. |
| `linked_entity_object_id` | `bigint` | yes | Optional link to `entities.id` for the object. See note below. |

### Note on `linked_entity_*_id`

Both `linked_entity_*_id` columns are `bigint` and are **not** declared
`REFERENCES entities(id)`. Two reasons:

1. `entities.id` is currently `SERIAL` (`int4`) per
   [migration 021](../../migrations/021_entity_graph.sql). We use `bigint` here
   so the column can absorb a future widening of `entities.id` without an
   ALTER on `paper_claims`.
2. Claim extraction runs **before** entity linking. We don't want claim
   INSERTs to fail when the linker hasn't yet produced an entity row for the
   subject / object. This mirrors the `document_entities.bibcode` decision in
   migration 021 (no FK because papers may not be ingested yet).

The trade-off is that referential integrity for these two columns is enforced
by the linker, not the database. Treat them as soft links.

### Constraints

- **Primary key:** `claim_id`.
- **Foreign key:** `bibcode` → `papers(bibcode)`.
- **Check constraint** `paper_claims_claim_type_check`:
  `claim_type IN ('factual', 'methodological', 'comparative', 'speculative', 'cited_from_other')`.
- **LOGGED-table guard:** the migration's terminal `DO $$ ... $$` block
  asserts `pg_class.relpersistence = 'p'` for `paper_claims` and raises
  otherwise — see [migration 041](../../migrations/041_papers_fulltext.sql)
  for the same pattern and the rationale (UNLOGGED tables are truncated on
  crash recovery; we lost 32M embeddings to that bug once already).

## Indexes

All five indexes use the `ix_paper_claims_*` naming convention and are
created with `IF NOT EXISTS` so re-running the migration is safe.

| Index | Type | Columns / Expression | Rationale |
|---|---|---|---|
| `ix_paper_claims_bibcode_section` | btree | `(bibcode, section_index)` | Lookup-by-paper, ordered by section. Drives the "show me claims from paper X" query and section-scoped backfill. |
| `ix_paper_claims_linked_entity_subject_id` | btree | `linked_entity_subject_id` | Reverse lookup from the entity graph: "which claims have entity E as subject?". |
| `ix_paper_claims_linked_entity_object_id` | btree | `linked_entity_object_id` | Reverse lookup from the entity graph: "which claims have entity E as object?". |
| `ix_paper_claims_claim_type` | btree | `claim_type` | Filter by claim type ("methodological claims only"). Low-cardinality (5 values) but cheap and useful as a composite key with bibcode / model filters. |
| `ix_paper_claims_claim_text_tsv` | GIN | `to_tsvector('english', claim_text)` | Full-text search over claim text. Drives natural-language claim search MCP tools without scanning the table. |

## Idempotency

The migration file uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT
EXISTS` throughout. A second run of
`psql $SCIX_TEST_DSN -f migrations/062_paper_claims.sql` is a no-op and exits
cleanly.

## Tests

Schema-level tests live in
[`tests/test_paper_claims_schema.py`](../../tests/test_paper_claims_schema.py).
They run against `scix_test` and assert: column types via
`information_schema.columns`, index presence via `pg_indexes`, the
`claim_type` CHECK fires on bogus values, a valid INSERT round-trips, and the
migration is idempotent.
