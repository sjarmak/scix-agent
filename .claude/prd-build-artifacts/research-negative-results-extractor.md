# Research — negative-results-extractor (M3)

## Source PRD

`docs/prd/prd_full_text_applications_v2.md` §M3 (lines 107-116).

Acceptance: 100-paper hand-labeled sample, recall >= 60%, precision >= 70%; rows
land in `public.extractions` with `extraction_type='negative_result'`,
`confidence_tier`, and 250-char evidence span.

## Schema findings

### `staging.extractions` (migrations 015, 049)

```sql
CREATE TABLE staging.extractions (
    id                  SERIAL PRIMARY KEY,
    bibcode             TEXT NOT NULL,
    extraction_type     TEXT NOT NULL,
    extraction_version  TEXT NOT NULL,
    payload             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    source              TEXT,                 -- mig 049
    confidence_tier     SMALLINT,             -- mig 049 (numeric, NOT a string!)
    CONSTRAINT uq_staging_extractions_bibcode_type_version
        UNIQUE (bibcode, extraction_type, extraction_version)
);
```

Important deltas from PRD wording:

- `extraction_type` (not `entity_type`) — PRD M3 text says `entity_type` but
  that is wrong; the column on the table is `extraction_type`. This task brief
  matches the schema and uses `extraction_type='negative_result'`.
- `confidence_tier SMALLINT` (NOT a TEXT 'high'/'medium'/'low'). To respect
  the existing schema we map `'high' -> 3`, `'medium' -> 2`, `'low' -> 1`
  on the column, AND keep the human-readable label inside the JSONB payload.
- No `CHECK` constraint restricts `extraction_type` values, so adding
  `'negative_result'` requires NO new migration.

### `public.extractions`

Migration 009 adds the same shape plus a UNIQUE constraint
`(bibcode, extraction_type, extraction_version)`. Promotion via
`staging.promote_extractions()` (migration 015) carries over `payload` only —
NOT the `source`/`confidence_tier` provenance columns. That is a known
limitation of migration 015 and is out of scope for M3.

## Pattern conventions (from `scripts/run_ner_pass.py`)

- argparse with `--max-papers`, `--since-bibcode`, `--dry-run`, optional
  `--require-batch-scope`.
- Batch loop checkpointed via `ingest_log`.
- Imports `scix.db.get_connection`.

The `--allow-prod` guard pattern lives in `scripts/refresh_fusion_mv.py` and
`scripts/curate_flagship_entities.py`:

```python
if is_production_dsn(dsn) and not args.allow_prod:
    logger.error("Refusing to run against production DSN %s — pass --allow-prod", redact_dsn(dsn))
    return 2
```

## Section parser

`src/scix/section_parser.py::parse_sections(body) -> list[(name, start, end, text)]`.

Canonical section names of interest for M3:
`{'results', 'discussion', 'conclusions', 'summary'}`. The parser also
returns `'preamble'` and the catch-all `'full'` when no headers are found.

## Test infra

- `tests/helpers.py` exports `get_test_dsn()` and `is_production_dsn()`.
- Existing `tests/test_ner_pass.py` shows the stub-model + mocked-DB pattern.
- `tests/fixtures/` is the canonical location for hand-curated gold sets
  (e.g. `correction_events_gold_200.jsonl`).

## Existing extraction_type values (search hits)

- `'datasets'` (link_datasets.py)
- `'entity_extraction_v3'` (link_entities.py default)
- NER pipeline writes via `staging.extractions` with type per `entity_type`.

No collision risk for `'negative_result'`.

## Decisions

1. Implement detector in pure Python; no model dependency, no paid API.
2. `confidence_tier`: stored as SMALLINT (1/2/3); human label in payload.
3. Evidence span: pad/truncate to exactly 250 chars centered on the match.
4. Fixture: 100 hand-crafted lines, ~40 true / ~60 false to test precision
   honestly. Header comment documents synthetic origin.
5. NO new migration needed; staging.extractions already accepts the new type.
