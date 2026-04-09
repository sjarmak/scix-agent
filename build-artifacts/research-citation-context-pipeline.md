# Research: citation-context-pipeline

## Key Findings

### Target Table (migrations/011_citation_contexts.sql)

- `citation_contexts`: id (serial PK), source_bibcode, target_bibcode, context_text, char_offset, intent
- Indexes on (source_bibcode, target_bibcode) and (target_bibcode)
- No FK constraints (citations may reference papers outside corpus)

### Section Parser (src/scix/section_parser.py)

- `parse_sections(body) -> list[tuple[str, int, int, str]]` — (name, start, end, text)
- Returns `[("full", 0, len(body), body)]` when no headers found
- Canonical section names: introduction, methods, results, discussion, conclusions, etc.
- Can use char offsets to determine which section a citation marker falls in

### DB Helpers (src/scix/db.py)

- `get_connection(dsn=None, autocommit=False)` — opens psycopg connection
- `DEFAULT_DSN = os.environ.get("SCIX_DSN", "dbname=scix")`
- Uses psycopg (v3) throughout

### Field Mapping (src/scix/field_mapping.py)

- Papers have a `raw` JSONB column containing unmapped fields
- The `reference` array is stored in `raw` (not a dedicated column): `raw->'reference'`
- Reference is a list of bibcode strings, 0-indexed in the array
- The `body` column exists as a direct text field

### Ingest Pipeline Patterns (src/scix/ingest.py)

- Uses COPY via psycopg's `cur.copy()` for bulk writes
- Staging table pattern: COPY into temp table, then INSERT...ON CONFLICT
- Batch processing with configurable batch_size
- Frozen dataclasses used for IndexDef

### Data Model for Pipeline

- Source query: papers WHERE body IS NOT NULL AND raw::jsonb ? 'reference'
- Citation markers: [N] where N is 1-indexed, maps to reference[N-1]
- Also handle [N, M, ...] (comma-separated) and [N-M] (ranges)
