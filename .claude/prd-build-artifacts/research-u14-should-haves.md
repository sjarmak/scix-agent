# Research — u14-should-haves

## Scope

PRD §S1 citation-consistency, §S3 query-time entity expansion, §S5 researcher feedback, §M12 dual-lane MCP contract.

## Key findings

### citation_edges schema (migration 001)

```sql
CREATE TABLE citation_edges (
    source_bibcode TEXT NOT NULL,
    target_bibcode TEXT NOT NULL,
    PRIMARY KEY (source_bibcode, target_bibcode)
);
CREATE INDEX idx_cite_target ON citation_edges(target_bibcode);
```

Source cites target. For paper P citing N others, consistency(P, E) = |{c in cites(P) : c linked to E}| / |cites(P)|.

### document_entities (mig 021 + 028)

Columns: bibcode, entity_id, link_type, confidence, match_method, evidence, harvest_run_id, tier, tier_version. PK: (bibcode, entity_id, link_type, tier). Reads from this table are NOT forbidden by AST lint — only writes. Perfect for citation_consistency reads.

### AST lint (scripts/ast_lint_resolver.py)

Forbidden patterns (outside resolve_entities.py):

- `INSERT|UPDATE|DELETE ... document_entities` (negative lookahead excludes `_jit_cache`/`_canonical`)
- `INSERT|UPDATE|DELETE ... document_entities_jit_cache`
- `FROM document_entities_canonical`

READS from `document_entities` via SELECT/FROM are fine. `entity_link_disputes` is not scanned. No noqa needed for either.

### Migration numbering

Latest: 035_entity_link_audits.sql. u13 claims 036. So we use 037.

### Test DSN helper

`tests/helpers.py` exports `get_test_dsn()` returning `SCIX_TEST_DSN` if safe, else None. Tests skip if None. Pattern used by test_query_log.py.

### DEFAULT_DSN pattern

`src/scix/db.py` exposes `DEFAULT_DSN = os.environ.get("SCIX_DSN", "dbname=scix")` and `get_connection(dsn=None)`. `src/scix/query_log.py` implements `_resolve_dsn` which prefers `SCIX_TEST_DSN` — reuse pattern.

### Existing modules for style reference

- `src/scix/query_log.py` — thin DB write module with `_resolve_dsn` + optional `conn` override.
- `tests/test_query_log.py` — integration test using `get_test_dsn()` + psycopg fixture.

## Implementation notes

- **citation_consistency.py**: accepts optional `conn` injection so unit test can pass a psycopg connection to a seeded test DB. Uses plain SELECT over `citation_edges` JOIN `document_entities`. Closed-form unit test seeds 5 papers into `scix_test` and asserts the fraction.
- **query_expansion.py**: pure-Python numpy cosine similarity over in-memory entity embedding matrix. Deterministic. Documents pgvector production path as TODO.
- **feedback.py**: `report_incorrect_link` — simple INSERT into `entity_link_disputes` with optional `conn`.
- **migration 037**: ALTER TABLE document_entities ADD COLUMN citation_consistency REAL; CREATE TABLE entity_link_disputes. Idempotent.
