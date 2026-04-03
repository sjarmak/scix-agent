# Research: Extraction Pipeline

## Codebase Patterns

### Database

- `db.py`: `get_connection(dsn, autocommit)` returns `psycopg.Connection`
- `IngestLog` class tracks progress per file with start/finish/update_counts/mark_failed
- `DEFAULT_DSN` from `SCIX_DSN` env var

### Extractions Table (001 + 009)

- Schema: `id SERIAL PK, bibcode TEXT REFERENCES papers, extraction_type TEXT, extraction_version TEXT, payload JSONB, created_at TIMESTAMPTZ`
- Migration 009 adds: `UNIQUE (bibcode, extraction_type, extraction_version)` — enables ON CONFLICT DO UPDATE

### Embed Pipeline Pattern (embed.py)

- Frozen dataclasses for inputs
- Server-side cursor for streaming reads
- Separate read/write connections
- Chunked writes with COPY or individual INSERTs
- Module-level logging via `logging.getLogger(__name__)`

### CLI Pattern (scripts/embed.py, scripts/ingest.py)

- `sys.path.insert(0, ...)` for src/ discovery
- argparse with `--dsn`, `--batch-size`, `-v/--verbose`
- `logging.basicConfig` in main()
- `if __name__ == "__main__": main()`

### Test Pattern (test_embed.py)

- Classes grouping related tests
- `pytest.fixture(autouse=True)` for setup/teardown
- `unittest.mock.patch` for external deps
- No DB required for unit tests

## Key Decisions

- extraction_log: No dedicated table exists. Will track via a simple approach (log to the JSONL checkpoint file naming + logging)
- Anthropic Batches API: Uses `client.messages.batches.create()`, `.retrieve()`, results via `.results()`
- ON CONFLICT target: `(bibcode, extraction_type, extraction_version)` per migration 009
- Extraction types: methods, datasets, instruments, materials
- Tool-use schema requested: will define as Anthropic tool schema
