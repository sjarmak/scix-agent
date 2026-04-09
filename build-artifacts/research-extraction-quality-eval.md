# Research: extraction-quality-eval

## Key Findings

### Extractions Table Schema

- `bibcode TEXT NOT NULL`
- `extraction_type TEXT NOT NULL` (methods, datasets, instruments, materials)
- `extraction_version TEXT NOT NULL`
- `payload JSONB NOT NULL` — contains `{"entities": ["mention1", "mention2", ...]}`
- `source TEXT` (llm, metadata, ner, openalex, citation_propagation)
- `confidence_tier TEXT` (high, medium, low)
- `extraction_model TEXT`
- Unique on (bibcode, extraction_type, extraction_version)

### EntityResolver API

- `EntityResolver(conn)` — takes psycopg.Connection
- `resolve(mention, discipline=None, fuzzy=False, fuzzy_threshold=0.3)` — returns `list[EntityCandidate]`
- `resolve_batch(mentions, ...)` — returns `dict[str, list[EntityCandidate]]`
- `EntityCandidate` fields: entity_id, canonical_name, entity_type, source, discipline, confidence, match_method
- `MATCH_METHODS = frozenset({"exact_canonical", "alias", "identifier", "fuzzy"})`
- Cascade: exact_canonical -> alias -> identifier -> fuzzy (optional)

### Script Patterns (from harvest_gcmd.py)

- `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))` at top
- argparse for CLI args (--dsn, --dry-run, -v/--verbose)
- logging.basicConfig setup
- `from scix.db import get_connection`

### Test Patterns (from test_harvest_ascl.py)

- `sys.path.insert(0, ...)` for scripts/ and src/
- unittest.mock for DB and HTTP
- pytest classes grouping related tests

### Mentions Extraction from Payload

- payload is `{"entities": ["mention_string_1", "mention_string_2"]}`
- Each extraction_type row has its own payload with entity list
- Need to query all extraction rows for sampled bibcodes and flatten mentions
