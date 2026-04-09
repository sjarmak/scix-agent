# Research: Citation Intent Classifier

## Codebase Patterns

### Data structures

- `CitationContext` in `src/scix/citation_context.py` already has an `intent: str | None = None` field
- Frozen dataclasses used throughout (`@dataclass(frozen=True)`)
- Type annotations on all signatures

### Anthropic API usage (extract.py)

- Uses `anthropic.Anthropic` client, instantiated with `ANTHROPIC_API_KEY` env var
- System prompt + messages + tool_use schema pattern
- Batch API via `client.messages.batches.create()`
- For single messages: `client.messages.create(model=..., system=..., messages=..., max_tokens=...)`

### DB patterns (db.py)

- `get_connection(dsn)` for connections
- psycopg (v3) used throughout
- Cursor context managers

### Test patterns (test_extract.py)

- `unittest.mock.MagicMock` for mocking clients and connections
- `pytest` with `tmp_path` fixture
- Class-based test organization
- Frozen dataclass immutability tests

### citation_contexts table

- Has columns: source_bibcode, target_bibcode, context_text, char_offset, intent
- `intent` column already exists (nullable)

### SciCite dataset

- HuggingFace datasets: `allenai/scicite`
- 3 labels: background (0), method (1), result (2)
- Need to map "result" -> "result_comparison" for our schema

### SciBERT

- Model: `allenai/scibert_scivocab_uncased`
- Fine-tune with `transformers.Trainer` for sequence classification
