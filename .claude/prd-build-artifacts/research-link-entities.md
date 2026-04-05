# Research: link-entities

## Findings

### extractions table (001_initial_schema.sql)

- Columns: id SERIAL PK, bibcode TEXT (FK papers), extraction_type TEXT, extraction_version TEXT, payload JSONB, created_at TIMESTAMPTZ
- Unique constraint: (bibcode, extraction_type, extraction_version)
- Payload structure for entity extraction: `{"entities": ["mention1", "mention2", ...]}` per extraction_type (methods, datasets, instruments, materials)

### document_entities table (021_entity_graph.sql)

- Columns: bibcode TEXT, entity_id INT (FK entities), link_type TEXT, confidence REAL, match_method TEXT, evidence JSONB, harvest_run_id INT
- PK: (bibcode, entity_id, link_type)
- No FK on bibcode (papers may not be ingested yet)

### entities table (021_entity_graph.sql)

- Columns: id SERIAL PK, canonical_name TEXT, entity_type TEXT, discipline TEXT, source TEXT, harvest_run_id INT, properties JSONB, created_at, updated_at
- Unique: (canonical_name, entity_type, source)
- Has entity_aliases table with lower(alias) index

### EntityResolver

- Does NOT exist yet. Must be built as part of this module or as a simple inline resolver.
- Resolution strategy: match mentions against entities table (canonical_name) and entity_aliases table (alias), case-insensitive.

### Normalization (src/scix/normalize.py)

- Deterministic pipeline: NFKC + lowercase + punctuation normalization + alias resolution + whitespace collapse
- Can be used to normalize mention strings before matching against entities.

### DB patterns (src/scix/db.py)

- `get_connection(dsn)` returns psycopg.Connection
- Chunked commits pattern used throughout (e.g. extract.py load_results_to_db)

### Extraction payload structure

- extract.py stores per-type: `{"entities": ["mention1", "mention2"]}`
- extraction_type values: "methods", "datasets", "instruments", "materials"
- The task spec says extraction_type='entity_extraction_v3' with payload like `{"instruments": [...], "datasets": [...], ...}`
- We'll support BOTH formats for robustness.
