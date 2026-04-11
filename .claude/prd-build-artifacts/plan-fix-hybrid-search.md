# Plan: fix-hybrid-search

## Changes to src/scix/search.py

### 1. Change hybrid_search() default model_name from "specter2" to "indus"

- Line 279: `model_name: str = "specter2"` -> `model_name: str = "indus"`
- Update docstring references from SPECTER2 to INDUS

### 2. Skip OpenAI signal entirely

- In hybrid_search(), before running OpenAI vector search, check if model has rows in paper_embeddings
- Add helper `_model_has_embeddings(conn, model_name)` that does a fast EXISTS check
- If text-embedding-3-large has 0 rows, skip the vector_search call and log a debug message
- This replaces the current approach of always trying and catching exceptions

### 3. Add cardinality-aware query routing

- Add `_estimate_filter_selectivity(conn, filters)` function:
  - Count matching rows with the filter WHERE clause using an EXPLAIN-based estimate or a fast COUNT with LIMIT
  - Compare against total corpus size (from pg_class reltuples)
  - Return selectivity ratio (0.0 to 1.0)
- Add `SELECTIVITY_THRESHOLD = 0.01` constant (1%)
- Add `_filter_first_vector_search()` function:
  - CTE: first get bibcodes matching filters
  - Then brute-force cosine similarity on just that subset (no HNSW index)
  - Returns SearchResult like vector_search()
- In hybrid_search(), before calling vector_search() with filters:
  - If filters present, estimate selectivity
  - If selectivity < 1%, use filter-first path instead of HNSW iterative scan

### 4. Update vector_search() default model_name

- Line 158: `model_name: str = "specter2"` -> `model_name: str = "indus"`

## Tests to add in tests/test_search.py

### 5. test_hybrid_search_default_model (unit test)

- Import hybrid_search, inspect default parameter
- Verify model_name default is "indus"

### 6. test_cardinality_routing (unit test with mock)

- Mock \_estimate_filter_selectivity to return < 0.01
- Verify filter-first path is used (mock \_filter_first_vector_search)
- Or: test the estimation function directly with a mock connection

### 7. test_openai_signal_skipped_when_no_rows

- Verify that when openai_embedding is provided but model has 0 rows, vector_search is not called for OpenAI
