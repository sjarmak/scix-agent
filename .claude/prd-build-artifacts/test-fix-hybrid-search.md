# Test Results: fix-hybrid-search

## Test Run

- 52 tests in test_search.py: ALL PASSED
- Full suite: 938 passed, 1 failed (pre-existing failure in test_harvest_spdf.py, unrelated), 45 skipped

## New Tests Added

1. **TestHybridSearchDefaultModel::test_hybrid_search_default_model** — PASSED
   Verifies hybrid_search defaults to model_name="indus" via inspect.signature

2. **TestHybridSearchDefaultModel::test_vector_search_default_model** — PASSED
   Verifies vector_search also defaults to model_name="indus"

3. **TestCardinalityRouting::test_selectivity_threshold_is_one_percent** — PASSED
   Confirms SELECTIVITY_THRESHOLD == 0.01

4. **TestCardinalityRouting::test_estimate_no_filters_returns_one** — PASSED
   Empty SearchFilters returns 1.0 without DB access

5. **TestCardinalityRouting::test_cardinality_routing_uses_filter_first** — PASSED
   With mocked selectivity < 0.01, \_filter_first_vector_search is called

6. **TestCardinalityRouting::test_cardinality_routing_uses_hnsw_for_broad_filters** — PASSED
   With mocked selectivity = 0.5, normal vector_search is called

7. **TestOpenAISignalSkipped::test_openai_skipped_when_no_rows** — PASSED
   When \_model_has_embeddings returns False, vector_search is not called for OpenAI

## Acceptance Criteria Verification

1. grep -n 'specter2' src/scix/search.py — 0 hits in defaults
2. hybrid_search() defaults model_name='indus', produces results via rrf_fuse()
3. Filter selectivity < 1% triggers filter-first CTE path
4. OpenAI signal skipped when model has 0 rows (\_model_has_embeddings check)
5. Existing tests pass (52/52 in test_search.py)
6. test_hybrid_search_default_model added and passing
7. test_cardinality_routing added (3 tests) and passing
