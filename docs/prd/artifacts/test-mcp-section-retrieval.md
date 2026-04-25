# Test results — `section_retrieval` MCP tool

## Tool count check

```
$ awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py \
    | grep -oE 'name="[a-z_]+"' | sort -u | wc -l
17
```

Tools currently registered (17):

```
citation_chain, citation_graph, citation_similarity, claim_blame,
concept_search, entity, entity_context, facet_counts, find_gaps,
find_replications, find_similar_by_examples, get_paper, graph_context,
read_paper, search, section_retrieval, temporal_evolution
```

`find_similar_by_examples` is registered conditionally (only when
`QDRANT_URL` is set and `qdrant_tools` imports succeed) but is present in
the source-level enumeration, hence the count of 17. The runtime
`startup_self_test` only enforces 16 expected tools (the new
`section_retrieval` plus the 13 baseline + 2 PRD MH-4 tools), letting the
optional Qdrant tool come and go without breaking the gate.

## Unit tests

```
$ PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest \
    tests/test_mcp_section_retrieval.py -v
============================== 32 passed in 0.76s ==============================
```

Coverage of acceptance criteria:

| AC | Test(s) |
| --- | --- |
| 1 — tool registered with required schema | `TestRegistration.test_section_retrieval_appears_in_list_tools`, `TestRegistration.test_section_retrieval_input_schema_shape` |
| 2 — filters object accepts the four documented keys | `TestRegistration.test_section_retrieval_input_schema_shape`, `TestSectionFilterClauses.*` |
| 3 — dense + BM25 + RRF k=60 fusion | `TestRrfFuse.*`, `TestSectionRetrievalDispatch.test_filters_propagate_to_dense_and_bm25_sql`, `TestSectionRetrievalDispatch.test_returned_object_schema` |
| 4 — `SET LOCAL hnsw.iterative_scan = 'relaxed'` and `ef_search = 100` inside the same transaction | `TestSectionRetrievalDispatch.test_filters_propagate_to_dense_and_bm25_sql` (asserts SET LOCAL calls in the script) |
| 5 — return key set: bibcode, section_heading, snippet, score, canonical_url | `TestSectionRetrievalDispatch.test_returned_object_schema` |
| 6 — snippet ≤ 500 chars | `TestTruncateSnippet.*`, `TestSectionRetrievalDispatch.test_returned_object_schema` |
| 7 — RRF math, snippet truncation, filters, registration, schema | the 32 tests in the new file |
| 8 — pytest exits 0 | confirmed |
| 9 — no paid-API imports | `test_mcp_server_does_not_import_paid_sdks`, `test_section_pipeline_does_not_import_paid_sdks`, plus shell-side diff check |
| 10 — `_smoke_call_new_tools` lists section_retrieval | extended in `src/scix/mcp_server.py:_smoke_call_new_tools` |
| 11 — no regression in test_mcp_smoke.py | confirmed below |

## Regression check — existing smoke tests

```
$ PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest tests/test_mcp_smoke.py -v
======================== 25 passed, 2 warnings in 4.14s ========================
```

The two `DeprecationWarning`s are from a transitive SWIG dependency of
`scix.claim_blame`, unrelated to this PRD.

## Section pipeline regression

```
$ PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest \
    tests/embeddings/test_section_pipeline.py -v
============================== 17 passed in 0.13s ==============================
```

## No paid-API imports added

```
$ git diff prd-build/section-embeddings-mcp-consolidation -- src/scix/mcp_server.py \
    | grep -E '^\+.*(import openai|from openai|import cohere|from anthropic|import voyageai|from voyageai)'
(no output)
```

Confirmed — none of the four paid SDKs (`openai`, `cohere`, `anthropic`,
`voyageai`) are referenced anywhere in the changes.
