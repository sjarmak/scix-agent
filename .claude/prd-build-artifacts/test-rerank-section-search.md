# Test Summary — rerank-section-search (M5)

## New tests (`tests/test_search_within_paper_rerank.py`)

```
tests/test_search_within_paper_rerank.py::test_signature_has_use_rerank_and_top_k PASSED
tests/test_search_within_paper_rerank.py::test_search_result_has_sections_list_and_legacy_headline PASSED
tests/test_search_within_paper_rerank.py::test_backward_compat_positional_call PASSED
tests/test_search_within_paper_rerank.py::test_rerank_improves_ndcg_at_3 PASSED
tests/test_search_within_paper_rerank.py::test_p95_latency_under_500ms PASSED
```

5/5 pass.

## Adjacent suites (regression checks)

- `tests/test_mcp_search.py` — **15/15 PASS** (AC7 satisfied; the only suite
  the unit explicitly requires).
- `tests/test_mcp_paper_tools.py` — **51 pass / 2 pre-existing failures**.
  The 2 failures (`test_read_paper_section_dispatches`,
  `test_read_paper_section_defaults`) are caused by the parallel
  section-role-classifier build adding a `role=None` kwarg to the dispatch
  call; they were failing on `main` *before* this unit's changes
  (verified by `git stash` + re-running). Out of scope for this unit.
- `tests/test_body_adr006_guard.py` — **all PASS** after I extended the
  multi-cursor mock helper invocations in `TestSearchWithinPaperADR006Guard`
  to account for the new ts_rank cursor my change introduces.

## Eval result (negative)

| Metric | Value |
| --- | --- |
| Baseline nDCG@3 (BM25 only) | 1.0000 |
| Reranked nDCG@3 (MiniLM)    | 0.9815 |
| Delta                       | -0.0185 |
| p95 latency (rerank, MiniLM)| ~18 ms |
| Improvement threshold       | +0.05 |

The synthetic 20-entry fixture is ranked perfectly by per-section ts_rank
(BM25 already nails the gold section in every entry), so MiniLM has no room
to improve and produces a tiny regression. This mirrors the prior
cross-encoder-reranker-local M4-FAIL outcome at the abstract level: ship
the code path, keep `SCIX_RERANK_DEFAULT_MODEL='off'` as the production
default, and document the negative result.

Per the unit description, this triggers the "(negative result)" suffix on
the commit message and the eval doc has been written to
`results/within_paper_rerank_eval.md`.
