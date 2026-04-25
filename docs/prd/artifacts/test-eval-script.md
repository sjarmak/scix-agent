# Test results — eval-script work unit

## Unit tests

```
$ python -m pytest tests/test_eval_retrieval_50q.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
collecting ... collected 29 items

tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_perfect_single_gold_at_top PASSED
tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_single_gold_at_rank_2 PASSED
tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_zero_when_no_overlap PASSED
tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_excluded_when_gold_empty PASSED
tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_relevant_at_rank_11_is_zero PASSED
tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_two_gold_perfect_top_two PASSED
tests/test_eval_retrieval_50q.py::TestNDCGAt10::test_two_gold_swapped_with_distractor PASSED
tests/test_eval_retrieval_50q.py::TestMRRAt10::test_first_position PASSED
tests/test_eval_retrieval_50q.py::TestMRRAt10::test_second_position PASSED
tests/test_eval_retrieval_50q.py::TestMRRAt10::test_no_match PASSED
tests/test_eval_retrieval_50q.py::TestMRRAt10::test_excluded_when_gold_empty PASSED
tests/test_eval_retrieval_50q.py::TestMRRAt10::test_only_top_10_counted PASSED
tests/test_eval_retrieval_50q.py::TestMRRAt10::test_first_relevant_wins_when_multi_gold PASSED
tests/test_eval_retrieval_50q.py::TestRecallAtK::test_full_recall PASSED
tests/test_eval_retrieval_50q.py::TestRecallAtK::test_half_recall PASSED
tests/test_eval_retrieval_50q.py::TestRecallAtK::test_zero_recall PASSED
tests/test_eval_retrieval_50q.py::TestRecallAtK::test_excluded_when_gold_empty PASSED
tests/test_eval_retrieval_50q.py::TestRecallAtK::test_truncates_at_k PASSED
tests/test_eval_retrieval_50q.py::TestAggregateMetrics::test_excludes_none PASSED
tests/test_eval_retrieval_50q.py::TestAggregateMetrics::test_empty_yields_zero_not_nan PASSED
tests/test_eval_retrieval_50q.py::TestRRFFuse::test_basic_intersection PASSED
tests/test_eval_retrieval_50q.py::TestRRFFuse::test_disjoint_rankings PASSED
tests/test_eval_retrieval_50q.py::TestDryRunOutputSchema::test_dry_run_schema PASSED
tests/test_eval_retrieval_50q.py::TestDryRunOutputSchema::test_dry_run_baseline_only PASSED
tests/test_eval_retrieval_50q.py::TestLoadQueries::test_real_gold_set_loads PASSED
tests/test_eval_retrieval_50q.py::TestLoadQueries::test_missing_field_raises PASSED
tests/test_eval_retrieval_50q.py::TestLoadQueries::test_skips_blank_and_comment_lines PASSED
tests/test_eval_retrieval_50q.py::TestScoreQuery::test_full_pipeline PASSED
tests/test_eval_retrieval_50q.py::TestScoreQuery::test_empty_gold_yields_none_metrics PASSED

============================== 29 passed in 0.11s ==============================
```

29/29 PASS in 0.11s.

## CLI dry-run smoke test

```
$ python scripts/eval_retrieval_50q.py --dry-run --modes baseline --output /tmp/eval_test_out.json
2026-04-25 19:31:46 INFO eval_retrieval_50q: dry-run output written to /tmp/eval_test_out.json
exit=0
```

Produced JSON validates the AC3 schema (excerpt — top of file):

```json
{
  "dry_run": true,
  "k": 10,
  "modes": {
    "baseline": {
      "by_bucket": {
        "author_specific": {
          "mrr_at_10": 0.0,
          "n_queries": 12,
          "n_scored_mrr": 0,
          ...
        },
        "concept":         {... "n_queries": 13 ...},
        "method":          {... "n_queries": 13 ...},
        "title_matchable": {... "n_queries": 12 ...}
      },
      "overall": {
        "ndcg_at_10": 0.0,
        "mrr_at_10": 0.0,
        "recall_at_50": 0.0,
        "n_queries": 50,
        ...
        "skipped_reason": "dry_run"
      }
    }
  }
}
```

Bucket sizes (12+13+13+12 = 50) match the gold set and confirm
``--dry-run`` reads the JSONL well enough to compute n-per-bucket without
running retrieval.

## Paid-API import grep

```
$ grep -E 'import openai|from openai|import cohere|from anthropic|import voyageai' scripts/eval_retrieval_50q.py
(no matches)
```

AC8 satisfied. Both encoders (INDUS for baseline, nomic for section) are
local open-weight models loaded via in-tree code paths
(``scix.embed.load_model("indus")`` and
``scix.embeddings.section_pipeline._load_model``).

## Acceptance criteria status

| AC | Status | Notes |
|----|--------|-------|
| 1  | PASS   | All five flags (`--queries`, `--output`, `--modes`, `--k`, `--dry-run`) wired with the spec defaults. |
| 2  | PASS   | nDCG@10 / MRR@10 / Recall@50 computed per bucket and overall. Empty `gold_bibcodes` excluded via `None` propagation. |
| 3  | PASS   | JSON shape `{modes: {<mode>: {overall: {...}, by_bucket: {<bucket>: {...}}}}}` verified by `test_dry_run_schema`. |
| 4  | PASS   | `docs/eval/retrieval_50q_2026-04.md` created with Overview, Methodology, Results table (placeholder), Discussion, Limitations sections including the upstream-blocked-encode note. |
| 5  | PASS   | `python scripts/eval_retrieval_50q.py --dry-run --modes baseline` exits 0 with no DB needed; live run probes `section_embeddings` and emits skip-stub for empty table. |
| 6  | PASS   | nDCG, MRR, Recall, JSON-schema, aggregator-exclusion, RRF, load_queries, score_query all covered. |
| 7  | PASS   | 29/29 tests green in 0.11s. |
| 8  | PASS   | grep confirms no `openai`/`anthropic`/`cohere`/`voyageai` imports; encoders are local INDUS + nomic. |
