# Research — eval-script work unit

## Inputs surveyed

### `eval/retrieval_50q.jsonl` (gold set — sibling unit `eval-queries-curate`)
- 50 lines (verified `wc -l = 50`)
- Schema per line: `query`, `bucket`, `discipline`, `gold_bibcodes`, `notes`
- Buckets observed in head: `title_matchable`, others per spec: `concept`, `method`, `author_specific`
- Disciplines per spec: `astrophysics`, `planetary`, `earth`, `heliophysics`
- `gold_bibcodes` is a list (may be empty per spec; head shows non-empty examples)

### Existing `scripts/eval_retrieval_50q.py` (1211 lines)
- Targets a *different* framework: citation-network ground truth on a 20K/10K
  pilot sample — picks seed papers, derives relevant-set from `citation_edges`,
  measures specter2/indus/nomic/lexical/hybrid against that.
- Exposes utility helpers we can reuse: `dcg_at_k`, `ndcg_at_k`, `recall_at_k`,
  `mrr`, `precision_at_k` (all bibcode-set based, semantically compatible).
- The dataclasses (`QueryEval`, `EvalSummary`) and `aggregate_results`,
  `paired_difference_test`, `_make_lexical_query`, `generate_report`,
  `FULL_CORPUS_METHODS`, `METHODS`, `_significance_pairs` are scaffolding for
  the *legacy* citation-GT framework, not the new gold-set framework.
- Decision: **full rewrite**. The new framework reads JSONL (not picks seeds
  from DB), drives three modes (baseline/section/fused), and emits a JSON
  schema specified by AC3 — none of the existing dataclasses match.

### Baseline retrieval — `src/scix/search.py::hybrid_search`
- Signature:
  ```python
  hybrid_search(conn, query_text, query_embedding=None, *,
                model_name="indus", filters=None, vector_limit=60,
                lexical_limit=60, rrf_k=60, top_n=20, ef_search=100,
                reranker=None, include_body=True, ...) -> SearchResult
  ```
- Returns `SearchResult` with `papers: list[dict]` — each paper dict has
  `bibcode` key. We can pass `query_embedding=None` to get lexical-only
  baseline, OR encode INDUS via `src/scix/embed.py::load_model` +
  `embed_batch(pooling="mean")` to get full INDUS+BM25+body-BM25 RRF.
- We will encode INDUS so the baseline is a *real* hybrid, matching the
  task brief that says "INDUS+BM25 RRF".

### Section retrieval — `src/scix/mcp_server.py::_handle_section_retrieval`
- Returns a JSON string `{"results": [{"bibcode", "section_heading",
  "snippet", "score", "canonical_url"}], "total": N}`.
- Internally delegates to module-private helpers:
  - `_encode_section_query(query)` — uses `src/scix/embeddings/section_pipeline.py`
    `_load_model(DEFAULT_MODEL)` + `encode_batch` with `"search_query: "` prefix.
  - `_section_dense_retrieve(conn, vec, filter_sql, params, fanout)`
  - `_section_bm25_retrieve(conn, query, filter_sql, params, fanout)`
  - `_rrf_fuse([dense_keys, bm25_keys], k_rrf=60)`
- We will call `_handle_section_retrieval(conn, {"query": q, "k": 50})`,
  parse the JSON, pull `bibcode` from each `result`, dedupe preserving order
  (multiple sections from same paper collapse to one bibcode at first hit).
- Empty `section_embeddings` table → dense leg returns no rows; BM25 leg can
  still match — but `papers_fulltext.sections_tsv` is also populated upstream
  by the same parser PRD, so both legs may be empty. We treat
  "empty section_embeddings" as the canonical signal to skip section/fused.

### Fused = baseline + section (RRF k=60)
- Take ranked bibcode lists from baseline and section, fuse with RRF
  formula `score(d) = sum_i 1/(60 + rank_i(d))`.
- Existing `scix.search.rrf_fuse` operates on `list[dict]` with `bibcode` key
  — we'll reuse it where convenient, but a tiny local RRF helper that takes
  `list[list[str]]` of bibcodes is cleaner for the eval flow.

## Constraints reaffirmed
- No paid API imports (memory note `feedback_no_paid_apis`). Section query
  encode goes via the local nomic model in `embeddings.section_pipeline`.
  Baseline INDUS encode goes via local `transformers` in `scix.embed`.
- Full-corpus encode is upstream-blocked → `--dry-run` must work without DB
  or model and emit a stub JSON. Section/fused must skip gracefully when
  `SELECT count(*) FROM section_embeddings = 0`.

## Output schema (AC3)
```json
{
  "dry_run": false,
  "k": 10,
  "modes": {
    "baseline": {
      "overall": {"ndcg_at_10": 0.0, "mrr_at_10": 0.0, "recall_at_50": 0.0,
                  "n_queries": 50, "n_scored_ndcg": 50, "n_scored_recall": 50},
      "by_bucket": {
        "title_matchable": {"ndcg_at_10": ..., ...},
        "concept": {...},
        "method": {...},
        "author_specific": {...}
      }
    },
    "section": {...},
    "fused": {...}
  }
}
```
