# Plan — eval-script work unit

## File targets
1. `scripts/eval_retrieval_50q.py` — full rewrite (CLI + retrieval + metrics +
   JSON output).
2. `tests/test_eval_retrieval_50q.py` — full rewrite (toy-ranking metric tests
   + JSON-schema test via `--dry-run`).
3. `docs/eval/retrieval_50q_2026-04.md` — new writeup template.

## CLI surface
```
python scripts/eval_retrieval_50q.py
    [--queries eval/retrieval_50q.jsonl]
    [--output docs/eval/retrieval_50q_2026-04.json]
    [--modes baseline,section,fused]
    [--k 10]
    [--dry-run]
```

## Module structure (new `scripts/eval_retrieval_50q.py`)

```python
# Constants
RRF_K = 60
DEFAULT_QUERIES = "eval/retrieval_50q.jsonl"
DEFAULT_OUTPUT  = "docs/eval/retrieval_50q_2026-04.json"
DEFAULT_MODES   = ("baseline", "section", "fused")
DEFAULT_K       = 10
RECALL_K        = 50
BUCKETS         = ("title_matchable", "concept", "method", "author_specific")

# Metric primitives (pure, no deps, easily testable)
def ndcg_at_10(retrieved: list[str], gold: list[str]) -> float | None
def mrr_at_10 (retrieved: list[str], gold: list[str]) -> float | None
def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None
    # Returns None when gold is empty (excluded from average)

# RRF
def rrf_fuse_bibcodes(rankings: list[list[str]], k_rrf: int = 60) -> list[str]

# IO
def load_queries(path: Path) -> list[dict]

# Retrieval drivers (each returns list[str] bibcodes, length up to RECALL_K)
def baseline_search(conn, query: str, k: int) -> list[str]
    # 1) Encode INDUS query vector via scix.embed.load_model("indus") + embed_batch
    # 2) Call scix.search.hybrid_search(conn, query, query_embedding=vec,
    #    model_name="indus", top_n=RECALL_K, include_body=True, rrf_k=RRF_K)
    # 3) Return [p["bibcode"] for p in result.papers]
    # Lazy-load model; cache module-level so multi-query runs share it.

def section_search(conn, query: str, k: int) -> list[str]
    # Call _handle_section_retrieval(conn, {"query": query, "k": RECALL_K})
    # Parse JSON, dedupe bibcodes preserving order.

def fused_search(baseline_bibs: list[str], section_bibs: list[str]) -> list[str]
    # rrf_fuse_bibcodes([baseline_bibs, section_bibs])

# Section availability probe (so empty table → skip section/fused gracefully)
def section_embeddings_available(conn) -> bool
    # SELECT 1 FROM section_embeddings LIMIT 1

# Mode runner
def run_mode(mode, queries, conn, k, dry_run, baseline_cache=None) -> dict
    # Returns {"overall": {...}, "by_bucket": {bucket: {...}}}
    # On dry_run or missing-section-data: emit zero-stub with same shape.
    # baseline_cache used by 'fused' mode to avoid re-running baseline.

# Aggregation
def aggregate_metrics(per_query: list[dict]) -> dict
    # Mean of non-None metric values; record n and n_scored.

# Score per query
def score_query(retrieved: list[str], gold: list[str], k: int) -> dict
    # {"ndcg_at_10": ..., "mrr_at_10": ..., "recall_at_50": ...}
    # None where excluded (gold empty).

# Output
def write_output(path: Path, payload: dict) -> None

# Entry
def main(argv: list[str] | None = None) -> int
```

## Dry-run behavior
- Skip DB connect + model load entirely.
- Build a stub JSON: every requested mode gets `overall` + `by_bucket` filled
  with `0.0` metrics and `"n_queries": <#queries>` if queries file is
  readable, else `0`.
- Top-level `"dry_run": true` marker.
- Exit 0.

## Empty section_embeddings handling
- After connecting, probe `section_embeddings` once via a SELECT.
- If empty AND `section` or `fused` is requested, emit zero-stub for those
  modes (same shape) with a `"skipped_reason": "section_embeddings_empty"`
  field on `overall`. Log a single INFO line. Continue to run baseline.

## Exclusion rules (from spec)
- nDCG@10: if `gold == []`, exclude from average.
- MRR@10: if `gold == []`, exclude from average.
- Recall@50: if `gold == []`, exclude (return None which the aggregator drops).

## Test fixture design (`tests/test_eval_retrieval_50q.py`)

Pure-Python tests (no DB, no model load, no torch import):

1. `test_ndcg_at_10_perfect_ranking`:
   gold=`["A"]`, retrieved=`["A", "B", "C"]` → `1.0`.
2. `test_ndcg_at_10_second_position`:
   gold=`["A"]`, retrieved=`["B", "A", "C"]` → `1/log2(3) ≈ 0.6309`.
3. `test_ndcg_at_10_not_in_top_k`:
   gold=`["A"]`, retrieved=11 items with A at rank 11 → `0.0`.
4. `test_ndcg_at_10_empty_gold` → returns `None`.
5. `test_mrr_at_10_first_position`:
   gold=`["A"]`, retrieved=`["A", "B"]` → `1.0`.
6. `test_mrr_at_10_second`:
   gold=`["A"]`, retrieved=`["B", "A"]` → `0.5`.
7. `test_mrr_at_10_no_match`:
   gold=`["A"]`, retrieved=`["B", "C", "D"]` → `0.0`.
8. `test_mrr_at_10_empty_gold` → `None`.
9. `test_recall_at_k_full`:
   gold=`["A", "B"]`, retrieved=`["A", "B", "X"]`, k=50 → `1.0`.
10. `test_recall_at_k_partial`:
    gold=`["A", "B"]`, retrieved=`["A", "X"]`, k=50 → `0.5`.
11. `test_recall_at_k_empty_gold` → `None`.
12. `test_rrf_fuse_bibcodes_basic`:
    `rrf_fuse_bibcodes([["A", "B"], ["B", "A"]])` puts both at top, A and B
    tied (or whichever the deterministic tiebreaker chooses).
13. `test_score_query_full`: integration of `score_query`.
14. `test_aggregate_excludes_none`: aggregator treats `None` as exclusion.
15. `test_dry_run_output_schema`:
    - Run `main(["--dry-run", "--modes", "baseline,section,fused",
                 "--output", str(tmp_path / "out.json")])` → returns `0`.
    - Load `out.json` → assert `"dry_run": True`, `"modes"` keys are exactly
      the requested modes, each mode has `"overall"` and `"by_bucket"`,
      each metric set has `ndcg_at_10`, `mrr_at_10`, `recall_at_50`,
      bucket dict has all four buckets.

## No paid API check
- Verify via grep — assertion handled at PR review level + CI grep.
- Script itself only imports `psycopg`, `scix.*` (which use local models).

## Commit
- Single commit on `work` branch:
  `prd-build: eval-script — 50-query retrieval eval (baseline/section/fused) + writeup template`
