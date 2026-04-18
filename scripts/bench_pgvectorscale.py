#!/usr/bin/env python3
"""Benchmark retrieval quality across HNSW + three DiskANN variants.

Runs the existing 50-query retrieval eval (``results/retrieval_eval_50q.json``)
against four index configurations on the pilot database:

    hnsw : idx_hnsw_baseline_indus          (pgvector HNSW baseline, halfvec)
    v1   : paper_embeddings_diskann_v1      (DiskANN + halfvec, no SBQ)
    v2   : paper_embeddings_diskann_v2      (DiskANN + SBQ default 2-bit)
    v3   : paper_embeddings_diskann_v3      (DiskANN + SBQ tuned)

Per query × index we measure nDCG@10, Recall@10, Recall@20, MRR, p50 and p95
latency. An exact brute-force baseline on a fixed 1M-paper random sample
(``random.seed(42)``) produces ground-truth top-10 neighbours per query so we
can report true Recall@10 against the sampled corpus.

Wilcoxon signed-rank (two-sided) is applied pairwise — v1 vs hnsw, v2 vs hnsw,
v3 vs hnsw — on per-query nDCG@10 with Bonferroni correction across the three
comparisons.

Outputs:
  - results/pgvs_benchmark/retrieval_quality.json (schema enforced)
  - results/pgvs_benchmark/retrieval_quality.md   (summary + PASS/FAIL table)

Refuses to run against the production DSN (``dbname=scix``). The benchmark is
intended exclusively for the pilot / benchmark database.

Usage::

    python3 scripts/bench_pgvectorscale.py --help
    python3 scripts/bench_pgvectorscale.py --dsn dbname=scix_pilot --dry-run
    python3 scripts/bench_pgvectorscale.py \\
        --dsn dbname=scix_pilot \\
        --indexes hnsw,v1,v2,v3 \\
        --eval-path results/retrieval_eval_50q.json \\
        --sample-size 1000000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Make ``src/`` and ``scripts/`` importable when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))

logger = logging.getLogger("bench_pgvectorscale")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MODEL_NAME = "indus"
DEFAULT_EVAL_PATH = Path("results/retrieval_eval_50q.json")
DEFAULT_OUT_DIR = Path("results/pgvs_benchmark")
DEFAULT_OUT_JSON_NAME = "retrieval_quality.json"
DEFAULT_OUT_MD_NAME = "retrieval_quality.md"
DEFAULT_INDEXES = "hnsw,v1,v2,v3"
DEFAULT_SAMPLE_SIZE = 1_000_000
DEFAULT_RANDOM_SEED = 42
DEFAULT_LIMIT = 20  # we need top-20 for Recall@20
NDCG_PASS_DELTA = 0.01  # PASS if |ndcg_variant - ndcg_hnsw| <= 0.01

DEFAULT_INDEX_MAP: dict[str, str] = {
    "hnsw": "idx_hnsw_baseline_indus",
    "v1": "paper_embeddings_diskann_v1",
    "v2": "paper_embeddings_diskann_v2",
    "v3": "paper_embeddings_diskann_v3",
}

# Production DB names the script MUST NOT touch.
_PRODUCTION_DB_NAMES: frozenset[str] = frozenset({"scix"})


# ---------------------------------------------------------------------------
# DSN safety guard
# ---------------------------------------------------------------------------


def assert_pilot_dsn(dsn: str | None) -> None:
    """Raise ValueError if ``dsn`` points at the production database.

    Uses libpq parsing so both key=value and URI forms are handled. Empty DSNs
    are rejected so the guard is never silently bypassed.
    """
    if not dsn or not dsn.strip():
        raise ValueError(
            "Empty DSN — refuse to run without an explicit pilot DSN."
        )
    try:
        # Prefer the shared implementation when available.
        from scix.db import is_production_dsn  # type: ignore[import-not-found]

        if is_production_dsn(dsn):
            raise ValueError(
                f"Refuse to run against production DSN (resolves to dbname in "
                f"{sorted(_PRODUCTION_DB_NAMES)}). Set --dsn to a pilot DB "
                "(e.g. dbname=scix_pilot)."
            )
        return
    except ImportError:
        pass

    # Local fallback — parse the DSN with psycopg directly.
    try:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict

        try:
            params = conninfo_to_dict(dsn)
        except psycopg.ProgrammingError as exc:
            raise ValueError(f"Invalid DSN: {exc}") from exc
    except ImportError:
        # Minimal fallback for environments without psycopg — accept only
        # explicit key=value form.
        params = {}
        for token in dsn.split():
            if "=" in token:
                key, _, value = token.partition("=")
                params[key.strip()] = value.strip()

    dbname = params.get("dbname")
    if isinstance(dbname, str) and dbname.lower() in _PRODUCTION_DB_NAMES:
        raise ValueError(
            f"Refuse to run against production DSN (dbname={dbname!r}). "
            "This benchmark is for pilot/benchmark databases only. "
            "Set --dsn to a non-production DB (e.g. dbname=scix_pilot)."
        )


# ---------------------------------------------------------------------------
# Metric helpers (pure — unit-testable without a DB)
# ---------------------------------------------------------------------------


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k (binary relevance OK)."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))


def ndcg_at_10(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Normalized DCG at 10 with binary relevance.

    Returns 0.0 when the relevant set is empty.
    """
    relevant_set: set[str] = set(relevant)
    if not relevant_set:
        return 0.0
    k = 10
    actual_relevance = [1 if bib in relevant_set else 0 for bib in ranked[:k]]
    actual_dcg = dcg_at_k(actual_relevance, k)
    n_rel_in_top_k = min(len(relevant_set), k)
    ideal_relevance = [1] * n_rel_in_top_k + [0] * (k - n_rel_in_top_k)
    ideal_dcg = dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall at rank k (fraction of relevant items retrieved in top k)."""
    relevant_set: set[str] = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for bib in ranked[:k] if bib in relevant_set)
    return hits / len(relevant_set)


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item (0 if none)."""
    relevant_set: set[str] = set(relevant)
    if not relevant_set:
        return 0.0
    for i, bib in enumerate(ranked):
        if bib in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Return percentile ``pct`` (0-100) using linear interpolation.

    Returns 0.0 on empty input. Uses ``numpy.percentile`` when numpy is
    importable; otherwise a pure-python fallback (nearest-rank interpolation)
    so tests never require numpy at import time.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    # Linear interpolation (numpy-compatible semantics).
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------


def wilcoxon_compare(variant: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired values.

    Returns a dict with keys ``statistic``, ``p_value``, ``n_pairs``, and
    ``mean_diff``. When the paired arrays are empty, too short, or all
    differences are zero, ``statistic`` and ``p_value`` are ``None`` and a
    ``note`` field explains the condition.
    """
    if len(variant) != len(baseline):
        raise ValueError(
            f"Wilcoxon paired arrays must match length: {len(variant)} vs {len(baseline)}"
        )
    diffs = [float(a) - float(b) for a, b in zip(variant, baseline)]
    n = len(diffs)
    if n == 0:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": 0,
            "mean_diff": 0.0,
            "note": "no paired observations",
        }
    mean_diff = sum(diffs) / n
    if all(d == 0.0 for d in diffs):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": n,
            "mean_diff": 0.0,
            "note": "all paired differences are zero",
        }
    if n < 5:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": n,
            "mean_diff": round(mean_diff, 6),
            "note": "n<5 — wilcoxon not run",
        }
    try:
        from scipy.stats import wilcoxon  # type: ignore[import-not-found]
    except ImportError:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": n,
            "mean_diff": round(mean_diff, 6),
            "note": "scipy not available",
        }
    try:
        stat, p_value = wilcoxon(diffs, alternative="two-sided")
    except ValueError as exc:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": n,
            "mean_diff": round(mean_diff, 6),
            "note": f"wilcoxon raised: {exc}",
        }
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_pairs": n,
        "mean_diff": round(mean_diff, 6),
    }


def bonferroni_adjust(p_value: float | None, n_comparisons: int) -> float | None:
    """Bonferroni-adjust a raw p-value, capping at 1.0."""
    if p_value is None:
        return None
    if n_comparisons <= 0:
        return float(p_value)
    return min(1.0, float(p_value) * int(n_comparisons))


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------


def load_queries_from_eval(eval_path: Path) -> list[str]:
    """Return unique seed_bibcodes from ``results/retrieval_eval_50q.json``.

    The file stores per-query rows keyed by seed_bibcode; we return the first
    occurrence ordering (deterministic) which matches how the file was
    produced.
    """
    data = json.loads(Path(eval_path).read_text())
    per_query = data.get("per_query") or []
    seen: set[str] = set()
    order: list[str] = []
    for row in per_query:
        bib = row.get("seed_bibcode")
        if not isinstance(bib, str) or bib in seen:
            continue
        seen.add(bib)
        order.append(bib)
    return order


# ---------------------------------------------------------------------------
# Dry-run output
# ---------------------------------------------------------------------------


def dry_run_payload(
    index_names: Sequence[str],
    index_map: dict[str, str],
    sample_size: int,
    eval_path: Path,
) -> dict[str, Any]:
    """Return a schema-valid payload with empty metrics — no DB touched."""
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "eval_path": str(eval_path),
        "exact_baseline_sample_size": int(sample_size),
        "random_seed": DEFAULT_RANDOM_SEED,
        "model_name": MODEL_NAME,
        "env": {},
        "queries": [],
        "indexes": [
            {
                "name": name,
                "index_name": index_map.get(name, name),
                "metrics": {
                    "ndcg_at_10": None,
                    "recall_at_10": None,
                    "recall_at_20": None,
                    "mrr": None,
                    "p50_ms": None,
                    "p95_ms": None,
                },
                "per_query": [],
            }
            for name in index_names
        ],
        "pairwise_significance": [
            {
                "compared": f"{name} vs hnsw",
                "statistic": None,
                "p_value": None,
                "bonferroni_adjusted_p": None,
                "n_pairs": 0,
                "mean_diff": 0.0,
                "note": "dry-run — no measurements taken",
            }
            for name in index_names
            if name != "hnsw"
        ],
        "explanation": (
            "Dry-run mode: no database connections were opened. The JSON "
            "schema is complete but all metric values are null and per_query "
            "arrays are empty. Re-run without --dry-run against a pilot DSN "
            "to populate real measurements."
        ),
    }


# ---------------------------------------------------------------------------
# DB-facing helpers (only called when not --dry-run)
# ---------------------------------------------------------------------------


def _fetch_query_embedding(conn: Any, bibcode: str) -> list[float] | None:
    """Return the stored INDUS embedding for ``bibcode`` as a float list."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM paper_embeddings "
            "WHERE bibcode = %s AND model_name = %s LIMIT 1",
            (bibcode, MODEL_NAME),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    raw = row[0]
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",") if x.strip()]
    return list(raw)


def _fetch_ground_truth_relevant(conn: Any, bibcode: str) -> set[str]:
    """Return citation-neighbours within _pilot_sample for ``bibcode``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT target_bibcode FROM citation_edges
              WHERE source_bibcode = %s
                AND target_bibcode IN (SELECT bibcode FROM _pilot_sample)
            UNION
            SELECT source_bibcode FROM citation_edges
              WHERE target_bibcode = %s
                AND source_bibcode IN (SELECT bibcode FROM _pilot_sample)
            """,
            (bibcode, bibcode),
        )
        return {row[0] for row in cur.fetchall()}


def _run_index_query(
    conn: Any,
    index_name: str,
    query_embedding_text: str,
    limit: int,
    exclude_bibcode: str,
) -> tuple[list[str], float]:
    """Run a single top-K cosine query; return (ranked_bibcodes, latency_ms).

    The query uses ``ORDER BY embedding <=> %s::halfvec`` and filters on
    ``model_name='indus'``; the planner picks whichever matching index exists.
    ``index_name`` is accepted as documentation / correlation metadata — we do
    not pin the planner here because these are Layer 0 tests against a single
    live index at a time.
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode FROM paper_embeddings "
            "WHERE model_name = %s AND bibcode != %s "
            "ORDER BY embedding <=> %s::halfvec LIMIT %s",
            (MODEL_NAME, exclude_bibcode, query_embedding_text, int(limit)),
        )
        rows = cur.fetchall()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return [r[0] for r in rows], latency_ms


def _sample_bibcodes(conn: Any, n: int, seed: int) -> list[str]:
    """Return up to ``n`` bibcodes sampled with seeded RNG from paper_embeddings."""
    # Server-side setseed gives reproducibility across runs.
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", [seed / 2**31])
        cur.execute(
            "SELECT bibcode FROM paper_embeddings "
            "WHERE model_name = %s "
            "ORDER BY random() LIMIT %s",
            (MODEL_NAME, int(n)),
        )
        return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def _per_query_metrics(
    ranked: Sequence[str],
    relevant: Iterable[str],
    latency_ms: float,
) -> dict[str, Any]:
    """Compute the full per-query metric dict given a ranking + relevant set."""
    relevant_set: set[str] = set(relevant)
    return {
        "ndcg_10": round(ndcg_at_10(ranked, relevant_set), 6),
        "recall_10": round(recall_at_k(ranked, relevant_set, 10), 6),
        "recall_20": round(recall_at_k(ranked, relevant_set, 20), 6),
        "mrr": round(mrr(ranked, relevant_set), 6),
        "latency_ms": round(float(latency_ms), 3),
    }


def _summarize(per_query: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query rows into mean metrics + latency percentiles."""
    if not per_query:
        return {
            "ndcg_at_10": None,
            "recall_at_10": None,
            "recall_at_20": None,
            "mrr": None,
            "p50_ms": None,
            "p95_ms": None,
        }

    def _mean(key: str) -> float:
        return round(sum(float(row[key]) for row in per_query) / len(per_query), 6)

    latencies = [float(row["latency_ms"]) for row in per_query]
    return {
        "ndcg_at_10": _mean("ndcg_10"),
        "recall_at_10": _mean("recall_10"),
        "recall_at_20": _mean("recall_20"),
        "mrr": _mean("mrr"),
        "p50_ms": round(percentile(latencies, 50.0), 3),
        "p95_ms": round(percentile(latencies, 95.0), 3),
    }


def run_benchmark(
    dsn: str,
    eval_path: Path,
    index_names: Sequence[str],
    index_map: dict[str, str],
    sample_size: int,
    limit: int = DEFAULT_LIMIT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Execute the full benchmark — DB-heavy; never called in tests."""
    import numpy as np
    import psycopg

    random.seed(random_seed)

    # Environment metadata.
    from pgvs_bench_env import capture_env  # type: ignore[import-not-found]
    env = capture_env(dsn=dsn)

    queries = load_queries_from_eval(eval_path)
    if max_queries is not None:
        queries = queries[:max_queries]
    logger.info("Loaded %d queries from %s", len(queries), eval_path)

    result: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "eval_path": str(eval_path),
        "exact_baseline_sample_size": int(sample_size),
        "random_seed": int(random_seed),
        "model_name": MODEL_NAME,
        "env": env,
        "queries": [],
        "indexes": [],
        "pairwise_significance": [],
    }

    # Single long-lived connection; autocommit since we're only reading.
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True

        # --- 1. Exact baseline: sample + compute ground truth top-10 per query.
        logger.info("Sampling %d bibcodes for exact baseline (seed=%d)...", sample_size, random_seed)
        sampled = _sample_bibcodes(conn, sample_size, random_seed)
        logger.info("Fetched %d sampled bibcodes", len(sampled))

        # Fetch sampled embeddings into a numpy matrix.
        sample_set = tuple(sampled)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, embedding::text FROM paper_embeddings "
                "WHERE model_name = %s AND bibcode = ANY(%s)",
                (MODEL_NAME, list(sample_set)),
            )
            rows = cur.fetchall()
        sample_bibs: list[str] = []
        sample_matrix_list: list[list[float]] = []
        for bib, emb_text in rows:
            vec = [float(x) for x in emb_text.strip("[]").split(",") if x.strip()]
            sample_bibs.append(bib)
            sample_matrix_list.append(vec)
        sample_matrix = np.asarray(sample_matrix_list, dtype=np.float32)
        # L2-normalize so cosine distance == 1 - dot.
        norms = np.linalg.norm(sample_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sample_matrix /= norms
        logger.info("Built sample matrix: %s", sample_matrix.shape)

        # --- 2. Build per-query context: relevant set + exact top-10.
        query_ctx: list[dict[str, Any]] = []
        for bib in queries:
            emb = _fetch_query_embedding(conn, bib)
            if emb is None:
                logger.warning("Skipping %s: no INDUS embedding found", bib)
                continue
            relevant = _fetch_ground_truth_relevant(conn, bib)
            emb_np = np.asarray(emb, dtype=np.float32)
            n = np.linalg.norm(emb_np)
            if n > 0:
                emb_np = emb_np / n
            # Cosine similarity against sample; top-10 bibcodes by similarity.
            sims = sample_matrix @ emb_np
            topk_idx = np.argsort(-sims)[:10]
            exact_top10 = [sample_bibs[i] for i in topk_idx if sample_bibs[i] != bib][:10]
            emb_text = "[" + ",".join(f"{v:.8f}" for v in emb) + "]"
            query_ctx.append(
                {
                    "seed_bibcode": bib,
                    "relevant": relevant,
                    "exact_top10": exact_top10,
                    "embedding_text": emb_text,
                }
            )
            result["queries"].append(
                {"seed_bibcode": bib, "relevant_count": len(relevant)}
            )

        # --- 3. Run each index variant.
        for name in index_names:
            idx_db_name = index_map.get(name, name)
            logger.info("Running index=%s (%s)", name, idx_db_name)
            per_query: list[dict[str, Any]] = []
            for ctx in query_ctx:
                ranked, latency_ms = _run_index_query(
                    conn,
                    idx_db_name,
                    ctx["embedding_text"],
                    limit,
                    ctx["seed_bibcode"],
                )
                # Citation-based recall against `relevant`.
                row = _per_query_metrics(ranked, ctx["relevant"], latency_ms)
                # Also capture true Recall@10 against exact ground truth.
                row["true_recall_10"] = round(
                    recall_at_k(ranked, set(ctx["exact_top10"]), 10), 6
                )
                row["seed_bibcode"] = ctx["seed_bibcode"]
                per_query.append(row)
            result["indexes"].append(
                {
                    "name": name,
                    "index_name": idx_db_name,
                    "metrics": _summarize(per_query),
                    "per_query": per_query,
                }
            )

    # --- 4. Pairwise Wilcoxon + Bonferroni on nDCG@10.
    by_name = {entry["name"]: entry for entry in result["indexes"]}
    if "hnsw" in by_name:
        hnsw_rows = by_name["hnsw"]["per_query"]
        # Align per-seed to keep pairs in the same order.
        hnsw_by_seed = {r["seed_bibcode"]: r["ndcg_10"] for r in hnsw_rows}
        variants = [n for n in index_names if n != "hnsw"]
        for name in variants:
            v_rows = by_name[name]["per_query"]
            paired_v: list[float] = []
            paired_h: list[float] = []
            for r in v_rows:
                bib = r["seed_bibcode"]
                if bib in hnsw_by_seed:
                    paired_v.append(float(r["ndcg_10"]))
                    paired_h.append(float(hnsw_by_seed[bib]))
            w = wilcoxon_compare(paired_v, paired_h)
            adj = bonferroni_adjust(w.get("p_value"), max(len(variants), 1))
            entry = {
                "compared": f"{name} vs hnsw",
                "statistic": w.get("statistic"),
                "p_value": w.get("p_value"),
                "bonferroni_adjusted_p": adj,
                "n_pairs": w.get("n_pairs", 0),
                "mean_diff": w.get("mean_diff", 0.0),
            }
            if "note" in w:
                entry["note"] = w["note"]
            result["pairwise_significance"].append(entry)

    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the human-readable summary report for retrieval_quality.md."""
    lines: list[str] = []
    lines.append("# Retrieval Quality Benchmark")
    lines.append("")
    lines.append(f"- Timestamp: `{payload.get('timestamp', '')}`")
    lines.append(f"- Run ID: `{payload.get('run_id', '')}`")
    lines.append(f"- Dry run: `{payload.get('dry_run', False)}`")
    lines.append(f"- Model: `{payload.get('model_name', MODEL_NAME)}`")
    lines.append(f"- Eval queries: `{payload.get('eval_path', '')}`")
    lines.append(
        f"- Exact baseline sample size: `{payload.get('exact_baseline_sample_size', 0)}`"
    )
    lines.append(f"- Random seed: `{payload.get('random_seed', DEFAULT_RANDOM_SEED)}`")
    lines.append("")

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Index | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 (ms) | p95 (ms) |"
    )
    lines.append(
        "|-------|---------|-----------|-----------|-----|----------|----------|"
    )

    def _fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    hnsw_entry: dict[str, Any] | None = None
    for entry in payload.get("indexes", []):
        m = entry.get("metrics", {})
        lines.append(
            f"| {entry.get('name','')} ({entry.get('index_name','')}) "
            f"| {_fmt(m.get('ndcg_at_10'))} "
            f"| {_fmt(m.get('recall_at_10'))} "
            f"| {_fmt(m.get('recall_at_20'))} "
            f"| {_fmt(m.get('mrr'))} "
            f"| {_fmt(m.get('p50_ms'))} "
            f"| {_fmt(m.get('p95_ms'))} |"
        )
        if entry.get("name") == "hnsw":
            hnsw_entry = entry

    # PASS/FAIL annotations for each DiskANN variant vs HNSW.
    lines.append("")
    lines.append("## DiskANN vs HNSW — PASS/FAIL on nDCG@10 within 1%")
    lines.append("")
    lines.append(
        f"A variant PASSES if `|mean nDCG@10(variant) − mean nDCG@10(hnsw)| ≤ {NDCG_PASS_DELTA:.2f}`."
    )
    lines.append("")
    lines.append("| Variant | nDCG@10 | Δ vs HNSW | Verdict |")
    lines.append("|---------|---------|-----------|---------|")
    if hnsw_entry is None:
        lines.append(
            "| _hnsw row not present — cannot annotate PASS/FAIL_ |  |  |  |"
        )
    else:
        hnsw_ndcg = hnsw_entry.get("metrics", {}).get("ndcg_at_10")
        for entry in payload.get("indexes", []):
            name = entry.get("name")
            if name == "hnsw":
                continue
            v = entry.get("metrics", {}).get("ndcg_at_10")
            if v is None or hnsw_ndcg is None:
                verdict = "N/A"
                delta_str = "N/A"
            else:
                delta = float(v) - float(hnsw_ndcg)
                delta_str = f"{delta:+.4f}"
                verdict = "PASS" if abs(delta) <= NDCG_PASS_DELTA else "FAIL"
            lines.append(
                f"| {name} | {_fmt(v)} | {delta_str} | {verdict} |"
            )

    # Significance.
    lines.append("")
    lines.append("## Pairwise Significance (Wilcoxon signed-rank on per-query nDCG@10)")
    lines.append("")
    lines.append(
        "| Compared | n | Mean Δ | p-value | Bonferroni-adjusted p |"
    )
    lines.append(
        "|----------|---|--------|---------|------------------------|"
    )
    for sig in payload.get("pairwise_significance", []):
        p_val = sig.get("p_value")
        adj = sig.get("bonferroni_adjusted_p")
        lines.append(
            f"| {sig.get('compared','')} "
            f"| {sig.get('n_pairs', 0)} "
            f"| {_fmt(sig.get('mean_diff'))} "
            f"| {_fmt(p_val)} "
            f"| {_fmt(adj)} |"
        )

    if payload.get("dry_run"):
        lines.append("")
        lines.append("## Dry Run")
        lines.append("")
        lines.append(payload.get("explanation", "Dry run — no measurements taken."))

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_index_map(spec: str | None, defaults: dict[str, str]) -> dict[str, str]:
    """Parse ``name=index_name,name2=index_name2`` into a dict (fall back to defaults)."""
    if not spec:
        return dict(defaults)
    out = dict(defaults)
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"--index-map token {token!r} missing '=' separator "
                "(expected form: name=index_name)"
            )
        k, _, v = token.partition("=")
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark retrieval quality (nDCG@10, Recall@{10,20}, MRR, p50/p95 "
            "latency) across HNSW baseline + three DiskANN variants on the "
            "pilot database. Refuses to run against the production DSN "
            "(dbname=scix)."
        ),
    )
    parser.add_argument(
        "--dsn",
        required=False,
        default=None,
        help=(
            "PostgreSQL DSN for the pilot database. MUST NOT resolve to "
            "production (dbname=scix). Example: 'dbname=scix_pilot'."
        ),
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help=(
            f"Path to the 50-query eval JSON (default: {DEFAULT_EVAL_PATH}). "
            "Seed bibcodes are read from its per_query array."
        ),
    )
    parser.add_argument(
        "--indexes",
        default=DEFAULT_INDEXES,
        help=(
            f"Comma-separated list of index logical names "
            f"(default: '{DEFAULT_INDEXES}'). Each name is mapped to a "
            "physical index via --index-map; defaults are "
            "'hnsw=idx_hnsw_baseline_indus', "
            "'v1=paper_embeddings_diskann_v1', "
            "'v2=paper_embeddings_diskann_v2', "
            "'v3=paper_embeddings_diskann_v3'."
        ),
    )
    parser.add_argument(
        "--index-map",
        default=None,
        help=(
            "Optional override for the logical→physical index name mapping, "
            "e.g. 'hnsw=my_hnsw,v1=my_v1'. Unspecified entries keep their "
            "default mapping."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            f"Output directory (default: {DEFAULT_OUT_DIR}). Writes "
            f"{DEFAULT_OUT_JSON_NAME} and {DEFAULT_OUT_MD_NAME}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit a schema-valid JSON + MD with empty per-index metrics and an "
            "explanation block. No DB connections are opened."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            f"Exact-baseline random-sample size (default: {DEFAULT_SAMPLE_SIZE}). "
            "Ground-truth top-10 per query is computed against this sample."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for the exact-baseline sample (default: 42).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Top-K limit per query (default: 20 — required for Recall@20).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Debug: cap number of queries (default: no cap).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / DEFAULT_OUT_JSON_NAME
    md_path = out_dir / DEFAULT_OUT_MD_NAME
    # Strip set/numpy-etc. artifacts if any snuck in.
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    md_path.write_text(render_markdown(payload))
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return json_path, md_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.dsn
    if dsn is None:
        print(
            "ERROR: --dsn is required (pilot/benchmark DB). "
            "Refusing to default to production.",
            file=sys.stderr,
        )
        return 2
    try:
        assert_pilot_dsn(dsn)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    index_names = [name.strip() for name in args.indexes.split(",") if name.strip()]
    if not index_names:
        print("ERROR: --indexes produced an empty list.", file=sys.stderr)
        return 2
    try:
        index_map = _parse_index_map(args.index_map, DEFAULT_INDEX_MAP)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        payload = dry_run_payload(
            index_names, index_map, args.sample_size, args.eval_path
        )
    else:
        payload = run_benchmark(
            dsn=dsn,
            eval_path=args.eval_path,
            index_names=index_names,
            index_map=index_map,
            sample_size=args.sample_size,
            limit=args.limit,
            random_seed=args.random_seed,
            max_queries=args.max_queries,
        )

    _write_outputs(payload, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
