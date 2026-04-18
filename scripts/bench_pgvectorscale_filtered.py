#!/usr/bin/env python3
"""Filtered-query retrieval benchmark for pgvector HNSW + pgvectorscale DiskANN.

Extends the unfiltered retrieval-quality benchmark (``scripts/bench_pgvectorscale.py``)
with two filter configurations. For each of four indexes (HNSW, DiskANN V1/V2/V3)
and each filter, measures nDCG@10, Recall@10, Recall@20, MRR, p50, p95 latency.

Filter configurations
=====================

    F1 : ``WHERE year = 2024``                                 (~10% selectivity)
    F2 : ``WHERE arxiv_class && '{astro-ph.GA,astro-ph.SR}'::text[]``  (~20% selectivity)

Filter strategy per index
=========================

- **HNSW (pgvector):** ``SET LOCAL hnsw.iterative_scan = 'relaxed_order'`` before
  the filtered query. Iterative scan is required for filtered queries to avoid
  returning fewer than K results. The per-query record captures this.
- **DiskANN V1/V2/V3 (pgvectorscale):** native label/WHERE filtering is supported
  by the access method; no session hint needed. Recorded as
  ``filter_strategy="native"``.

Unfiltered baseline
===================

Reads ``results/pgvs_benchmark/retrieval_quality.json`` if present to recover
the unfiltered p95 per index. Each per-cell record receives
``unfiltered_p95_ms`` and a boolean ``p95_degradation_flag`` that is True when
filtered p95 >= ``2x`` unfiltered p95. When the baseline file is missing the
flag is recorded as the string ``"unknown"``.

Outputs
=======

- ``results/pgvs_benchmark/filtered_queries.json``
- ``results/pgvs_benchmark/filtered_queries.md``  (flags any index with
  ``p95 under filter >= 2x unfiltered p95`` in an explicit column)

Safety
======

Refuses to run against the production DSN (``dbname=scix``). The benchmark is
intended exclusively for the pilot / benchmark database.

Usage
=====

::

    python3 scripts/bench_pgvectorscale_filtered.py --help
    python3 scripts/bench_pgvectorscale_filtered.py --dsn dbname=scix_pilot --dry-run
    python3 scripts/bench_pgvectorscale_filtered.py \\
        --dsn dbname=scix_pilot \\
        --filter both \\
        --eval-path results/retrieval_eval_50q.json \\
        --out results/pgvs_benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

logger = logging.getLogger("bench_pgvectorscale_filtered")


# ---------------------------------------------------------------------------
# Defaults & constants
# ---------------------------------------------------------------------------

MODEL_NAME = "indus"
DEFAULT_EVAL_PATH = Path("results/retrieval_eval_50q.json")
DEFAULT_OUT_DIR = Path("results/pgvs_benchmark")
DEFAULT_OUT_JSON_NAME = "filtered_queries.json"
DEFAULT_OUT_MD_NAME = "filtered_queries.md"
DEFAULT_UNFILTERED_BASELINE = Path("results/pgvs_benchmark/retrieval_quality.json")
DEFAULT_INDEXES = "hnsw,v1,v2,v3"
DEFAULT_LIMIT = 20  # need top-20 for Recall@20
DEGRADATION_THRESHOLD = 2.0  # flag p95 filtered >= 2x unfiltered

DEFAULT_INDEX_MAP: dict[str, str] = {
    "hnsw": "idx_hnsw_baseline_indus",
    "v1": "paper_embeddings_diskann_v1",
    "v2": "paper_embeddings_diskann_v2",
    "v3": "paper_embeddings_diskann_v3",
}

# Filter configurations — clause strings are asserted by grep checks in the
# acceptance criteria: the substrings "year = 2024" and "astro-ph" must appear
# in this source file.
FILTERS: dict[str, dict[str, Any]] = {
    "f1": {
        "clause": "year = 2024",
        "selectivity_pct_estimated": 10,
        "description": "F1: papers from 2024 (~10% of corpus)",
    },
    "f2": {
        "clause": "arxiv_class && '{astro-ph.GA,astro-ph.SR}'::text[]",
        "selectivity_pct_estimated": 20,
        "description": "F2: arxiv_class overlaps {astro-ph.GA, astro-ph.SR} (~20%)",
    },
}

FILTER_CHOICES = ["f1", "f2", "both"]

# Production DB names the script MUST NOT touch.
_PRODUCTION_DB_NAMES: frozenset[str] = frozenset({"scix"})


# ---------------------------------------------------------------------------
# DSN safety guard
# ---------------------------------------------------------------------------


def assert_pilot_dsn(dsn: str | None) -> None:
    """Raise ``ValueError`` if ``dsn`` points at the production database.

    Uses the shared ``scix.db.is_production_dsn`` helper when available so
    policy stays consistent with the rest of the repo; falls back to libpq
    DSN parsing. Empty DSNs are rejected so the guard is never silently
    bypassed. The error message contains both ``"production"`` and
    ``"refuse"`` to satisfy tests that grep for either.
    """
    if not dsn or not dsn.strip():
        raise ValueError(
            "Empty DSN — refuse to run without an explicit pilot DSN."
        )
    try:
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
    """Normalized DCG at 10 with binary relevance. Returns 0.0 when no relevant."""
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
    """Recall at rank k — fraction of relevant items retrieved in top k."""
    relevant_set: set[str] = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for bib in ranked[:k] if bib in relevant_set)
    return hits / len(relevant_set)


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant item (0 if none)."""
    relevant_set: set[str] = set(relevant)
    if not relevant_set:
        return 0.0
    for i, bib in enumerate(ranked):
        if bib in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Return percentile ``pct`` (0-100) using linear interpolation.

    Returns 0.0 on empty input. Pure python — numpy not required at import.
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
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Degradation flagging (pure — testable)
# ---------------------------------------------------------------------------


def flag_p95_degradation(
    filtered_p95: float | None,
    unfiltered_p95: float | None,
    threshold: float = DEGRADATION_THRESHOLD,
) -> bool:
    """Return True when filtered p95 indicates ``>=threshold x`` slowdown.

    Returns ``False`` (never raises) when either argument is ``None`` or the
    unfiltered baseline is non-positive. Callers that need to distinguish
    "not degraded" from "cannot evaluate" should check for ``None`` inputs
    themselves and render ``"unknown"``.

    Args:
        filtered_p95: p95 latency measured under the filter (ms).
        unfiltered_p95: p95 latency from the unfiltered baseline (ms).
        threshold: multiplicative factor; default 2.0. The test is
            ``filtered >= threshold * unfiltered``.

    Returns:
        True when degradation meets or exceeds the threshold.
    """
    if filtered_p95 is None or unfiltered_p95 is None:
        return False
    try:
        f = float(filtered_p95)
        u = float(unfiltered_p95)
    except (TypeError, ValueError):
        return False
    if u <= 0:
        return False
    return f >= threshold * u


def filter_strategy_for(index_name: str) -> str:
    """Return a human-readable description of the filter strategy used.

    HNSW uses ``hnsw.iterative_scan = 'relaxed_order'``; DiskANN variants use
    pgvectorscale's native label-filtered search.
    """
    if index_name == "hnsw":
        return "pgvector: SET LOCAL hnsw.iterative_scan = 'relaxed_order'"
    return "pgvectorscale: native label/WHERE filtering"


# ---------------------------------------------------------------------------
# Unfiltered baseline loader
# ---------------------------------------------------------------------------


def load_unfiltered_baseline(path: Path) -> dict[str, float | None]:
    """Return ``{logical_index_name: p95_ms}`` from the unfiltered bench JSON.

    Reads ``results/pgvs_benchmark/retrieval_quality.json`` if present. Keys
    are the logical names (``hnsw``, ``v1``, ``v2``, ``v3``). Missing file or
    unreadable JSON returns an empty dict — caller decides how to render.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.info("Unfiltered baseline %s unavailable (%s)", path, exc)
        return {}
    out: dict[str, float | None] = {}
    for entry in data.get("indexes", []) or []:
        name = entry.get("name")
        p95 = entry.get("metrics", {}).get("p95_ms")
        if isinstance(name, str):
            out[name] = float(p95) if isinstance(p95, (int, float)) else None
    return out


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------


def load_queries_from_eval(eval_path: Path) -> list[str]:
    """Return unique ``seed_bibcode`` values from the 50-query eval JSON.

    Preserves first-occurrence ordering so results are deterministic.
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
# Cell construction + summary
# ---------------------------------------------------------------------------


def _empty_metrics() -> dict[str, float | None]:
    return {
        "ndcg_at_10": None,
        "recall_at_10": None,
        "recall_at_20": None,
        "mrr": None,
        "p50_ms": None,
        "p95_ms": None,
    }


def _summarize(per_query: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate per-query rows into mean metrics + latency percentiles."""
    if not per_query:
        return _empty_metrics()

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


def _build_cell(
    *,
    logical_name: str,
    physical_name: str,
    filter_key: str,
    metrics: dict[str, float | None],
    per_query: list[dict[str, Any]],
    unfiltered_p95: float | None,
    unfiltered_baseline_present: bool,
) -> dict[str, Any]:
    """Construct a single (index × filter) result cell."""
    p95_filt = metrics.get("p95_ms")
    if not unfiltered_baseline_present:
        flag: bool | str = "unknown"
    else:
        flag = flag_p95_degradation(p95_filt, unfiltered_p95, DEGRADATION_THRESHOLD)
    ratio: float | None
    if (
        isinstance(p95_filt, (int, float))
        and isinstance(unfiltered_p95, (int, float))
        and unfiltered_p95 > 0
    ):
        ratio = round(float(p95_filt) / float(unfiltered_p95), 3)
    else:
        ratio = None
    return {
        "index": logical_name,
        "index_name": physical_name,
        "filter": filter_key,
        "filter_clause": FILTERS[filter_key]["clause"],
        "filter_selectivity_pct_estimated": FILTERS[filter_key]["selectivity_pct_estimated"],
        "filter_strategy": filter_strategy_for(logical_name),
        "metrics": metrics,
        "unfiltered_p95_ms": unfiltered_p95,
        "p95_ratio_filtered_over_unfiltered": ratio,
        "p95_degradation_threshold": DEGRADATION_THRESHOLD,
        "p95_degradation_flag": flag,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Dry-run payload
# ---------------------------------------------------------------------------


def dry_run_payload(
    index_names: Sequence[str],
    filter_keys: Sequence[str],
    index_map: dict[str, str],
    eval_path: Path,
    unfiltered_baseline: dict[str, float | None],
    unfiltered_baseline_present: bool,
) -> dict[str, Any]:
    """Return a schema-valid payload with empty metrics — no DB touched."""
    results: list[dict[str, Any]] = []
    for logical in index_names:
        physical = index_map.get(logical, logical)
        for fkey in filter_keys:
            results.append(
                _build_cell(
                    logical_name=logical,
                    physical_name=physical,
                    filter_key=fkey,
                    metrics=_empty_metrics(),
                    per_query=[],
                    unfiltered_p95=unfiltered_baseline.get(logical),
                    unfiltered_baseline_present=unfiltered_baseline_present,
                )
            )
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "eval_path": str(eval_path),
        "model_name": MODEL_NAME,
        "env": {},
        "filters": {k: FILTERS[k] for k in filter_keys},
        "degradation_threshold": DEGRADATION_THRESHOLD,
        "unfiltered_baseline_path": str(DEFAULT_UNFILTERED_BASELINE),
        "unfiltered_baseline_present": unfiltered_baseline_present,
        "results": results,
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


def _fetch_ground_truth_relevant(conn: Any, bibcode: str, filter_clause: str) -> set[str]:
    """Return citation-neighbours that also satisfy the active filter.

    The filter is applied against the ``papers`` table so the ground truth
    shrinks to the set of relevant docs that would actually be reachable
    through a filtered search.
    """
    sql = f"""
        SELECT target_bibcode FROM citation_edges
          WHERE source_bibcode = %s
            AND target_bibcode IN (
              SELECT bibcode FROM papers WHERE {filter_clause}
            )
        UNION
        SELECT source_bibcode FROM citation_edges
          WHERE target_bibcode = %s
            AND source_bibcode IN (
              SELECT bibcode FROM papers WHERE {filter_clause}
            )
    """
    with conn.cursor() as cur:
        cur.execute(sql, (bibcode, bibcode))
        return {row[0] for row in cur.fetchall()}


def _run_filtered_hnsw_query(
    conn: Any,
    query_embedding_text: str,
    filter_clause: str,
    limit: int,
    exclude_bibcode: str,
) -> tuple[list[str], float]:
    """Filtered nearest-neighbour query on the HNSW baseline.

    Sets ``hnsw.iterative_scan = 'relaxed_order'`` at session scope before the
    query so pgvector emits additional candidates past the filter.
    """
    sql = (
        "SELECT pe.bibcode FROM paper_embeddings pe "
        "JOIN papers p ON p.bibcode = pe.bibcode "
        "WHERE pe.model_name = %s AND pe.bibcode != %s "
        f"AND {filter_clause} "
        "ORDER BY pe.embedding <=> %s::halfvec LIMIT %s"
    )
    with conn.cursor() as cur:
        # SET LOCAL scopes to the current transaction; use a short txn.
        cur.execute("BEGIN")
        try:
            cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
            t0 = time.perf_counter()
            cur.execute(sql, (MODEL_NAME, exclude_bibcode, query_embedding_text, int(limit)))
            rows = cur.fetchall()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
    return [r[0] for r in rows], latency_ms


def _run_filtered_diskann_query(
    conn: Any,
    query_embedding_text: str,
    filter_clause: str,
    limit: int,
    exclude_bibcode: str,
) -> tuple[list[str], float]:
    """Filtered nearest-neighbour query on a DiskANN variant.

    pgvectorscale supports native WHERE filtering; no session hint required.
    """
    sql = (
        "SELECT pe.bibcode FROM paper_embeddings pe "
        "JOIN papers p ON p.bibcode = pe.bibcode "
        "WHERE pe.model_name = %s AND pe.bibcode != %s "
        f"AND {filter_clause} "
        "ORDER BY pe.embedding <=> %s::halfvec LIMIT %s"
    )
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (MODEL_NAME, exclude_bibcode, query_embedding_text, int(limit)))
        rows = cur.fetchall()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return [r[0] for r in rows], latency_ms


def _run_filtered_query(
    conn: Any,
    logical_index: str,
    query_embedding_text: str,
    filter_clause: str,
    limit: int,
    exclude_bibcode: str,
) -> tuple[list[str], float]:
    """Dispatch to the HNSW or DiskANN filtered-query helper."""
    if logical_index == "hnsw":
        return _run_filtered_hnsw_query(
            conn, query_embedding_text, filter_clause, limit, exclude_bibcode
        )
    return _run_filtered_diskann_query(
        conn, query_embedding_text, filter_clause, limit, exclude_bibcode
    )


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    dsn: str,
    eval_path: Path,
    index_names: Sequence[str],
    filter_keys: Sequence[str],
    index_map: dict[str, str],
    unfiltered_baseline: dict[str, float | None],
    unfiltered_baseline_present: bool,
    limit: int = DEFAULT_LIMIT,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Execute the full filtered benchmark — DB-heavy; never called in tests."""
    import psycopg

    # Environment metadata (best-effort).
    try:
        from pgvs_bench_env import capture_env  # type: ignore[import-not-found]
        env = capture_env(dsn=dsn)
    except Exception as exc:  # noqa: BLE001 - env capture must not abort the run
        logger.warning("capture_env failed: %s", exc)
        env = {}

    queries = load_queries_from_eval(eval_path)
    if max_queries is not None:
        queries = queries[:max_queries]
    logger.info("Loaded %d queries from %s", len(queries), eval_path)

    payload: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "eval_path": str(eval_path),
        "model_name": MODEL_NAME,
        "env": env,
        "filters": {k: FILTERS[k] for k in filter_keys},
        "degradation_threshold": DEGRADATION_THRESHOLD,
        "unfiltered_baseline_path": str(DEFAULT_UNFILTERED_BASELINE),
        "unfiltered_baseline_present": unfiltered_baseline_present,
        "results": [],
    }

    with psycopg.connect(dsn) as conn:
        conn.autocommit = True

        # Pre-fetch query embeddings once per seed.
        query_ctx: list[dict[str, Any]] = []
        for bib in queries:
            emb = _fetch_query_embedding(conn, bib)
            if emb is None:
                logger.warning("Skipping %s: no INDUS embedding found", bib)
                continue
            emb_text = "[" + ",".join(f"{v:.8f}" for v in emb) + "]"
            query_ctx.append({"seed_bibcode": bib, "embedding_text": emb_text})

        # For each (filter, index) cell, run all queries.
        for fkey in filter_keys:
            fclause = FILTERS[fkey]["clause"]
            # Ground-truth relevants depend on the filter — recompute per seed.
            per_seed_relevant: dict[str, set[str]] = {}
            for ctx in query_ctx:
                per_seed_relevant[ctx["seed_bibcode"]] = _fetch_ground_truth_relevant(
                    conn, ctx["seed_bibcode"], fclause
                )

            for logical in index_names:
                physical = index_map.get(logical, logical)
                logger.info("Running filter=%s index=%s (%s)", fkey, logical, physical)
                per_query: list[dict[str, Any]] = []
                for ctx in query_ctx:
                    ranked, latency_ms = _run_filtered_query(
                        conn,
                        logical,
                        ctx["embedding_text"],
                        fclause,
                        limit,
                        ctx["seed_bibcode"],
                    )
                    relevant = per_seed_relevant.get(ctx["seed_bibcode"], set())
                    row = _per_query_metrics(ranked, relevant, latency_ms)
                    row["seed_bibcode"] = ctx["seed_bibcode"]
                    row["relevant_count"] = len(relevant)
                    per_query.append(row)
                metrics = _summarize(per_query)
                payload["results"].append(
                    _build_cell(
                        logical_name=logical,
                        physical_name=physical,
                        filter_key=fkey,
                        metrics=metrics,
                        per_query=per_query,
                        unfiltered_p95=unfiltered_baseline.get(logical),
                        unfiltered_baseline_present=unfiltered_baseline_present,
                    )
                )

    return payload


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmt_flag(v: Any) -> str:
    if v is True:
        return "YES"
    if v is False:
        return "no"
    return "unknown"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the filtered-queries summary MD.

    The table includes an explicit ``>=2x degradation under filter`` flag
    column so reviewers can scan a single column to find regressions.
    """
    lines: list[str] = []
    lines.append("# Filtered-Query Retrieval Benchmark")
    lines.append("")
    lines.append(f"- Timestamp: `{payload.get('timestamp', '')}`")
    lines.append(f"- Run ID: `{payload.get('run_id', '')}`")
    lines.append(f"- Dry run: `{payload.get('dry_run', False)}`")
    lines.append(f"- Model: `{payload.get('model_name', MODEL_NAME)}`")
    lines.append(f"- Eval queries: `{payload.get('eval_path', '')}`")
    lines.append(
        f"- Unfiltered baseline: `{payload.get('unfiltered_baseline_path', '')}` "
        f"(present: `{payload.get('unfiltered_baseline_present', False)}`)"
    )
    lines.append(
        f"- Degradation threshold: `>= {payload.get('degradation_threshold', DEGRADATION_THRESHOLD)}x` unfiltered p95"
    )
    lines.append("")

    # Filter definitions.
    lines.append("## Filter configurations")
    lines.append("")
    lines.append("| Key | WHERE clause | Est. selectivity |")
    lines.append("|-----|--------------|------------------|")
    for fkey, fdef in payload.get("filters", {}).items():
        lines.append(
            f"| {fkey} | `{fdef.get('clause','')}` "
            f"| ~{fdef.get('selectivity_pct_estimated','?')}% |"
        )
    lines.append("")
    lines.append(
        "F1 is `year = 2024` (pgvector HNSW uses "
        "`hnsw.iterative_scan = 'relaxed_order'`; DiskANN uses native WHERE). "
        "F2 is `arxiv_class && '{astro-ph.GA,astro-ph.SR}'::text[]`."
    )
    lines.append("")

    # p95 + degradation flag table (the headline).
    lines.append("## p95 latency & degradation flag")
    lines.append("")
    lines.append(
        "| Index | Filter | p95 filt (ms) | p95 unfilt (ms) | ratio | "
        ">=2x degradation under filter |"
    )
    lines.append(
        "|-------|--------|---------------|-----------------|-------|"
        "-------------------------------|"
    )
    for cell in payload.get("results", []):
        m = cell.get("metrics", {})
        lines.append(
            f"| {cell.get('index','')} ({cell.get('index_name','')}) "
            f"| {cell.get('filter','')} "
            f"| {_fmt(m.get('p95_ms'))} "
            f"| {_fmt(cell.get('unfiltered_p95_ms'))} "
            f"| {_fmt(cell.get('p95_ratio_filtered_over_unfiltered'))} "
            f"| {_fmt_flag(cell.get('p95_degradation_flag'))} |"
        )
    lines.append("")

    # Quality metrics table.
    lines.append("## Quality metrics")
    lines.append("")
    lines.append(
        "| Index | Filter | nDCG@10 | Recall@10 | Recall@20 | MRR | p50 (ms) |"
    )
    lines.append(
        "|-------|--------|---------|-----------|-----------|-----|----------|"
    )
    for cell in payload.get("results", []):
        m = cell.get("metrics", {})
        lines.append(
            f"| {cell.get('index','')} "
            f"| {cell.get('filter','')} "
            f"| {_fmt(m.get('ndcg_at_10'))} "
            f"| {_fmt(m.get('recall_at_10'))} "
            f"| {_fmt(m.get('recall_at_20'))} "
            f"| {_fmt(m.get('mrr'))} "
            f"| {_fmt(m.get('p50_ms'))} |"
        )
    lines.append("")

    # Filter strategy notes.
    lines.append("## Filter strategy per index")
    lines.append("")
    lines.append("| Index | Strategy |")
    lines.append("|-------|----------|")
    seen: set[str] = set()
    for cell in payload.get("results", []):
        name = cell.get("index", "")
        if name in seen:
            continue
        seen.add(name)
        lines.append(f"| {name} | {cell.get('filter_strategy','')} |")
    lines.append("")

    if payload.get("dry_run"):
        lines.append("## Dry Run")
        lines.append("")
        lines.append(payload.get("explanation", "Dry run — no measurements taken."))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_index_map(spec: str | None, defaults: dict[str, str]) -> dict[str, str]:
    """Parse ``name=index_name,...`` into a dict (fall back to defaults)."""
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


def _resolve_filter_keys(choice: str) -> list[str]:
    if choice == "both":
        return ["f1", "f2"]
    return [choice]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filtered-query retrieval benchmark — F1 year, F2 arxiv_class. "
            "Measures nDCG@10, Recall@{10,20}, MRR, p50, p95 latency per "
            "(index, filter) pair and flags indexes whose p95 under filter "
            "is >= 2x their unfiltered p95. Refuses production DSN (dbname=scix)."
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
        "--filter",
        dest="filter_choice",
        choices=FILTER_CHOICES,
        default="both",
        help=(
            f"Filter configuration to run: {FILTER_CHOICES}. Default: both. "
            "F1 = year = 2024 (~10%% selectivity); "
            "F2 = arxiv_class overlap with astro-ph.{GA,SR} (~20%% selectivity)."
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
        "--indexes",
        default=DEFAULT_INDEXES,
        help=(
            f"Comma-separated logical index names (default: '{DEFAULT_INDEXES}'). "
            "Mapped to physical index names via --index-map."
        ),
    )
    parser.add_argument(
        "--index-map",
        default=None,
        help=(
            "Optional override for logical->physical index mapping, e.g. "
            "'hnsw=my_hnsw,v1=my_v1'."
        ),
    )
    parser.add_argument(
        "--unfiltered-baseline",
        type=Path,
        default=DEFAULT_UNFILTERED_BASELINE,
        help=(
            f"Path to unfiltered retrieval_quality.json (default: "
            f"{DEFAULT_UNFILTERED_BASELINE}). Missing file -> flag='unknown'."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Top-K limit per query (default: 20).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Debug: cap number of queries (default: no cap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit a schema-valid JSON + MD with empty per-cell metrics. "
            "No DB connections opened."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / DEFAULT_OUT_JSON_NAME
    md_path = out_dir / DEFAULT_OUT_MD_NAME
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    md_path.write_text(render_markdown(payload))
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return json_path, md_path


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

    filter_keys = _resolve_filter_keys(args.filter_choice)

    baseline_path = Path(args.unfiltered_baseline)
    unfiltered_baseline = load_unfiltered_baseline(baseline_path)
    baseline_present = bool(unfiltered_baseline)

    if args.dry_run:
        payload = dry_run_payload(
            index_names=index_names,
            filter_keys=filter_keys,
            index_map=index_map,
            eval_path=args.eval_path,
            unfiltered_baseline=unfiltered_baseline,
            unfiltered_baseline_present=baseline_present,
        )
    else:
        payload = run_benchmark(
            dsn=dsn,
            eval_path=args.eval_path,
            index_names=index_names,
            filter_keys=filter_keys,
            index_map=index_map,
            unfiltered_baseline=unfiltered_baseline,
            unfiltered_baseline_present=baseline_present,
            limit=args.limit,
            max_queries=args.max_queries,
        )

    _write_outputs(payload, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
