#!/usr/bin/env python3
"""Fusion-calibration sweep for a known-item retrieval diagnostic.

Motivation
----------
``dfba`` banked a dense INDUS nDCG@10 of ~0.864 against a BM25 nDCG@10 of
~0.088, yet the *naive* Reciprocal Rank Fusion hybrid (dense + BM25 fused
with equal rank weight) measurably HURTS top-rank quality versus dense alone:
the weak BM25 lane injects low-quality candidates near the top of the fused
list. That is the open retrieval question blocking a clean ADASS hybrid claim.

The 1,200-query input uses a paper's exact title as its query and keeps that
paper first in ``gold_bibcodes``. It therefore measures known-item retrieval,
not general search relevance. The harness preserves all ten source deciles,
selects a fusion configuration on a deterministic stratified tuning split,
and reports it once on disjoint held-out confirmation queries.

Every successful run records per-query scores, paired bootstrap uncertainty,
the gold-file hash, git/model/Qdrant/corpus provenance, and a query-coverage
gate. Retrieval exceptions remain errors and fail the default 100% gate; they
are never converted into zero scores.

Fusion strategies swept
-----------------------
* ``dense_only``       — reference: the INDUS dense ranking unchanged.
* ``bm25_only``        — reference: the combined lexical ranking unchanged.
* ``naive_rrf``        — RRF(dense, bm25) with equal rank weight (the lane
                         that "hurts"). Swept over ``k``.
* ``weighted_sum``     — min-max-normalize each lane's raw scores to [0, 1],
                         then ``w*dense + (1-w)*bm25``. Swept over ``w``.
* ``rank_cutoff_rrf``  — RRF where BM25 contributes only its top-``cutoff``
                         ranks (dense contributes fully). Caps BM25's deep-tail
                         noise while keeping dense recall. Swept over ``cutoff``.
* ``dense_prior``      — ``norm_dense + lam*norm_bm25`` over the lane union with
                         small ``lam``, so dense anchors the ordering and BM25
                         only nudges. Swept over ``lam``.

Local-only: query encoding uses the local INDUS model via
``eval_retrieval_50q._indus_encode`` (no paid API SDKs — see project memory
``feedback_no_paid_apis``). The dense lane serves from Qdrant when
``QDRANT_URL`` is set (ADR-013); BM25 lanes serve from Postgres.

Usage
-----
::

    # Schema-only (no DB, no model) — validates wiring + fusion math
    python scripts/fusion_sweep.py --dry-run

    # Live known-item protocol (writes Markdown + full JSON evidence)
    scix-batch python scripts/fusion_sweep.py

This script is READ-ONLY against the corpus: it only issues SELECT/kNN queries.
It still needs a worker + an INDUS encode, so run it under ``scix-batch`` during
an operator RAM window (see CLAUDE.md memory-isolation rules).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

# Make ``src`` and the sibling eval module importable from a checkout root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the gold loader + metric primitives — single source of truth.
from eval_retrieval_50q import (  # noqa: E402  (path injected above)
    RRF_K,
    _dedupe_preserving_order,
    aggregate_metrics,
    recall_at_k,
    rrf_fuse_bibcodes,
    score_query,
)

from scix.eval.known_item_protocol import (  # noqa: E402
    EVALUATION_KIND,
    BenchmarkQuery,
    Lane,
    QueryLanes,
    enforce_coverage_gate,
    load_benchmark_queries,
    paired_bootstrap_delta,
    render_markdown,
    split_queries,
)

logger = logging.getLogger("fusion_sweep")

# ---------------------------------------------------------------------------
# Constants / sweep grids
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: str = "eval/recall_gold_v1.jsonl"
DEFAULT_OUTPUT: str = "results/fusion_sweep_known_item_v2.md"
DEFAULT_JSON_OUTPUT: str = "results/fusion_sweep_known_item_v2.json"
# Per-lane candidate pool: fetch generously so fusion has material to reorder
# and Recall@50 is not capped by a thin lane.
DEFAULT_POOL: int = 100
DEFAULT_TUNING_FRACTION: float = 0.5
DEFAULT_MIN_COVERAGE: float = 1.0
DEFAULT_SPLIT_SALT: str = "fusion-sweep-v2"
MODEL_NAME: str = "indus"
MODEL_ID: str = "nasa-impact/nasa-smd-ibm-st-v2"
QDRANT_COLLECTION: str = "scix_indus_v2_papers_s1"

RRF_K_GRID: tuple[int, ...] = (10, 30, 60, 100)
WEIGHT_GRID: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95)
CUTOFF_GRID: tuple[int, ...] = (5, 10, 20)
LAMBDA_GRID: tuple[float, ...] = (0.05, 0.1, 0.2)


# ---------------------------------------------------------------------------
# Pure fusion primitives — no DB, no model, unit-testable.
# Each returns a fused list of bibcodes (descending preference).
# ---------------------------------------------------------------------------


def min_max_normalize(lane: Lane) -> dict[str, float]:
    """Min-max normalize a lane's raw scores to [0, 1], keyed by bibcode.

    A lane with a single distinct score (or one item) maps every member to
    1.0 — the lane carries rank information but no usable spread, so we treat
    all of its members as equally (fully) relevant within the lane.
    """
    if not lane:
        return {}
    scores = [s for _, s in lane]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 0.0:
        return {bib: 1.0 for bib, _ in lane}
    return {bib: (s - lo) / span for bib, s in lane}


def fuse_dense_only(dense: Lane, bm25: Lane) -> list[str]:
    """Reference: the dense ranking unchanged."""
    return _dedupe_preserving_order(bib for bib, _ in dense)


def fuse_bm25_only(dense: Lane, bm25: Lane) -> list[str]:
    """Reference: the combined lexical ranking unchanged."""
    return _dedupe_preserving_order(bib for bib, _ in bm25)


def fuse_naive_rrf(dense: Lane, bm25: Lane, *, k: int = RRF_K) -> list[str]:
    """Equal-rank-weight RRF over the two lanes (the lane that 'hurts')."""
    return rrf_fuse_bibcodes([[bib for bib, _ in dense], [bib for bib, _ in bm25]], k_rrf=k)


def fuse_weighted_sum(dense: Lane, bm25: Lane, *, w_dense: float) -> list[str]:
    """Min-max-normalized weighted sum: ``w*dense + (1-w)*bm25``."""
    nd = min_max_normalize(dense)
    nb = min_max_normalize(bm25)
    w_bm25 = 1.0 - w_dense
    keys = set(nd) | set(nb)
    combined = {bib: w_dense * nd.get(bib, 0.0) + w_bm25 * nb.get(bib, 0.0) for bib in keys}
    return sorted(keys, key=lambda b: (-combined[b], b))


def fuse_rank_cutoff_rrf(dense: Lane, bm25: Lane, *, k: int = RRF_K, cutoff: int) -> list[str]:
    """RRF where BM25 contributes only its top-``cutoff`` ranks.

    Dense contributes its full ranking, so recall is preserved while BM25's
    weak deep tail can no longer inject candidates near the top.
    """
    dense_bibs = [bib for bib, _ in dense]
    bm25_bibs = [bib for bib, _ in bm25][: max(cutoff, 0)]
    return rrf_fuse_bibcodes([dense_bibs, bm25_bibs], k_rrf=k)


def fuse_dense_prior(dense: Lane, bm25: Lane, *, lam: float) -> list[str]:
    """Dense-anchored fusion: ``norm_dense + lam*norm_bm25`` over the union.

    With small ``lam`` the dense score (spanning [0, 1]) dominates the
    ordering; BM25 only nudges, and BM25-exclusive candidates (capped at
    ``lam``) sort below the dense hits. Dense is the prior, BM25 the evidence.
    """
    nd = min_max_normalize(dense)
    nb = min_max_normalize(bm25)
    keys = set(nd) | set(nb)
    combined = {bib: nd.get(bib, 0.0) + lam * nb.get(bib, 0.0) for bib in keys}
    return sorted(keys, key=lambda b: (-combined[b], b))


@dataclass(frozen=True)
class FusionConfig:
    """One point in the fusion sweep: a named strategy + its parameters."""

    name: str
    fuse: Callable[[Lane, Lane], list[str]]

    def run(self, dense: Lane, bm25: Lane) -> list[str]:
        return self.fuse(dense, bm25)


def build_sweep() -> list[FusionConfig]:
    """Construct the full grid of fusion configs to evaluate."""
    configs: list[FusionConfig] = [
        FusionConfig("dense_only", fuse_dense_only),
        FusionConfig("bm25_only", fuse_bm25_only),
    ]
    for k in RRF_K_GRID:
        configs.append(
            FusionConfig(f"naive_rrf(k={k})", lambda d, b, k=k: fuse_naive_rrf(d, b, k=k))
        )
    for w in WEIGHT_GRID:
        configs.append(
            FusionConfig(
                f"weighted_sum(w_dense={w})",
                lambda d, b, w=w: fuse_weighted_sum(d, b, w_dense=w),
            )
        )
    for c in CUTOFF_GRID:
        configs.append(
            FusionConfig(
                f"rank_cutoff_rrf(cutoff={c})",
                lambda d, b, c=c: fuse_rank_cutoff_rrf(d, b, cutoff=c),
            )
        )
    for lam in LAMBDA_GRID:
        configs.append(
            FusionConfig(
                f"dense_prior(lam={lam})",
                lambda d, b, lam=lam: fuse_dense_prior(d, b, lam=lam),
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Live lane retrieval — imports kept local so --dry-run avoids torch / DB.
# ---------------------------------------------------------------------------


def dense_lane(conn: Any, query_embedding: list[float], *, pool: int) -> Lane:
    """INDUS dense ranking as (bibcode, cosine_similarity), via vector_search."""
    from scix.search import vector_search

    result = vector_search(conn, query_embedding, model_name="indus", limit=pool)
    return [(p["bibcode"], float(p.get("score", 0.0))) for p in result.papers if "bibcode" in p]


def bm25_lane(conn: Any, query_text: str, *, pool: int) -> Lane:
    """Combined lexical ranking: RRF(title+abstract BM25, body BM25).

    Each sub-lane is ts_rank_cd-ranked in Postgres; we fuse them into one
    BM25 lane (matching how the baseline treats lexical) and attach the RRF
    score so weighted/dense-prior strategies have a usable per-item score.
    """
    from scix.search import lexical_search, lexical_search_body

    ta = lexical_search(conn, query_text, limit=pool)
    body = lexical_search_body(conn, query_text, limit=pool)
    ta_bibs = _dedupe_preserving_order(p["bibcode"] for p in ta.papers if "bibcode" in p)
    body_bibs = _dedupe_preserving_order(p["bibcode"] for p in body.papers if "bibcode" in p)
    # Fuse the two lexical sub-lanes into one ranking, then synthesize a
    # descending score from rank position so downstream normalization works.
    fused_bibs = rrf_fuse_bibcodes([ta_bibs, body_bibs], k_rrf=RRF_K)[:pool]
    n = len(fused_bibs)
    return [(bib, float(n - i)) for i, bib in enumerate(fused_bibs)]


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _aggregate_scores(rows: list[dict[str, float | None]]) -> dict[str, Any]:
    aggregate = aggregate_metrics(rows)
    for metric in ("recall_at_10", "recall_at_20"):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        aggregate[metric] = sum(values) / len(values) if values else 0.0
    return aggregate


def evaluate_config(
    config: FusionConfig,
    lanes_by_query: Sequence[QueryLanes],
    *,
    k: int,
) -> dict[str, Any]:
    """Apply one fusion config across all queries and aggregate metrics."""
    scored: list[tuple[BenchmarkQuery, dict[str, float | None]]] = []
    per_query: list[dict[str, Any]] = []
    for lane_result in lanes_by_query:
        if lane_result.error is not None:
            continue
        q = lane_result.query
        fused = config.run(lane_result.dense, lane_result.bm25)
        metrics = score_query(fused, list(q.gold_bibcodes), k=k)
        metrics["recall_at_10"] = recall_at_k(fused, list(q.gold_bibcodes), 10)
        metrics["recall_at_20"] = recall_at_k(fused, list(q.gold_bibcodes), 20)
        scored.append((q, metrics))
        per_query.append(
            {
                "query_id": q.query_id,
                "decile": q.decile,
                "ndcg_at_10": metrics["ndcg_at_10"],
                "mrr_at_10": metrics["mrr_at_10"],
                "recall_at_10": metrics["recall_at_10"],
                "recall_at_20": metrics["recall_at_20"],
                "recall_at_50": metrics["recall_at_50"],
            }
        )
    overall = _aggregate_scores([row for _, row in scored])
    by_decile = {
        str(decile): _aggregate_scores([row for q, row in scored if q.decile == decile])
        for decile in range(10)
    }
    return {"overall": overall, "by_decile": by_decile, "per_query": per_query}


def compute_lanes(conn: Any, queries: Sequence[BenchmarkQuery], *, pool: int) -> list[QueryLanes]:
    """Encode + retrieve both lanes once per query (the expensive pass)."""
    from eval_retrieval_50q import _indus_encode

    out: list[QueryLanes] = []
    for i, q in enumerate(queries, start=1):
        try:
            vec = _indus_encode(q.query)
            dense = dense_lane(conn, vec, pool=pool)
            bm25 = bm25_lane(conn, q.query, pool=pool)
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("lane retrieval failed for %r", q.query)
            try:
                conn.rollback()
            except Exception:
                logger.exception("rollback failed after retrieval error")
            dense, bm25 = [], []
        out.append(QueryLanes(query=q, dense=dense, bm25=bm25, error=error))
        logger.info(
            "lanes %d/%d: dense=%d bm25=%d (%s)", i, len(queries), len(dense), len(bm25), q.bucket
        )
    return out


def _safe_qdrant_endpoint(raw_url: str | None) -> str | None:
    if not raw_url:
        return None
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("QDRANT_URL must be an HTTP(S) URL with a hostname")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def collect_provenance(conn: Any, queries_path: Path) -> dict[str, Any]:
    """Capture enough immutable/runtime identity to reproduce or reject a run."""
    qdrant_endpoint = _safe_qdrant_endpoint(os.environ.get("QDRANT_URL"))
    if qdrant_endpoint is None:
        raise RuntimeError("QDRANT_URL is required for the ADR-013 INDUS benchmark lane")
    from eval_retrieval_50q import _indus_state

    model = _indus_state.get("model")
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not model_revision:
        raise RuntimeError("loaded INDUS model did not expose an immutable revision")
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with conn.cursor() as cur:
        cur.execute("SELECT max(version), max(applied_at) FROM schema_migrations")
        migration_version, migration_applied_at = cur.fetchone()
        cur.execute("SELECT reltuples::bigint FROM pg_class WHERE oid = 'papers'::regclass")
        paper_count_estimate = int(cur.fetchone()[0])
    info = conn.info
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_kind": EVALUATION_KIND,
        "gold": {
            "path": str(queries_path),
            "sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        },
        "git_revision": git_revision,
        "model": {
            "name": MODEL_NAME,
            "huggingface_id": MODEL_ID,
            "revision": model_revision,
        },
        "qdrant": {
            "endpoint": qdrant_endpoint,
            "collection": QDRANT_COLLECTION,
        },
        "corpus": {
            "database": info.dbname,
            "host": info.host,
            "port": info.port,
            "paper_count_estimate": paper_count_estimate,
            "schema_migration": migration_version,
            "schema_migration_applied_at": (
                migration_applied_at.isoformat() if migration_applied_at else None
            ),
        },
    }


def write_outputs(md_path: Path, json_path: Path, markdown: str, payload: dict[str, Any]) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fusion-calibration sweep over the retrieval gold set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--queries", type=Path, default=Path(DEFAULT_QUERIES), help="Path to JSONL gold set"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Path to write the Markdown report",
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path(DEFAULT_JSON_OUTPUT),
        help="Path to write the raw JSON results",
    )
    p.add_argument("--pool", type=int, default=DEFAULT_POOL, help="Per-lane candidate pool size")
    p.add_argument("--k", type=int, default=10, help="Cutoff for nDCG/MRR (Recall is always at 50)")
    p.add_argument(
        "--tuning-fraction",
        type=float,
        default=DEFAULT_TUNING_FRACTION,
        help="Per-decile fraction reserved for configuration selection",
    )
    p.add_argument(
        "--split-salt",
        default=DEFAULT_SPLIT_SALT,
        help="Frozen salt for deterministic stratified assignment",
    )
    p.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="Minimum successful query fraction; default fails closed on any error",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip DB and model; validate wiring + emit a stub report",
    )
    return p


def _load_queries(args: argparse.Namespace) -> list[BenchmarkQuery]:
    if not args.queries.exists():
        if not args.dry_run:
            raise FileNotFoundError(args.queries)
        return []
    return load_benchmark_queries(args.queries)


def _run_dry(
    args: argparse.Namespace, queries: list[BenchmarkQuery], configs: list[FusionConfig]
) -> int:
    dense: Lane = [("A", 0.9), ("B", 0.5), ("C", 0.1)]
    bm25: Lane = [("C", 3.0), ("D", 2.0), ("A", 1.0)]
    payload = {
        "dry_run": True,
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "n_configs": len(configs),
        "sample_fused_orderings": {config.name: config.run(dense, bm25) for config in configs},
    }
    markdown = (
        "# Fusion-calibration known-item diagnostic — v2 (dry-run stub)\n\n"
        f"- {len(configs)} fusion configs validated on synthetic lanes.\n"
        f"- Gold set `{args.queries}`: {len(queries)} queries loaded.\n"
        "- Live run pending operator RAM window (needs INDUS encode).\n"
    )
    write_outputs(args.output, args.json_output, markdown, payload)
    logger.info("dry-run stub written to %s", args.output)
    return 0


def _retrieve(
    args: argparse.Namespace, queries: list[BenchmarkQuery]
) -> tuple[list[QueryLanes], dict[str, Any], dict[str, Any]]:
    from scix.db import get_connection

    conn = None
    try:
        conn = get_connection()
        lanes = compute_lanes(conn, queries, pool=args.pool)
        coverage = enforce_coverage_gate(lanes, minimum=args.min_coverage)
        provenance = collect_provenance(conn, args.queries)
        return lanes, coverage, provenance
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                logger.exception("failed to close benchmark database connection")


def _evaluate_protocol(
    args: argparse.Namespace,
    configs: list[FusionConfig],
    tuning_queries: list[BenchmarkQuery],
    confirmation_queries: list[BenchmarkQuery],
    lanes: list[QueryLanes],
) -> tuple[dict[str, Any], dict[str, Any], FusionConfig, dict[str, Any]]:
    by_query_id = {row.query.query_id: row for row in lanes if row.error is None}
    tuning_lanes = [
        by_query_id[query.query_id] for query in tuning_queries if query.query_id in by_query_id
    ]
    confirmation_lanes = [
        by_query_id[query.query_id]
        for query in confirmation_queries
        if query.query_id in by_query_id
    ]
    tuning_results = {
        config.name: evaluate_config(config, tuning_lanes, k=args.k) for config in configs
    }
    candidates = [config for config in configs if config.name not in {"dense_only", "bm25_only"}]
    selected = max(
        candidates,
        key=lambda config: float(tuning_results[config.name]["overall"]["ndcg_at_10"]),
    )
    confirmation_configs = [
        next(config for config in configs if config.name == "dense_only"),
        next(config for config in configs if config.name == "bm25_only"),
        selected,
    ]
    confirmation_results = {
        config.name: evaluate_config(config, confirmation_lanes, k=args.k)
        for config in confirmation_configs
    }
    uncertainty = {
        metric: paired_bootstrap_delta(
            confirmation_results[selected.name]["per_query"],
            confirmation_results["dense_only"]["per_query"],
            metric=metric,
        )
        for metric in ("ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_50")
    }
    return tuning_results, confirmation_results, selected, uncertainty


def _run_live(
    args: argparse.Namespace, queries: list[BenchmarkQuery], configs: list[FusionConfig]
) -> int:
    tuning_queries, confirmation_queries = split_queries(
        queries, tuning_fraction=args.tuning_fraction, salt=args.split_salt
    )
    lanes, coverage, provenance = _retrieve(args, queries)
    tuning_results, confirmation_results, selected, uncertainty = _evaluate_protocol(
        args, configs, tuning_queries, confirmation_queries, lanes
    )
    payload = {
        "dry_run": False,
        "protocol": {
            "evaluation_kind": EVALUATION_KIND,
            "tuning_fraction": args.tuning_fraction,
            "split_salt": args.split_salt,
            "selected_config": selected.name,
            "n_tuning": len(tuning_queries),
            "n_confirmation": len(confirmation_queries),
        },
        "provenance": provenance,
        "coverage": coverage,
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "pool": args.pool,
        "k": args.k,
        "tuning_results": tuning_results,
        "confirmation_results": confirmation_results,
        "paired_uncertainty_vs_dense": uncertainty,
    }
    markdown = render_markdown(
        confirmation_results,
        queries_path=str(args.queries),
        n_queries=len(confirmation_queries),
        k=args.k,
        paired_uncertainty=uncertainty,
        provenance=provenance,
    )
    write_outputs(args.output, args.json_output, markdown, payload)
    logger.info("live sweep written to %s", args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    try:
        queries = _load_queries(args)
        configs = build_sweep()
        return (
            _run_dry(args, queries, configs) if args.dry_run else _run_live(args, queries, configs)
        )
    except FileNotFoundError:
        logger.error("queries file %s not found", args.queries)
        return 2


if __name__ == "__main__":
    sys.exit(main())
