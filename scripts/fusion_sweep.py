#!/usr/bin/env python3
"""Fusion-calibration sweep over the retrieval gold set.

Motivation
----------
``dfba`` banked a dense INDUS nDCG@10 of ~0.864 against a BM25 nDCG@10 of
~0.088, yet the *naive* Reciprocal Rank Fusion hybrid (dense + BM25 fused
with equal rank weight) measurably HURTS top-rank quality versus dense alone:
the weak BM25 lane injects low-quality candidates near the top of the fused
list. That is the open retrieval question blocking a clean ADASS hybrid claim.

This script reuses the metric primitives and gold-set loader from
``eval_retrieval_50q.py`` and sweeps several fusion strategies over the
separate dense and BM25 lanes so we can report (a) a *defensible* hybrid
number that beats dense-alone, and (b) a principled production fusion config.

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

    # Live sweep over the 50q gold (writes results/fusion_sweep_v1.md)
    python scripts/fusion_sweep.py

    # Live sweep over a larger gold set (e.g. h2sj recall_gold_v1 once it lands)
    python scripts/fusion_sweep.py --queries eval/recall_gold_v1.jsonl

This script is READ-ONLY against the corpus: it only issues SELECT/kNN queries.
It still needs a worker + an INDUS encode, so run it under ``scix-batch`` during
an operator RAM window (see CLAUDE.md memory-isolation rules).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# Make ``src`` and the sibling eval module importable from a checkout root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the gold loader + metric primitives — single source of truth.
from eval_retrieval_50q import (  # noqa: E402  (path injected above)
    RECALL_K,
    RRF_K,
    EvalQuery,
    _dedupe_preserving_order,
    aggregate_metrics,
    load_queries,
    recall_at_k,
    rrf_fuse_bibcodes,
    score_query,
)

logger = logging.getLogger("fusion_sweep")

# ---------------------------------------------------------------------------
# Constants / sweep grids
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: str = "eval/retrieval_50q.jsonl"
DEFAULT_OUTPUT: str = "results/fusion_sweep_v1.md"
DEFAULT_JSON_OUTPUT: str = "results/fusion_sweep_v1.json"
# Per-lane candidate pool: fetch generously so fusion has material to reorder
# and Recall@50 is not capped by a thin lane.
DEFAULT_POOL: int = 100

RRF_K_GRID: tuple[int, ...] = (10, 30, 60, 100)
WEIGHT_GRID: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95)
CUTOFF_GRID: tuple[int, ...] = (5, 10, 20)
LAMBDA_GRID: tuple[float, ...] = (0.05, 0.1, 0.2)


# A lane is a ranked list of (bibcode, raw_score), descending by score.
Lane = list[tuple[str, float]]


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


def _mean_recall(fused_by_query: list[tuple[EvalQuery, list[str]]], k: int) -> float:
    """Mean recall@k over (query, fused) pairs, skipping empty-gold queries.

    ``score_query`` only reports recall at the fixed ``RECALL_K`` cutoff; the
    recall gold set is graded at the shallower 10/20 cutoffs the bead asks for,
    so we recompute recall here from the same ``recall_at_k`` primitive.
    """
    vals = [recall_at_k(fused, list(q.gold_bibcodes), k) for q, fused in fused_by_query]
    scored = [v for v in vals if v is not None]
    return sum(scored) / len(scored) if scored else 0.0


def evaluate_config(
    config: FusionConfig,
    lanes_by_query: list[tuple[EvalQuery, Lane, Lane]],
    *,
    k: int,
) -> dict[str, Any]:
    """Apply one fusion config across all queries and aggregate metrics."""
    scored: list[tuple[EvalQuery, dict[str, float | None]]] = []
    fused_by_query: list[tuple[EvalQuery, list[str]]] = []
    for q, dense, bm25 in lanes_by_query:
        try:
            fused = config.run(dense, bm25)
        except Exception:
            logger.exception("fusion %s failed on %r — scoring empty", config.name, q.query)
            fused = []
        scored.append((q, score_query(fused, list(q.gold_bibcodes), k=k)))
        fused_by_query.append((q, fused))
    overall = aggregate_metrics([row for _, row in scored])
    overall["recall_at_10"] = _mean_recall(fused_by_query, 10)
    overall["recall_at_20"] = _mean_recall(fused_by_query, 20)
    # Bucket the report by whatever buckets the gold set actually uses (the 50q
    # set has four; the 1200q recall set has one, ``recall_decile``) so the
    # per-bucket table never renders rows for absent buckets.
    buckets_present = _dedupe_preserving_order(q.bucket for q, _ in scored)
    by_bucket = {
        bucket: aggregate_metrics([row for q, row in scored if q.bucket == bucket])
        for bucket in buckets_present
    }
    return {"overall": overall, "by_bucket": by_bucket}


def compute_lanes(
    conn: Any, queries: Sequence[EvalQuery], *, pool: int
) -> list[tuple[EvalQuery, Lane, Lane]]:
    """Encode + retrieve both lanes once per query (the expensive pass)."""
    from eval_retrieval_50q import _indus_encode

    out: list[tuple[EvalQuery, Lane, Lane]] = []
    for i, q in enumerate(queries, start=1):
        try:
            vec = _indus_encode(q.query)
            dense = dense_lane(conn, vec, pool=pool)
            bm25 = bm25_lane(conn, q.query, pool=pool)
        except Exception:
            logger.exception("lane retrieval failed for %r — using empty lanes", q.query)
            dense, bm25 = [], []
        out.append((q, dense, bm25))
        logger.info(
            "lanes %d/%d: dense=%d bm25=%d (%s)", i, len(queries), len(dense), len(bm25), q.bucket
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def render_markdown(
    results: dict[str, dict[str, Any]], *, queries_path: str, n_queries: int, k: int
) -> str:
    """Render the sweep as a Markdown report ranked by overall nDCG@k."""
    dense_ref = results.get("dense_only", {}).get("overall", {})
    dense_ndcg = float(dense_ref.get("ndcg_at_10", 0.0))

    bm25_ndcg = float(results.get("bm25_only", {}).get("overall", {}).get("ndcg_at_10", 0.0))
    # The bead's premise — "naive RRF HURTS top-rank vs dense-alone" — only
    # applies when dense is the dominant lane. When BM25 ranks at or above
    # dense on this gold set, the premise does not reproduce and any hybrid
    # "lift" is an artifact of a weak dense lane, not a transferable result.
    dense_dominates = dense_ndcg > bm25_ndcg

    rows = sorted(results.items(), key=lambda kv: -float(kv[1]["overall"].get("ndcg_at_10", 0.0)))

    lines: list[str] = []
    lines.append("# Fusion-calibration sweep — v1")
    lines.append("")
    lines.append(f"- Gold set: `{queries_path}` ({n_queries} queries)")
    lines.append(f"- nDCG/MRR cutoff: {k}; Recall cutoffs: 10, 20, {RECALL_K}")
    lines.append(
        "- Lanes: INDUS dense (Qdrant, ADR-013) vs combined PG BM25 "
        "(title+abstract ⊕ body, RRF). READ-ONLY."
    )
    lines.append("")
    lines.append("## Overall (ranked by nDCG@10)")
    lines.append("")
    lines.append(
        "| Fusion config | nDCG@10 | MRR@10 | Recall@10 | Recall@20 | Recall@50 | "
        "ΔnDCG vs dense_only |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, block in rows:
        o = block["overall"]
        delta = float(o.get("ndcg_at_10", 0.0)) - dense_ndcg
        lines.append(
            f"| {name} | {_fmt(o.get('ndcg_at_10', 0.0))} | "
            f"{_fmt(o.get('mrr_at_10', 0.0))} | {_fmt(o.get('recall_at_10', 0.0))} | "
            f"{_fmt(o.get('recall_at_20', 0.0))} | {_fmt(o.get('recall_at_50', 0.0))} | "
            f"{delta:+.4f} |"
        )
    lines.append("")

    beats = [
        (n, float(b["overall"].get("ndcg_at_10", 0.0)) - dense_ndcg)
        for n, b in rows
        if n not in ("dense_only", "bm25_only")
        and float(b["overall"].get("ndcg_at_10", 0.0)) > dense_ndcg
    ]
    # Single headline config used by BOTH the verdict and the per-bucket table,
    # so the report never crowns two different "winners".
    headline_name = beats[0][0] if beats else rows[0][0]
    headline = results[headline_name]

    lines.append("## Verdict")
    lines.append("")
    if not dense_dominates:
        lines.append(
            f"**Premise does NOT reproduce on this gold set.** dense_only nDCG@10 "
            f"{_fmt(dense_ndcg)} is *below* bm25_only {_fmt(bm25_ndcg)}, so the "
            f'"naive RRF hurts top-rank vs dense-alone" regime — which assumes a '
            f"dominant dense lane — does not hold here. Any hybrid that beats "
            f"dense-alone is merely outrunning a weak dense lane, NOT a defensible "
            f"paper number. A transferable fusion result needs a dense-dominant "
            f"instrument (e.g. `eval/recall_gold_v1.jsonl`, 1200q decile recall); "
            f"this 50q curated set has documented findability gaps and cannot "
            f"answer the question. Best raw config here is `{headline_name}` at "
            f"nDCG@10 {_fmt(headline['overall'].get('ndcg_at_10', 0.0))} — report "
            f"as diagnostic only."
        )
    elif beats:
        gain = beats[0][1]
        lines.append(
            f"**Winning fusion: `{headline_name}`** — nDCG@10 "
            f"{_fmt(headline['overall']['ndcg_at_10'])} "
            f"(+{gain:.4f} over dense-alone). A defensible hybrid number that "
            f"does NOT regress top-rank versus dense."
        )
    else:
        lines.append(
            f"**No fusion beats dense-alone** (dense_only nDCG@10 {_fmt(dense_ndcg)}). "
            f"Best hybrid is `{headline_name}` at "
            f"{_fmt(headline['overall'].get('ndcg_at_10', 0.0))}. The honest paper "
            f"claim is dense-alone for top-rank; hybrid only helps recall — "
            f"report that rather than a fabricated hybrid win."
        )
    lines.append("")

    lines.append(f"## Per-bucket nDCG@10 (`{headline_name}` vs dense_only)")
    lines.append("")
    lines.append("| Bucket | dense_only | " + headline_name + " |")
    lines.append("|---|---|---|")
    for bucket in headline.get("by_bucket", {}):
        d = float(
            results.get("dense_only", {})
            .get("by_bucket", {})
            .get(bucket, {})
            .get("ndcg_at_10", 0.0)
        )
        w = float(headline["by_bucket"].get(bucket, {}).get("ndcg_at_10", 0.0))
        lines.append(f"| {bucket} | {_fmt(d)} | {_fmt(w)} |")
    lines.append("")
    return "\n".join(lines)


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
        "--dry-run",
        action="store_true",
        help="Skip DB and model; validate wiring + emit a stub report",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    configs = build_sweep()

    if not args.queries.exists():
        if not args.dry_run:
            logger.error("queries file %s not found", args.queries)
            return 2
        queries: list[EvalQuery] = []
    else:
        queries = load_queries(args.queries)

    if args.dry_run:
        # Validate the full config grid runs on synthetic lanes (no DB/model).
        synth_dense: Lane = [("A", 0.9), ("B", 0.5), ("C", 0.1)]
        synth_bm25: Lane = [("C", 3.0), ("D", 2.0), ("A", 1.0)]
        sample = {c.name: c.run(synth_dense, synth_bm25) for c in configs}
        payload = {
            "dry_run": True,
            "queries_path": str(args.queries),
            "n_queries": len(queries),
            "n_configs": len(configs),
            "sample_fused_orderings": sample,
        }
        md = (
            "# Fusion-calibration sweep — v1 (dry-run stub)\n\n"
            f"- {len(configs)} fusion configs wired and validated on synthetic lanes.\n"
            f"- Gold set `{args.queries}`: {len(queries)} queries loaded.\n"
            "- Live run pending operator RAM window (needs INDUS encode).\n"
        )
        write_outputs(args.output, args.json_output, md, payload)
        logger.info("dry-run stub written to %s", args.output)
        return 0

    # Live sweep — connect once, encode + retrieve lanes once, then sweep configs.
    try:
        from scix.db import get_connection
    except ImportError:
        logger.exception("scix.db unavailable; cannot run live sweep")
        return 1

    conn = None
    try:
        conn = get_connection()
        lanes = compute_lanes(conn, queries, pool=args.pool)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    results = {c.name: evaluate_config(c, lanes, k=args.k) for c in configs}
    payload = {
        "dry_run": False,
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "pool": args.pool,
        "k": args.k,
        "results": results,
    }
    markdown = render_markdown(
        results, queries_path=str(args.queries), n_queries=len(queries), k=args.k
    )
    write_outputs(args.output, args.json_output, markdown, payload)
    logger.info("live sweep written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
