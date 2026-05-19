#!/usr/bin/env python3
"""Lexical-search recall regression eval across SCIX_LEXICAL_POOL settings.

Bead scix_experiments-a2fp (follow-up to 3t37 / itgl). The candidate-pool cap
in :func:`scix.search.lexical_search` (default 5000, see ``_LEXICAL_POOL_DEFAULT``)
bounds ``ts_rank_cd`` cost so common single-token queries don't blow the 30s
statement timeout. The cap trades recall for latency: when the match set
exceeds the pool, only the first ~POOL rows the bitmap heap scan returns get
ranked. This script measures that recall cost directly.

What it does
------------
Runs the 50-query gold set (``eval/retrieval_50q.jsonl``) against
``lexical_search`` once per pool setting in {1000, 5000, 50000, INF}, scoring
nDCG@10 and Recall@20 against ``gold_bibcodes``. INF (uncapped) is the recall
ceiling; every capped pool is reported as a paired delta against it.

Acceptance (a2fp): the default pool (5000) must drop nDCG@10 by <= 2 percentage
points vs INF. The paired delta is computed only over queries that scored
successfully under *both* settings, so an INF timeout never silently inflates a
capped pool's apparent score. On FAIL the per-bucket table shows which query
class drives the regression — raise ``_LEXICAL_POOL_DEFAULT`` or make the pool
a per-query-class tunable.

Operational notes
-----------------
* Read-only (SELECT only) — safe against prod ``scix``; it must run there
  because the eval needs the real corpus, not ``scix_test``.
* Multi-minute: 50 queries x 4 pools = 200 FTS calls, and the INF pass on
  broad concept terms is slow by construction (that is the regression being
  measured). Not for CI. Run under ``scix-batch`` to stay clear of the oomd
  cgroup kill — though FTS is light, the run is long.
* The eval connection raises ``statement_timeout`` (default 300s) above the
  30s prod default so the INF baseline can actually complete; otherwise the
  recall ceiling would itself be truncated by timeouts.

Usage
-----
::

    scix-batch python scripts/eval_lexical_recall_pool.py
    python scripts/eval_lexical_recall_pool.py --pools 5000,INF --quiet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# Make ``src`` and the sibling ``scripts/`` dir importable from a checkout root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the gold-set loader and metric primitives from the sibling harness so
# scoring stays identical to the canonical 50q eval (DRY — one definition).
from eval_retrieval_50q import (  # noqa: E402
    EvalQuery,
    _dedupe_preserving_order,
    load_queries,
    ndcg_at_10,
    recall_at_k,
)

logger = logging.getLogger("eval_lexical_recall_pool")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: str = "eval/retrieval_50q.jsonl"
DEFAULT_OUTPUT: str = "docs/eval/lexical_recall_pool_2026-05.json"
# Pool labels in eval order. INF maps to the SCIX_LEXICAL_POOL token that
# disables the cap (see scix.search._LEXICAL_POOL_UNBOUNDED).
DEFAULT_POOLS: tuple[str, ...] = ("1000", "5000", "50000", "INF")
# Must match scix.search._LEXICAL_POOL_DEFAULT — the pool the live MCP uses.
DEFAULT_POOL_LABEL: str = "5000"
BASELINE_POOL_LABEL: str = "INF"
RECALL_K: int = 20
NDCG_K: int = 10
# top-20 returned rows is enough for Recall@20 and nDCG@10 (reads top-10).
RETRIEVE_LIMIT: int = 20
# Acceptance threshold from bead a2fp: <= 2 percentage-point nDCG@10 drop.
NDCG_DROP_THRESHOLD_PP: float = 2.0
# Eval-connection statement timeout. Well above the 30s prod default so the
# uncapped INF baseline can finish on broad terms instead of timing out.
DEFAULT_STATEMENT_TIMEOUT_MS: int = 300_000
BUCKETS: tuple[str, ...] = (
    "title_matchable",
    "concept",
    "method",
    "author_specific",
)


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------


class QueryResult:
    """Outcome of one (query, pool) pair.

    ``error`` is ``None`` on success, otherwise a short reason string. Errored
    queries are excluded from averages and paired deltas rather than scored as
    zero — a timed-out INF baseline must not look like a recall floor.
    """

    __slots__ = ("query", "bucket", "ndcg", "recall", "latency_ms", "error", "n_hits")

    def __init__(
        self,
        *,
        query: str,
        bucket: str,
        ndcg: float | None,
        recall: float | None,
        latency_ms: float,
        error: str | None,
        n_hits: int,
    ) -> None:
        self.query = query
        self.bucket = bucket
        self.ndcg = ndcg
        self.recall = recall
        self.latency_ms = latency_ms
        self.error = error
        self.n_hits = n_hits

    @property
    def scored(self) -> bool:
        """True when the query ran cleanly and has non-empty gold to score."""
        return self.error is None and self.ndcg is not None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _pool_env_value(label: str) -> str:
    """Map a pool label to its SCIX_LEXICAL_POOL env-var value."""
    return label  # labels are already valid tokens ("1000", "INF", ...)


def _pool_cap(label: str) -> int | None:
    """Resolve a pool label to its integer row cap, or ``None`` for unbounded."""
    if label.lower() in {"inf", "all", "none"}:
        return None
    return int(label)


def match_set_size(
    conn: Any, query_text: str, ts_config: str = "scix_english"
) -> int:
    """Count the rows ``lexical_search`` would match before the candidate cap.

    Mirrors the ``cand`` CTE predicate (``p.tsv @@ plainto_tsquery(...)``) with
    no filters — the eval calls ``lexical_search`` filter-free. This is what
    decides whether a given pool cap actually truncates the candidate set: the
    cap only changes results for queries whose match set exceeds it.
    """
    sql = (
        f"SELECT count(*) FROM papers p "
        f"WHERE p.tsv @@ plainto_tsquery('{ts_config}', %s)"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (query_text,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def run_pool(
    conn: Any,
    pool_label: str,
    queries: Sequence[EvalQuery],
) -> list[QueryResult]:
    """Run every query through ``lexical_search`` at one pool setting.

    Sets ``SCIX_LEXICAL_POOL`` for the duration; ``_resolve_lexical_pool`` reads
    the env var on every call, so no module reload is needed.
    """
    from scix.search import lexical_search

    prev = os.environ.get("SCIX_LEXICAL_POOL")
    os.environ["SCIX_LEXICAL_POOL"] = _pool_env_value(pool_label)
    results: list[QueryResult] = []
    try:
        for q in queries:
            gold = list(q.gold_bibcodes)
            t0 = time.perf_counter()
            error: str | None = None
            retrieved: list[str] = []
            try:
                sr = lexical_search(conn, q.query, limit=RETRIEVE_LIMIT)
                retrieved = _dedupe_preserving_order(
                    p.get("bibcode") for p in sr.papers
                )
            except Exception as exc:  # noqa: BLE001 — record, don't score as 0
                # A failed SELECT leaves the connection in an aborted txn;
                # roll back so the next query starts clean.
                try:
                    conn.rollback()
                except Exception:
                    pass
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "pool=%s query=%r failed: %s", pool_label, q.query, error
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(
                QueryResult(
                    query=q.query,
                    bucket=q.bucket,
                    ndcg=ndcg_at_10(retrieved, gold) if error is None else None,
                    recall=(
                        recall_at_k(retrieved, gold, RECALL_K)
                        if error is None
                        else None
                    ),
                    latency_ms=latency_ms,
                    error=error,
                    n_hits=len(retrieved),
                )
            )
    finally:
        if prev is None:
            os.environ.pop("SCIX_LEXICAL_POOL", None)
        else:
            os.environ["SCIX_LEXICAL_POOL"] = prev
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: Sequence[QueryResult]) -> dict[str, Any]:
    """Overall + per-bucket means over scored queries, plus latency diagnostics."""
    scored = [r for r in results if r.scored]
    errored = [r for r in results if r.error is not None]

    def block(rows: Sequence[QueryResult]) -> dict[str, Any]:
        ok = [r for r in rows if r.scored]
        return {
            "ndcg_at_10": _mean([r.ndcg for r in ok]),  # type: ignore[misc]
            "recall_at_20": _mean([r.recall for r in ok]),  # type: ignore[misc]
            "n_scored": len(ok),
        }

    return {
        "overall": block(scored),
        "by_bucket": {b: block([r for r in results if r.bucket == b]) for b in BUCKETS},
        "n_queries": len(results),
        "n_errored": len(errored),
        "errored_queries": [
            {"query": r.query, "error": r.error} for r in errored
        ],
        "latency_ms": {
            "max": max((r.latency_ms for r in results), default=0.0),
            "mean": _mean([r.latency_ms for r in results]),
            "total": sum(r.latency_ms for r in results),
        },
    }


def paired_delta(
    pool_results: Sequence[QueryResult],
    baseline_results: Sequence[QueryResult],
) -> dict[str, Any]:
    """nDCG@10 / Recall@20 delta vs the INF baseline, paired per query.

    Computed only over queries that scored cleanly under *both* settings, so a
    timeout under either side cannot distort the delta. Positive delta = the
    capped pool scored *lower* than INF (a recall drop), reported in
    percentage points.
    """
    base_by_q = {r.query: r for r in baseline_results}
    paired_pool_ndcg: list[float] = []
    paired_base_ndcg: list[float] = []
    paired_pool_recall: list[float] = []
    paired_base_recall: list[float] = []
    by_bucket: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in pool_results:
        b = base_by_q.get(r.query)
        if b is None or not r.scored or not b.scored:
            continue
        paired_pool_ndcg.append(r.ndcg)  # type: ignore[arg-type]
        paired_base_ndcg.append(b.ndcg)  # type: ignore[arg-type]
        paired_pool_recall.append(r.recall)  # type: ignore[arg-type]
        paired_base_recall.append(b.recall)  # type: ignore[arg-type]
        by_bucket[r.bucket].append((r.ndcg, b.ndcg))  # type: ignore[arg-type]

    pool_ndcg = _mean(paired_pool_ndcg)
    base_ndcg = _mean(paired_base_ndcg)
    pool_recall = _mean(paired_pool_recall)
    base_recall = _mean(paired_base_recall)

    def drop_pp(pool: float | None, base: float | None) -> float | None:
        if pool is None or base is None:
            return None
        return (base - pool) * 100.0

    bucket_drops: dict[str, dict[str, Any]] = {}
    for bucket in BUCKETS:
        pairs = by_bucket.get(bucket, [])
        if not pairs:
            bucket_drops[bucket] = {"ndcg_drop_pp": None, "n_paired": 0}
            continue
        p_mean = sum(p for p, _ in pairs) / len(pairs)
        b_mean = sum(b for _, b in pairs) / len(pairs)
        bucket_drops[bucket] = {
            "ndcg_drop_pp": (b_mean - p_mean) * 100.0,
            "n_paired": len(pairs),
        }

    return {
        "n_paired": len(paired_pool_ndcg),
        "ndcg_at_10_pool": pool_ndcg,
        "ndcg_at_10_baseline": base_ndcg,
        "ndcg_drop_pp": drop_pp(pool_ndcg, base_ndcg),
        "recall_at_20_pool": pool_recall,
        "recall_at_20_baseline": base_recall,
        "recall_drop_pp": drop_pp(pool_recall, base_recall),
        "by_bucket": bucket_drops,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "   n/a"
    return f"{value * 100:6.2f}" if pct else f"{value:6.3f}"


def _fmt_pp(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:+5.2f}"


def render_report(payload: dict[str, Any]) -> str:
    """Human-readable summary table for stdout."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Lexical-search recall vs SCIX_LEXICAL_POOL (bead a2fp)")
    lines.append(f"gold set: {payload['queries_path']}  ({payload['n_queries']} queries)")
    lines.append("=" * 72)
    lines.append("")
    header = (
        f"{'pool':>8} | {'nDCG@10':>8} | {'Recall@20':>10} | {'errors':>6} | "
        f"{'max ms':>9} | {'cap hits':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for label in payload["pools_order"]:
        pb = payload["pools"][label]
        agg = pb["aggregate"]
        ov = agg["overall"]
        lines.append(
            f"{label:>8} | {_fmt(ov['ndcg_at_10']):>8} | "
            f"{_fmt(ov['recall_at_20']):>10} | {agg['n_errored']:>6} | "
            f"{agg['latency_ms']['max']:>9.0f} | {pb['n_cap_engaged']:>8}"
        )
    lines.append("")
    lines.append(
        "'cap hits' = queries whose match set exceeds the pool, i.e. where the "
        "cap actually truncates."
    )
    lines.append("")
    lines.append(f"Paired delta vs {BASELINE_POOL_LABEL} baseline (drop in percentage points):")
    dh = f"{'pool':>8} | {'nDCG drop':>9} | {'Recall drop':>11} | {'n paired':>8}"
    lines.append(dh)
    lines.append("-" * len(dh))
    for label in payload["pools_order"]:
        if label == BASELINE_POOL_LABEL:
            continue
        d = payload["pools"][label]["delta_vs_baseline"]
        lines.append(
            f"{label:>8} | {_fmt_pp(d['ndcg_drop_pp']):>9} | "
            f"{_fmt_pp(d['recall_drop_pp']):>11} | {d['n_paired']:>8}"
        )
    lines.append("")
    acc = payload["acceptance"]
    lines.append("Per-bucket nDCG@10 drop at default pool "
                 f"({DEFAULT_POOL_LABEL}) vs {BASELINE_POOL_LABEL}:")
    for bucket in BUCKETS:
        bd = acc["by_bucket"].get(bucket, {})
        lines.append(
            f"  {bucket:>16}: {_fmt_pp(bd.get('ndcg_drop_pp')):>6} pp "
            f"(n={bd.get('n_paired', 0)})"
        )
    lines.append("")
    verdict = "PASS" if acc["pass"] else "FAIL"
    drop = acc["ndcg_drop_pp"]
    drop_str = "n/a" if drop is None else f"{drop:+.2f} pp"
    lines.append(
        f"ACCEPTANCE [{verdict}]: default pool {DEFAULT_POOL_LABEL} nDCG@10 drop "
        f"= {drop_str} (threshold <= {NDCG_DROP_THRESHOLD_PP:.1f} pp)"
    )
    if not acc["pass"]:
        lines.append(
            "  -> raise scix.search._LEXICAL_POOL_DEFAULT, or make the pool a "
            "per-query-class tunable (see by-bucket drops above)."
        )
    elif acc["unexercised"]:
        lines.append(
            f"  CAVEAT: the cap was UNEXERCISED — no query's match set exceeds "
            f"the default pool ({DEFAULT_POOL_LABEL}); largest is "
            f"{acc['max_match_set']} rows. The PASS is vacuous: it shows the "
            f"50q gold set does not stress the cap, NOT that the cap is "
            f"recall-safe on broad single-token queries. A meaningful recall-"
            f"cost measurement needs a gold set with high-frequency terms."
        )
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_pools(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--pools requires at least one value")
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        # Validate each token resolves: an integer, or an unbounded keyword.
        if p.lower() not in {"inf", "all", "none"}:
            try:
                if int(p) <= 0:
                    raise ValueError
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"pool {p!r} must be a positive integer or INF/ALL/NONE"
                ) from None
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lexical-search recall regression eval across SCIX_LEXICAL_POOL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--queries", type=Path, default=Path(DEFAULT_QUERIES),
        help="Path to the JSONL gold set",
    )
    p.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT),
        help="Path to write the JSON results",
    )
    p.add_argument(
        "--pools", type=_parse_pools, default=list(DEFAULT_POOLS),
        help="Comma-separated SCIX_LEXICAL_POOL values to sweep",
    )
    p.add_argument(
        "--statement-timeout-ms", type=int, default=DEFAULT_STATEMENT_TIMEOUT_MS,
        help="statement_timeout for the eval connection (raised so INF finishes)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable report (still writes JSON)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)

    if not args.queries.exists():
        logger.error("queries file %s not found", args.queries)
        return 2
    queries = load_queries(args.queries)
    n_with_gold = sum(1 for q in queries if q.gold_bibcodes)
    logger.info(
        "loaded %d queries (%d with non-empty gold) from %s",
        len(queries), n_with_gold, args.queries,
    )

    pools = list(args.pools)
    if BASELINE_POOL_LABEL not in pools:
        logger.error(
            "--pools must include %s — it is the recall baseline for all deltas",
            BASELINE_POOL_LABEL,
        )
        return 2

    try:
        from scix.db import get_connection
    except ImportError:
        logger.exception("scix.db is unavailable; cannot run the eval")
        return 1

    conn = None
    pool_results: dict[str, list[QueryResult]] = {}
    match_sizes: dict[str, int] = {}
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # SET does not accept bound parameters; set_config() does. The
            # value must be text and is validated as an int by argparse.
            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (str(int(args.statement_timeout_ms)),),
            )
        conn.commit()
        # Pool-independent: count each query's full match set once. This is
        # what tells us whether any pool actually truncates.
        logger.info("counting match-set sizes for %d queries", len(queries))
        for q in queries:
            match_sizes[q.query] = match_set_size(conn, q.query)
        for label in pools:
            logger.info("running pool=%s over %d queries", label, len(queries))
            pool_results[label] = run_pool(conn, label, queries)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    baseline = pool_results[BASELINE_POOL_LABEL]
    pools_block: dict[str, Any] = {}
    for label in pools:
        results = pool_results[label]
        cap = _pool_cap(label)
        # A query is "cap-engaged" when its match set exceeds the pool — only
        # then can the cap change which rows get ranked. INF (cap=None) never
        # engages.
        n_engaged = (
            0 if cap is None
            else sum(1 for q in queries if match_sizes.get(q.query, 0) > cap)
        )
        block: dict[str, Any] = {
            "aggregate": aggregate(results),
            "n_cap_engaged": n_engaged,
        }
        if label != BASELINE_POOL_LABEL:
            block["delta_vs_baseline"] = paired_delta(results, baseline)
        pools_block[label] = block

    max_match_set = max(match_sizes.values(), default=0)

    # Acceptance: default-pool nDCG@10 drop vs INF baseline.
    if DEFAULT_POOL_LABEL in pools_block and "delta_vs_baseline" in pools_block[DEFAULT_POOL_LABEL]:
        default_delta = pools_block[DEFAULT_POOL_LABEL]["delta_vs_baseline"]
        drop_pp = default_delta["ndcg_drop_pp"]
        by_bucket = default_delta["by_bucket"]
        default_engaged = pools_block[DEFAULT_POOL_LABEL]["n_cap_engaged"]
    else:
        # Default pool not swept, or it *is* the baseline — no regression to test.
        default_delta = None
        drop_pp = None
        by_bucket = {}
        default_engaged = 0
    passed = drop_pp is None or drop_pp <= NDCG_DROP_THRESHOLD_PP
    acceptance = {
        "default_pool": DEFAULT_POOL_LABEL,
        "baseline_pool": BASELINE_POOL_LABEL,
        "ndcg_drop_pp": drop_pp,
        "threshold_pp": NDCG_DROP_THRESHOLD_PP,
        "pass": bool(passed),
        "by_bucket": by_bucket,
        # When the cap never truncates at the default pool the PASS proves only
        # that this gold set does not stress the cap — not that the cap is safe.
        "unexercised": default_engaged == 0,
        "n_cap_engaged_at_default": default_engaged,
        "max_match_set": max_match_set,
    }

    payload: dict[str, Any] = {
        "bead": "scix_experiments-a2fp",
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "n_queries_with_gold": n_with_gold,
        "retrieve_limit": RETRIEVE_LIMIT,
        "statement_timeout_ms": int(args.statement_timeout_ms),
        "pools_order": pools,
        "pools": pools_block,
        "match_set_sizes": dict(sorted(match_sizes.items(), key=lambda kv: -kv[1])),
        "acceptance": acceptance,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("results written to %s", args.output)

    if not args.quiet:
        print(render_report(payload))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
