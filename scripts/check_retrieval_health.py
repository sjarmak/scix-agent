#!/usr/bin/env python3
"""Probe the production lexical, body-BM25, and dense retrieval lanes.

The process performs one known-good hybrid query, validates every lane's
timing, checks for degraded results, and verifies the Qdrant collection count
against the PostgreSQL synchronization watermark.  It exits after one probe;
it is not a connection holder or cache warmer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import check_pipeline_health as pipeline_health  # noqa: E402

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402
from scix.qdrant_dense import INDUS_COLLECTION, dense_client  # noqa: E402
from scix.search import SearchResult, hybrid_search  # noqa: E402

logger = logging.getLogger("check_retrieval_health")

DEFAULT_QUERY = "galaxy formation"
DEFAULT_POINT_TOLERANCE = 500
NOTIFY_LABEL = "retrieval-health"
NOTIFY_TITLE = "three-lane retrieval health breach"
NOTIFY_SUBJECT = "The three-lane retrieval health prober"
REPRODUCE_COMMAND = ".venv/bin/python scripts/check_retrieval_health.py --allow-prod"

EXPECTED_POINTS_SQL = "SELECT count(*) FROM indus_qdrant_synced"
LANE_TIMINGS = (
    ("lexical", "lexical_ms"),
    ("body_bm25", "body_lexical_ms"),
    ("dense", "vector_ms"),
)


@dataclass(frozen=True)
class ProbeResult:
    """One independently reportable health assertion."""

    name: str
    ok: bool
    detail: str


def query_expected_points(conn: psycopg.Connection) -> int:
    """Return the number of points the Qdrant serving collection should hold."""
    with conn.cursor() as cursor:
        cursor.execute(EXPECTED_POINTS_SQL)
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("expected-point-count query returned no row")
    return int(row[0])


def _collection_status(info: Any) -> str:
    status = info.status
    value = getattr(status, "value", status)
    return str(value).lower()


def _lane_results(result: SearchResult) -> list[ProbeResult]:
    dropped = tuple(result.metadata.get("dropped_lanes") or ())
    has_results = bool(result.papers)
    checks: list[ProbeResult] = []
    for name, timing_key in LANE_TIMINGS:
        timing = float(result.timing_ms.get(timing_key, 0.0))
        reasons: list[str] = []
        if timing <= 0:
            reasons.append(f"{timing_key}={timing:g}ms; must be > 0")
        if dropped:
            reasons.append(f"dropped_lanes={list(dropped)!r}; expected []")
        if not has_results:
            reasons.append("known-good query returned no papers")
        detail = "; ".join(reasons) if reasons else f"{timing_key}={timing:.2f}ms"
        checks.append(ProbeResult(name, not reasons, detail))
    return checks


def _collection_result(info: Any, *, expected_points: int, tolerance: int) -> ProbeResult:
    status = _collection_status(info)
    points = int(info.points_count)
    difference = abs(points - expected_points)
    reasons: list[str] = []
    if status != "green":
        reasons.append(f"status={status!r}; expected 'green'")
    if difference > tolerance:
        reasons.append(
            f"points_count={points}, expected={expected_points}, difference={difference} "
            f"outside tolerance={tolerance}"
        )
    detail = (
        "; ".join(reasons)
        if reasons
        else f"status=green, points_count={points}, expected={expected_points}, tolerance={tolerance}"
    )
    return ProbeResult("qdrant_collection", not reasons, detail)


def probe_retrieval(
    conn: psycopg.Connection,
    query_embedding: list[float],
    qdrant_client: Any,
    *,
    query: str,
    expected_points: int,
    point_tolerance: int,
    search_fn: Callable[..., SearchResult] = hybrid_search,
) -> list[ProbeResult]:
    """Exercise all retrieval lanes and the Qdrant collection in one pass."""
    search_result = search_fn(
        conn,
        query,
        query_embedding,
        model_name="indus",
        include_body=True,
    )
    collection_info = qdrant_client.get_collection(INDUS_COLLECTION)
    return [
        *_lane_results(search_result),
        _collection_result(
            collection_info,
            expected_points=expected_points,
            tolerance=point_tolerance,
        ),
    ]


def render(results: list[ProbeResult]) -> str:
    lines = [
        f"{'PASS' if result.ok else 'FAIL'} {result.name}: {result.detail}" for result in results
    ]
    failures = [result for result in results if not result.ok]
    if failures:
        lines.append(f"retrieval health: {len(failures)}/{len(results)} checks FAILED")
    else:
        lines.append(f"retrieval health: all {len(results)} checks passed")
    return "\n".join(lines)


def _embed_query(query: str) -> list[float]:
    from scix.embed import embed_batch, load_model

    model, tokenizer = load_model("indus", device="cpu")
    vectors = embed_batch(model, tokenizer, [query], batch_size=1, pooling="mean")
    if len(vectors) != 1 or not vectors[0]:
        raise RuntimeError("INDUS query encoder returned no vector")
    return list(vectors[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe all three retrieval lanes end to end")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN (read-only)")
    parser.add_argument("--allow-prod", action="store_true", help="Allow the production DSN")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Known-good query text")
    parser.add_argument(
        "--point-tolerance",
        type=int,
        default=DEFAULT_POINT_TOLERANCE,
        help="Maximum absolute difference from indus_qdrant_synced count",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Maintain a deduplicated retrieval-health bead on breach/recovery",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.point_tolerance < 0:
        logger.error("--point-tolerance must be non-negative")
        return 2
    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing production DSN %s without --allow-prod",
            redact_dsn(args.dsn),
        )
        return 2
    if not os.environ.get("QDRANT_URL"):
        detail = "QDRANT_URL is unset; the production dense lane cannot be probed"
        logger.error(detail)
        if not args.notify:
            return 2
        results = [ProbeResult("probe_execution", False, detail)]
    else:
        try:
            vector = _embed_query(args.query)
            with get_connection(args.dsn) as conn:
                expected_points = query_expected_points(conn)
                results = probe_retrieval(
                    conn,
                    vector,
                    dense_client(timeout=10.0),
                    query=args.query,
                    expected_points=expected_points,
                    point_tolerance=args.point_tolerance,
                )
        except Exception as exc:  # noqa: BLE001 - CLI trust boundary
            logger.exception("retrieval probe failed before producing complete results: %s", exc)
            results = [ProbeResult("probe_execution", False, f"{type(exc).__name__}: {exc}")]

    print(render(results))
    if args.notify:
        alert_results = [
            pipeline_health.CheckResult(result.name, result.ok, result.detail) for result in results
        ]
        try:
            action = pipeline_health.notify(
                alert_results,
                now=dt.datetime.now(dt.timezone.utc),
                label=NOTIFY_LABEL,
                title=NOTIFY_TITLE,
                subject=NOTIFY_SUBJECT,
                reproduce_command=REPRODUCE_COMMAND,
            )
        except (pipeline_health.NotifyError, OSError, subprocess.SubprocessError) as exc:
            pipeline_health.report_notification_failure(exc)
            return 3
        logger.info("notification: %s", action)

    return 1 if any(not result.ok for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
