#!/usr/bin/env python3
"""Concurrent-query stress benchmark for pgvector/pgvectorscale indexes.

Runs the 50-query eval workload under concurrent load at configurable
thread counts (default 10 and 50) against the HNSW baseline and each
pgvectorscale StreamingDiskANN variant (v1/v2/v3). Each worker thread
pulls a random query from the eval set, executes it against each
configured index, and records wall-clock latency. At the end we report
sustained QPS, p50/p95/p99 latency, and total queries executed per
(index_name, thread_count).

Outputs:
    results/pgvs_benchmark/concurrent_stress.json
    results/pgvs_benchmark/concurrent_stress.md

Safety:
    Refuses to run against production DSN (dbname=scix). Uses
    ``assert_pilot_dsn`` mirroring scripts/build_hnsw_baseline.py.

Usage:
    python scripts/bench_pgvectorscale_concurrent.py --help
    python scripts/bench_pgvectorscale_concurrent.py --dry-run
    python scripts/bench_pgvectorscale_concurrent.py \
        --dsn "dbname=scix_pgvs_pilot" \
        --thread-counts 10,50 \
        --duration-seconds 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg.conninfo import conninfo_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bench_pgvectorscale_concurrent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT = Path("results/pgvs_benchmark/concurrent_stress.json")
DEFAULT_MD_OUT = Path("results/pgvs_benchmark/concurrent_stress.md")
DEFAULT_DSN = os.environ.get("SCIX_PILOT_DSN") or os.environ.get(
    "SCIX_DSN", "dbname=scix_pgvs_pilot"
)
DEFAULT_THREAD_COUNTS = "10,50"
DEFAULT_DURATION_SECONDS = 60
DEFAULT_QUERY_LIMIT = 10
DEFAULT_QUERY_SET_SIZE = 50
MODEL_NAME = "indus"

# Production database names this script must NEVER run against.
_PRODUCTION_DB_NAMES: frozenset[str] = frozenset({"scix"})

# Known index names (mirrors sibling scripts).
HNSW_BASELINE_INDEX = "idx_hnsw_baseline_indus"
DISKANN_INDEX_PREFIX = "paper_embeddings_diskann"

# Mapping from configuration label -> index name. A label is how results are
# keyed in the output JSON.
DEFAULT_INDEX_LABELS: dict[str, str] = {
    "hnsw_baseline": HNSW_BASELINE_INDEX,
    "diskann_v1": f"{DISKANN_INDEX_PREFIX}_v1",
    "diskann_v2": f"{DISKANN_INDEX_PREFIX}_v2",
    "diskann_v3": f"{DISKANN_INDEX_PREFIX}_v3",
}


# ---------------------------------------------------------------------------
# DSN safety
# ---------------------------------------------------------------------------


def assert_pilot_dsn(dsn: str) -> None:
    """Raise ValueError if ``dsn`` points at a production database.

    Error message contains both 'production' and 'refuse' so callers that
    grep for either keyword see the refusal. Mirrors the guard in
    scripts/build_hnsw_baseline.py.
    """
    if not dsn or not dsn.strip():
        raise ValueError(
            "Empty DSN — refuse to run without an explicit pilot DSN."
        )
    try:
        params = conninfo_to_dict(dsn)
    except psycopg.ProgrammingError as exc:
        raise ValueError(f"Invalid DSN: {exc}") from exc
    dbname = params.get("dbname")
    if isinstance(dbname, str) and dbname.lower() in _PRODUCTION_DB_NAMES:
        raise ValueError(
            f"Refuse to run against production DSN (dbname={dbname!r}). "
            f"This script is for pilot/benchmark databases only. "
            f"Set --dsn or SCIX_PILOT_DSN to a non-production database."
        )


# ---------------------------------------------------------------------------
# Pure helpers (no DB access — easily unit-testable)
# ---------------------------------------------------------------------------


def parse_thread_counts(raw: str) -> list[int]:
    """Parse a comma-separated list of positive ints.

    >>> parse_thread_counts("10,50")
    [10, 50]
    """
    if not raw or not raw.strip():
        raise ValueError("--thread-counts must not be empty")
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid thread count {token!r}: not an integer"
            ) from exc
        if value <= 0:
            raise ValueError(
                f"Thread counts must be positive integers, got {value}"
            )
        out.append(value)
    if not out:
        raise ValueError("--thread-counts parsed to an empty list")
    return out


def compute_percentiles(durations_ms: list[float]) -> dict[str, float]:
    """Return {p50, p95, p99} for a list of latency samples in milliseconds.

    Uses ``numpy.percentile`` with linear interpolation (numpy default),
    which is the same backend used by the tests. A pure function: no DB or
    filesystem access, deterministic for fixed input.
    """
    if not durations_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(durations_ms, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return {"p50": p50, "p95": p95, "p99": p99}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexSpec:
    """Describes one index under test."""

    label: str
    index_name: str


@dataclass
class WorkerStats:
    """Per-worker latency accumulator."""

    durations_ms: list[float] = field(default_factory=list)
    errors: int = 0

    def record(self, duration_ms: float) -> None:
        self.durations_ms.append(duration_ms)

    def record_error(self) -> None:
        self.errors += 1


@dataclass
class ResultEntry:
    """Metrics for one (index_label, thread_count) cell."""

    index_name: str
    thread_count: int
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    queries_executed: int
    errors: int
    duration_seconds: float


# ---------------------------------------------------------------------------
# Query set loading
# ---------------------------------------------------------------------------


def load_query_embeddings(
    dsn: str,
    *,
    limit: int = DEFAULT_QUERY_SET_SIZE,
) -> list[tuple[str, str]]:
    """Load up to ``limit`` (bibcode, embedding_text) pairs from the pilot DB.

    The embedding is returned as its psycopg text representation so it can be
    bound as a halfvec parameter with ``%s::halfvec`` at query time. This
    avoids shipping a separate 50q fixture — any pilot DB with indus rows
    provides the stress workload.

    Returns an empty list if no embeddings are present.
    """
    sql = (
        "SELECT bibcode, embedding::text "
        "FROM paper_embeddings "
        "WHERE model_name = %s AND embedding IS NOT NULL "
        "ORDER BY bibcode "
        "LIMIT %s"
    )
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (MODEL_NAME, limit))
            rows = cur.fetchall()
    return [(r[0], r[1]) for r in rows]


# ---------------------------------------------------------------------------
# Worker + orchestrator
# ---------------------------------------------------------------------------


_WORKER_QUERY_SQL = (
    "SELECT bibcode "
    "FROM paper_embeddings "
    "WHERE model_name = %s "
    "ORDER BY embedding <=> %s::halfvec "
    "LIMIT %s"
)


def _worker_loop(
    pool: Any,
    index: IndexSpec,
    query_set: list[tuple[str, str]],
    deadline: float,
    stats: WorkerStats,
    rng: random.Random,
    limit: int = DEFAULT_QUERY_LIMIT,
) -> None:
    """Run queries until ``deadline`` wall-clock time, recording latencies.

    Uses a connection from the shared psycopg_pool. On exception, increments
    the error counter and continues until the deadline.
    """
    while time.monotonic() < deadline:
        _bib, qvec = rng.choice(query_set)
        t0 = time.perf_counter()
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(_WORKER_QUERY_SQL, (MODEL_NAME, qvec, limit))
                    cur.fetchall()
        except Exception:  # noqa: BLE001 - stress test: count and continue
            stats.record_error()
            continue
        duration_ms = (time.perf_counter() - t0) * 1000.0
        stats.record(duration_ms)


def run_stress_cell(
    dsn: str,
    index: IndexSpec,
    thread_count: int,
    duration_seconds: int,
    query_set: list[tuple[str, str]],
    seed: int = 42,
) -> ResultEntry:
    """Run one (index, thread_count) benchmark cell and return metrics.

    Opens a dedicated psycopg_pool sized to ``thread_count`` workers, hints
    planner options to prefer ``index.index_name`` when possible, spins up a
    ThreadPoolExecutor, and collects latency samples until the duration
    expires. On return, the pool is closed cleanly.
    """
    # Import inside the function so tests that only touch pure helpers do not
    # need psycopg_pool installed.
    from psycopg_pool import ConnectionPool

    logger.info(
        "Stress cell: index=%s threads=%d duration=%ds",
        index.label, thread_count, duration_seconds,
    )

    pool = ConnectionPool(
        dsn,
        min_size=max(1, thread_count // 2),
        max_size=thread_count,
        open=True,
        timeout=30,
    )
    try:
        # Per-thread stats containers so we never touch shared mutable state
        # from two threads at once.
        stats_by_worker: list[WorkerStats] = [
            WorkerStats() for _ in range(thread_count)
        ]
        deadline = time.monotonic() + duration_seconds
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            for worker_id in range(thread_count):
                worker_rng = random.Random(seed + worker_id)
                futures.append(
                    executor.submit(
                        _worker_loop,
                        pool,
                        index,
                        query_set,
                        deadline,
                        stats_by_worker[worker_id],
                        worker_rng,
                    )
                )
            # Drain futures — surfaces any thread-level exceptions.
            for fut in as_completed(futures):
                fut.result()
        actual_duration = time.monotonic() - t0
    finally:
        pool.close()

    all_durations: list[float] = []
    total_errors = 0
    for stats in stats_by_worker:
        all_durations.extend(stats.durations_ms)
        total_errors += stats.errors

    percentiles = compute_percentiles(all_durations)
    queries_executed = len(all_durations)
    qps = queries_executed / actual_duration if actual_duration > 0 else 0.0
    return ResultEntry(
        index_name=index.index_name,
        thread_count=thread_count,
        qps=round(qps, 3),
        p50_ms=round(percentiles["p50"], 3),
        p95_ms=round(percentiles["p95"], 3),
        p99_ms=round(percentiles["p99"], 3),
        queries_executed=queries_executed,
        errors=total_errors,
        duration_seconds=round(actual_duration, 3),
    )


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------


def entry_to_json(entry: ResultEntry, index_label: str) -> dict[str, Any]:
    """Serialize a ResultEntry to a JSON-safe dict."""
    return {
        "index_label": index_label,
        "index_name": entry.index_name,
        "thread_count": entry.thread_count,
        "qps": entry.qps,
        "p50_ms": entry.p50_ms,
        "p95_ms": entry.p95_ms,
        "p99_ms": entry.p99_ms,
        "queries_executed": entry.queries_executed,
        "errors": entry.errors,
        "duration_seconds": entry.duration_seconds,
    }


def build_result_document(
    *,
    entries: list[dict[str, Any]],
    dsn: str,
    thread_counts: list[int],
    duration_seconds: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Assemble the top-level JSON document."""
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dsn_dbname": conninfo_to_dict(dsn).get("dbname"),
        "thread_counts": list(thread_counts),
        "duration_seconds": duration_seconds,
        "model_name": MODEL_NAME,
        "dry_run": dry_run,
        "entries": entries,
    }


def render_markdown(doc: dict[str, Any]) -> str:
    """Render a compact markdown summary from the result document."""
    lines = [
        "# Concurrent Stress Benchmark",
        "",
        f"- Run ID: `{doc['run_id']}`",
        f"- Timestamp: {doc['timestamp']}",
        f"- DSN dbname: `{doc.get('dsn_dbname')}`",
        f"- Thread counts: {doc['thread_counts']}",
        f"- Duration (s): {doc['duration_seconds']}",
        f"- Dry run: {doc['dry_run']}",
        "",
        "| Index | Threads | QPS | p50 (ms) | p95 (ms) | p99 (ms) | Queries | Errors |",
        "|-------|--------:|----:|---------:|---------:|---------:|--------:|-------:|",
    ]
    for e in doc["entries"]:
        lines.append(
            f"| {e.get('index_label', e['index_name'])} "
            f"| {e['thread_count']} "
            f"| {e['qps']:.3f} "
            f"| {e['p50_ms']:.3f} "
            f"| {e['p95_ms']:.3f} "
            f"| {e['p99_ms']:.3f} "
            f"| {e['queries_executed']} "
            f"| {e['errors']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    doc: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    """Write the JSON document and markdown summary to disk."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(doc))
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def dry_run_entries(
    thread_counts: list[int],
    index_labels: dict[str, str],
) -> list[dict[str, Any]]:
    """Produce schema-complete, zero-metric entries for --dry-run."""
    out: list[dict[str, Any]] = []
    for label, index_name in index_labels.items():
        for tc in thread_counts:
            entry = ResultEntry(
                index_name=index_name,
                thread_count=tc,
                qps=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                queries_executed=0,
                errors=0,
                duration_seconds=0.0,
            )
            out.append(entry_to_json(entry, label))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sibling_md_path(json_out: Path) -> Path:
    """Return the sibling .md output path for a given JSON --out value."""
    return json_out.with_suffix(".md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concurrent-query stress benchmark for HNSW + StreamingDiskANN "
            "indexes on paper_embeddings. Refuses production DSN."
        ),
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help=(
            "PostgreSQL DSN for the pilot/benchmark database "
            "(refuses production 'dbname=scix'). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--thread-counts",
        default=DEFAULT_THREAD_COUNTS,
        help=(
            "Comma-separated list of thread counts to sweep "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help=(
            "Wall-clock duration per (index, thread_count) cell, in seconds "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            "Output JSON path. A sibling .md file is written alongside. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--query-set-size",
        type=int,
        default=DEFAULT_QUERY_SET_SIZE,
        help=(
            "Number of query embeddings to sample from paper_embeddings "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for per-worker random draws (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip DB access. Write a schema-complete JSON/MD shell with zero "
            "metrics for every (index, thread_count) cell and exit."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        assert_pilot_dsn(args.dsn)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        thread_counts = parse_thread_counts(args.thread_counts)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    md_path = _sibling_md_path(args.out)

    if args.dry_run:
        entries = dry_run_entries(thread_counts, DEFAULT_INDEX_LABELS)
        doc = build_result_document(
            entries=entries,
            dsn=args.dsn,
            thread_counts=thread_counts,
            duration_seconds=args.duration_seconds,
            dry_run=True,
        )
        write_outputs(doc, args.out, md_path)
        return 0

    # Load query embeddings once (shared across all cells).
    logger.info(
        "Loading up to %d query embeddings from %s",
        args.query_set_size, args.dsn,
    )
    query_set = load_query_embeddings(args.dsn, limit=args.query_set_size)
    if not query_set:
        logger.error(
            "No indus embeddings found on %s — cannot run benchmark.",
            args.dsn,
        )
        return 3

    entries: list[dict[str, Any]] = []
    for label, index_name in DEFAULT_INDEX_LABELS.items():
        index = IndexSpec(label=label, index_name=index_name)
        for tc in thread_counts:
            result = run_stress_cell(
                args.dsn,
                index,
                tc,
                args.duration_seconds,
                query_set,
                seed=args.seed,
            )
            entries.append(entry_to_json(result, label))

    doc = build_result_document(
        entries=entries,
        dsn=args.dsn,
        thread_counts=thread_counts,
        duration_seconds=args.duration_seconds,
        dry_run=False,
    )
    write_outputs(doc, args.out, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
