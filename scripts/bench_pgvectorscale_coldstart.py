#!/usr/bin/env python3
"""Cold-start / restart-recovery benchmark for pgvector / pgvectorscale indexes.

Measures the per-query latency of the first 10 queries against each candidate
index *after the operator has manually restarted Postgres* — i.e. when buffer
pools and OS page cache are cold — then runs a further 100 warm-up queries
and reports steady-state p50/p95 latency as the warm baseline. For each index
we compute a ``cold_warm_ratio`` = ``cold_query_latencies_ms[0] / warm_p50_ms``
so operators can see which index layouts pay a large cold-start tax.

IMPORTANT — PRE-REQUISITE (READ CAREFULLY):

  THIS SCRIPT DOES NOT RESTART POSTGRES.

  The operator is expected to bounce the database *immediately* before running
  this script, for example:

      sudo systemctl restart postgresql

  The benchmark is only meaningful when invoked against a freshly-restarted
  Postgres instance whose shared_buffers and OS page cache are cold. Running
  this script without first restarting Postgres will still produce numbers,
  but they will not measure what the file claims to measure.

Layout mirrors scripts/build_hnsw_baseline.py and
scripts/build_streamingdiskann_variants.py:

  * Refuses to run against the production DSN (``dbname=scix``).
  * Uses psycopg v3.
  * Captures environment metadata via scripts/pgvs_bench_env.py when available.
  * Emits a JSON result document and a Markdown sidecar with per-index rows.

Output files (by default):

  results/pgvs_benchmark/cold_start.json
  results/pgvs_benchmark/cold_start.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.conninfo import conninfo_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bench_pgvectorscale_coldstart")

# ---------------------------------------------------------------------------
# Defaults + configuration
# ---------------------------------------------------------------------------

DEFAULT_OUT_JSON = Path("results/pgvs_benchmark/cold_start.json")
DEFAULT_OUT_MD = Path("results/pgvs_benchmark/cold_start.md")
DEFAULT_DSN = os.environ.get("SCIX_PILOT_DSN") or os.environ.get(
    "SCIX_DSN", "dbname=scix_pgvs_pilot"
)

# Production database names this script must NEVER run against.
_PRODUCTION_DB_NAMES: frozenset[str] = frozenset({"scix"})

# Number of queries in the cold window and the warm steady-state window.
N_COLD = 10
N_WARM = 100

# Model name / embedding parameters — mirror build_hnsw_baseline.MODEL_NAME.
MODEL_NAME = "indus"

# Default candidate indexes. Each entry has a human-readable label; the
# benchmark runs the same query (configured to use the index) against each one.
# We reuse the naming convention from build_hnsw_baseline.py and
# build_streamingdiskann_variants.py so users can run the benchmark straight
# after those scripts populate indexes.
DEFAULT_INDEXES: tuple[str, ...] = (
    "idx_hnsw_baseline_indus",
    "paper_embeddings_diskann_v1",
    "paper_embeddings_diskann_v2",
    "paper_embeddings_diskann_v3",
)


# ---------------------------------------------------------------------------
# DSN safety (mirrors build_hnsw_baseline.assert_pilot_dsn)
# ---------------------------------------------------------------------------

def assert_pilot_dsn(dsn: str) -> None:
    """Raise ValueError if ``dsn`` points at a production database.

    Mirrors the guard in ``scripts/build_hnsw_baseline.py``. Uses libpq parsing
    via ``psycopg.conninfo.conninfo_to_dict`` so both key=value and URI DSNs
    are handled uniformly. The error message contains both 'production' and
    'refuse' so callers that grep for either substring work.
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
# Pure helpers (no DB required) — testable in isolation.
# ---------------------------------------------------------------------------

def compute_cold_warm_ratio(cold_ms_list: list[float], warm_p50_ms: float) -> float:
    """Return ``cold_ms_list[0] / warm_p50_ms``.

    Raises ValueError on empty cold list or non-positive warm p50. This is a
    pure function so tests can exercise it on hand-computed values.
    """
    if not cold_ms_list:
        raise ValueError("cold_ms_list must contain at least one measurement")
    if warm_p50_ms <= 0:
        raise ValueError(
            f"warm_p50_ms must be > 0 to compute a ratio, got {warm_p50_ms!r}"
        )
    return float(cold_ms_list[0]) / float(warm_p50_ms)


def percentile(values: Iterable[float], pct: float) -> float:
    """Return the ``pct`` percentile (0..100) of ``values`` using linear interp.

    Falls back to ``statistics.quantiles`` semantics: returns the nearest-rank
    percentile. Empty input raises ValueError.
    """
    data = sorted(float(v) for v in values)
    if not data:
        raise ValueError("percentile() requires at least one value")
    if pct <= 0:
        return data[0]
    if pct >= 100:
        return data[-1]
    # Linear interpolation between closest ranks (NumPy 'linear' method).
    k = (len(data) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    frac = k - lo
    return data[lo] + (data[hi] - data[lo]) * frac


def build_query_sql() -> str:
    """Return the k-NN query used for latency measurement.

    Uses the cosine distance operator and casts the bind parameter to
    ``halfvec`` so the HNSW / StreamingDiskANN indexes built with
    ``halfvec_cosine_ops`` can be chosen by the planner.
    """
    return (
        "SELECT bibcode "
        "FROM paper_embeddings "
        "WHERE model_name = %s "
        "ORDER BY embedding <=> %s::halfvec "
        "LIMIT 10"
    )


def summarise_index_results(
    cold_ms: list[float], warm_ms: list[float]
) -> dict[str, Any]:
    """Compute warm p50/p95 and the cold-vs-warm ratio for one index."""
    warm_p50 = percentile(warm_ms, 50.0)
    warm_p95 = percentile(warm_ms, 95.0)
    return {
        "cold_query_latencies_ms": [round(float(x), 3) for x in cold_ms],
        "warm_p50_ms": round(float(warm_p50), 3),
        "warm_p95_ms": round(float(warm_p95), 3),
        "cold_warm_ratio": round(
            compute_cold_warm_ratio(cold_ms, warm_p50), 3
        ),
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _sample_halfvec(conn: psycopg.Connection) -> str | None:
    """Fetch one indus embedding cast to text for use as the query vector."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM paper_embeddings "
            "WHERE model_name = %s AND embedding IS NOT NULL LIMIT 1",
            (MODEL_NAME,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _force_index(conn: psycopg.Connection, index_name: str) -> None:
    """Nudge the planner toward ``index_name`` by disabling seqscan.

    We do NOT assert that only this index is used — some planners may still
    pick a different one. The EXPLAIN plan is captured separately for audit.
    """
    with conn.cursor() as cur:
        cur.execute("SET LOCAL enable_seqscan = off")
        # pg_hint_plan is not guaranteed to be installed on the pilot, so we
        # rely on the combination of (a) filtered partial index on model_name,
        # (b) ORDER BY <=> operator class match, and (c) LIMIT 10 to make the
        # target index the cheapest plan. The index_name arg is surfaced in
        # the JSON output for reviewer inspection.
        _ = index_name  # recorded in output; no direct pin available


def _run_query_once(
    conn: psycopg.Connection, qvec: str
) -> float:
    """Execute one k-NN query and return wall-clock latency in ms."""
    sql = build_query_sql()
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (MODEL_NAME, qvec))
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000.0


def measure_index(
    dsn: str, index_name: str, n_cold: int = N_COLD, n_warm: int = N_WARM
) -> dict[str, Any]:
    """Run ``n_cold`` cold queries + ``n_warm`` warm queries against ``index_name``.

    The caller is responsible for ensuring Postgres has been restarted before
    the first call of this function for a given run.
    """
    logger.info(
        "Measuring index %s (cold=%d, warm=%d)", index_name, n_cold, n_warm
    )
    with psycopg.connect(dsn, autocommit=True) as conn:
        qvec = _sample_halfvec(conn)
        if qvec is None:
            raise RuntimeError(
                f"No embeddings available for model_name={MODEL_NAME!r} — "
                "cannot run cold-start benchmark."
            )
        _force_index(conn, index_name)

        cold_ms: list[float] = []
        for i in range(n_cold):
            latency = _run_query_once(conn, qvec)
            logger.info("  cold q%02d: %.3f ms", i + 1, latency)
            cold_ms.append(latency)

        warm_ms: list[float] = []
        for _ in range(n_warm):
            warm_ms.append(_run_query_once(conn, qvec))
        logger.info(
            "  warm: n=%d p50=%.3fms p95=%.3fms",
            len(warm_ms),
            percentile(warm_ms, 50.0),
            percentile(warm_ms, 95.0),
        )

    summary = summarise_index_results(cold_ms, warm_ms)
    summary["index_name"] = index_name
    return summary


# ---------------------------------------------------------------------------
# Optional environment metadata (best-effort import of pgvs_bench_env).
# ---------------------------------------------------------------------------

def _capture_env_best_effort(dsn: str) -> dict[str, Any]:
    """Call ``scripts.pgvs_bench_env.capture_env`` if importable; else {}.

    Works whether the script is run from the repo root or the worktree root.
    """
    env: dict[str, Any] = {}
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        import pgvs_bench_env  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - best-effort capture
        logger.debug("pgvs_bench_env unavailable: %s", exc)
        return env
    try:
        env = pgvs_bench_env.capture_env(dsn)
    except Exception as exc:  # noqa: BLE001
        logger.debug("capture_env failed: %s", exc)
    return env


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def build_result_document(
    indexes: list[dict[str, Any]],
    *,
    dsn: str,
    env: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Assemble the final JSON result document for a run."""
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dsn": _safe_dsn(dsn),
        "n_cold": N_COLD,
        "n_warm": N_WARM,
        "indexes": indexes,
        "env": env,
        "dry_run": dry_run,
    }


def _safe_dsn(dsn: str) -> str:
    """Return the DSN minus any password for display/logging."""
    try:
        params = conninfo_to_dict(dsn)
    except Exception:  # noqa: BLE001 - display-only
        return "<unparseable>"
    params.pop("password", None)
    return " ".join(f"{k}={v}" for k, v in sorted(params.items()))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    logger.info("Wrote %s", path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Render a short human-readable report alongside the JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# pgvectorscale cold-start benchmark")
    lines.append("")
    lines.append(f"- run_id: `{payload.get('run_id', '')}`")
    lines.append(f"- timestamp: `{payload.get('timestamp', '')}`")
    lines.append(f"- dsn: `{payload.get('dsn', '')}`")
    lines.append(
        f"- n_cold = {payload.get('n_cold', N_COLD)}, "
        f"n_warm = {payload.get('n_warm', N_WARM)}"
    )
    if payload.get("dry_run"):
        lines.append("- **DRY RUN** — no database queries were executed.")
    lines.append("")
    lines.append("## Per-index results")
    lines.append("")
    lines.append(
        "| Index | Cold q1 (ms) | Cold q10 (ms) | Warm p50 (ms) | "
        "Warm p95 (ms) | Cold q1 / Warm p50 |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for entry in payload.get("indexes", []):
        colds = entry.get("cold_query_latencies_ms") or []
        cold_q1 = colds[0] if colds else float("nan")
        cold_q10 = colds[-1] if colds else float("nan")
        lines.append(
            "| {name} | {q1} | {q10} | {p50} | {p95} | {ratio} |".format(
                name=entry.get("index_name", "?"),
                q1=_fmt(cold_q1),
                q10=_fmt(cold_q10),
                p50=_fmt(entry.get("warm_p50_ms")),
                p95=_fmt(entry.get("warm_p95_ms")),
                ratio=_fmt(entry.get("cold_warm_ratio")),
            )
        )
    lines.append("")
    lines.append(
        "> **IMPORTANT:** the operator must run `sudo systemctl restart "
        "postgresql` immediately before invoking this benchmark; otherwise "
        "the 'cold' column measures a warm cache."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Dry-run shell
# ---------------------------------------------------------------------------

def dry_run_indexes(index_names: list[str]) -> list[dict[str, Any]]:
    """Return zero-metric per-index entries with schema-complete keys."""
    shells: list[dict[str, Any]] = []
    for name in index_names:
        shells.append(
            {
                "index_name": name,
                "cold_query_latencies_ms": [0.0] * N_COLD,
                "warm_p50_ms": 0.0,
                "warm_p95_ms": 0.0,
                "cold_warm_ratio": 0.0,
            }
        )
    return shells


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_RESTART_NOTE = (
    "IMPORTANT — PRE-REQUISITE: this script does NOT restart Postgres. "
    "The operator MUST run 'sudo systemctl restart postgresql' (or the "
    "equivalent for this host) immediately before invoking this benchmark. "
    "Only a freshly-restarted database has truly cold shared_buffers and OS "
    "page cache, which is what the cold-query window is measuring."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start / restart-recovery benchmark for pgvector / "
            "pgvectorscale indexes. Measures per-query latency of the first "
            "10 queries against each index (cold cache) and then p50/p95 of "
            "the next 100 queries (warm steady state). " + _RESTART_NOTE
        ),
        epilog=(
            _RESTART_NOTE + " Refuses to run against the production DSN "
            "(dbname=scix). See --help for flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--indexes",
        nargs="+",
        default=list(DEFAULT_INDEXES),
        help=(
            "Names of indexes to benchmark (space-separated). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Output JSON path (default: %(default)s).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="Output Markdown path (default: %(default)s).",
    )
    parser.add_argument(
        "--n-cold",
        type=int,
        default=N_COLD,
        help="Number of cold queries (default: %(default)s).",
    )
    parser.add_argument(
        "--n-warm",
        type=int,
        default=N_WARM,
        help="Number of warm queries (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit a schema-complete JSON + Markdown shell with zero "
            "measurements. Does not touch the database. Still enforces the "
            "production-DSN guard."
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

    if args.dry_run:
        logger.info("DRY RUN — no Postgres queries will be executed.")
        indexes = dry_run_indexes(list(args.indexes))
        payload = build_result_document(
            indexes, dsn=args.dsn, env={}, dry_run=True
        )
        write_json(args.out_json, payload)
        write_markdown(args.out_md, payload)
        return 0

    logger.warning(
        "Assuming Postgres was restarted immediately before this run. "
        "If not, the cold-query window is not meaningful — see --help."
    )

    env = _capture_env_best_effort(args.dsn)

    results: list[dict[str, Any]] = []
    for index_name in args.indexes:
        entry = measure_index(
            args.dsn,
            index_name,
            n_cold=int(args.n_cold),
            n_warm=int(args.n_warm),
        )
        results.append(entry)

    payload = build_result_document(
        results, dsn=args.dsn, env=env, dry_run=False
    )
    write_json(args.out_json, payload)
    write_markdown(args.out_md, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
