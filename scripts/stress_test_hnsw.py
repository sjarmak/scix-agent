#!/usr/bin/env python3
"""Stress-test parallel HNSW rebuild under concurrent ingestion.

Creates an isolated test table with sample embeddings, then runs three scenarios:
  1. Serial HNSW build (baseline)
  2. Parallel HNSW build (no concurrent writes)
  3. Parallel HNSW build + concurrent COPY-based ingestion

Measures wall-clock time, deadlocks, row integrity, and index usability.

Usage:
    python scripts/stress_test_hnsw.py [--base-rows 150000] [--ingest-rows 50000]
                                       [--batch-size 5000] [--workers 7]
                                       [--no-cleanup]

Requires: psql access to scix database (uses SCIX_DSN or default dbname=scix).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("stress_hnsw")

TABLE = "_hnsw_stress_test"
TABLE_INGEST = "_hnsw_stress_ingest"
INDEX_NAME = "idx_hnsw_stress"
DSN = os.environ.get("SCIX_DSN", "dbname=scix")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioResult:
    """Immutable result of a single test scenario."""

    name: str
    build_seconds: float
    rows_before: int
    rows_after: int
    ingest_seconds: float | None
    ingest_rows: int
    ingest_errors: list[str]
    deadlocks: int
    index_valid: bool
    ann_query_ok: bool
    ann_latency_ms: float


@dataclass
class IngestStats:
    """Mutable container for ingestion thread results."""

    rows_inserted: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    deadlocks: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count(conn: psycopg.Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
        return cur.fetchone()[0]


def _drop_index(conn: psycopg.Connection) -> None:
    conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")


def _check_index_valid(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.indisvalid FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = %s",
            (INDEX_NAME,),
        )
        row = cur.fetchone()
        return row is not None and row[0]


def _ann_query(conn: psycopg.Connection) -> tuple[bool, float]:
    """Run ANN query, return (success, latency_ms)."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT embedding FROM {TABLE} LIMIT 1")
            row = cur.fetchone()
            if row is None:
                return False, 0.0
            qvec = row[0]

            t0 = time.perf_counter()
            cur.execute(
                f"SELECT bibcode, embedding <=> %s::vector AS dist "
                f"FROM {TABLE} ORDER BY embedding <=> %s::vector LIMIT 10",
                (qvec, qvec),
            )
            results = cur.fetchall()
            latency = (time.perf_counter() - t0) * 1000
            return len(results) == 10, round(latency, 1)
    except Exception as e:
        logger.error("ANN query failed: %s", e)
        return False, 0.0


# ---------------------------------------------------------------------------
# Setup / cleanup
# ---------------------------------------------------------------------------

def setup_test_tables(conn: psycopg.Connection, base_rows: int, ingest_rows: int) -> None:
    """Create test table with base_rows and staging table with ingest_rows."""
    logger.info("Setting up: %d base + %d ingest rows from paper_embeddings (indus)", base_rows, ingest_rows)

    conn.execute(f"DROP TABLE IF EXISTS {TABLE_INGEST} CASCADE")
    conn.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")

    # Main test table — base rows only
    conn.execute(
        f"CREATE TABLE {TABLE} (bibcode text NOT NULL, embedding vector(768) NOT NULL)"
    )
    conn.execute(
        f"INSERT INTO {TABLE} (bibcode, embedding) "
        f"SELECT bibcode, embedding FROM paper_embeddings "
        f"WHERE model_name = 'indus' AND embedding IS NOT NULL "
        f"LIMIT %s",
        (base_rows,),
    )

    # Ingest staging — rows to be COPYed during concurrent test
    conn.execute(
        f"CREATE TABLE {TABLE_INGEST} (bibcode text NOT NULL, embedding vector(768) NOT NULL)"
    )
    conn.execute(
        f"INSERT INTO {TABLE_INGEST} (bibcode, embedding) "
        f"SELECT bibcode, embedding FROM paper_embeddings "
        f"WHERE model_name = 'indus' AND embedding IS NOT NULL "
        f"OFFSET %s LIMIT %s",
        (base_rows, ingest_rows),
    )
    conn.commit()

    actual_base = _count(conn, TABLE)
    actual_ingest = _count(conn, TABLE_INGEST)
    logger.info("Created %s: %d rows", TABLE, actual_base)
    logger.info("Created %s: %d rows", TABLE_INGEST, actual_ingest)


def cleanup(conn: psycopg.Connection) -> None:
    _drop_index(conn)
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_INGEST} CASCADE")
    conn.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
    conn.commit()
    logger.info("Cleaned up test tables")


# ---------------------------------------------------------------------------
# HNSW build
# ---------------------------------------------------------------------------

def _build_hnsw(workers: int | None, use_concurrently: bool = False) -> float:
    """Build HNSW index. Returns wall-clock seconds."""
    conn = get_connection(DSN, autocommit=True)
    try:
        _drop_index(conn)

        if workers is not None:
            conn.execute(f"SET max_parallel_maintenance_workers = {int(workers)}")
        else:
            conn.execute("SET max_parallel_maintenance_workers = 0")

        concurrently = "CONCURRENTLY " if use_concurrently else ""
        sql = (
            f"CREATE INDEX {concurrently}{INDEX_NAME} ON {TABLE} "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )

        logger.info("Building HNSW (workers=%s, concurrently=%s)...", workers, use_concurrently)
        t0 = time.monotonic()
        conn.execute(sql)
        elapsed = time.monotonic() - t0
        logger.info("HNSW build: %.1fs", elapsed)
        return elapsed
    finally:
        conn.close()


def _build_hnsw_thread(workers: int, use_concurrently: bool, result: dict[str, Any]) -> None:
    """Thread target for HNSW build."""
    try:
        result["build_seconds"] = _build_hnsw(workers, use_concurrently)
        result["error"] = None
    except Exception as e:
        result["build_seconds"] = 0.0
        result["error"] = str(e)
        logger.error("HNSW build error: %s", e)


# ---------------------------------------------------------------------------
# Concurrent COPY-based ingestion (matches embed_fast.py pattern)
# ---------------------------------------------------------------------------

def _concurrent_ingest(batch_size: int, stats: IngestStats) -> None:
    """INSERT rows from staging into test table in batches (simulates COPY pipeline)."""
    conn = get_connection(DSN)
    try:
        total = _count(conn, TABLE_INGEST)
        offset = 0
        t0 = time.monotonic()

        while offset < total:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {TABLE} (bibcode, embedding) "
                        f"SELECT bibcode, embedding FROM {TABLE_INGEST} "
                        f"OFFSET %s LIMIT %s",
                        (offset, batch_size),
                    )
                    inserted = cur.rowcount
                conn.commit()
                stats.rows_inserted += inserted
                offset += batch_size
                logger.info("Ingested batch: +%d (total: %d/%d)", inserted, stats.rows_inserted, total)
            except Exception as e:
                conn.rollback()
                err = str(e)
                stats.errors.append(err)
                if "deadlock" in err.lower():
                    stats.deadlocks += 1
                    logger.warning("Deadlock at offset %d, retrying...", offset)
                    time.sleep(0.5)
                else:
                    logger.error("Ingest error at offset %d: %s", offset, e)
                    offset += batch_size

        stats.seconds = time.monotonic() - t0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_serial(conn: psycopg.Connection) -> ScenarioResult:
    """Scenario 1: Serial HNSW build, no concurrent writes."""
    logger.info("=" * 60)
    logger.info("SCENARIO 1: Serial HNSW build (baseline)")
    logger.info("=" * 60)

    rows_before = _count(conn, TABLE)
    build_time = _build_hnsw(workers=0)
    rows_after = _count(conn, TABLE)
    valid = _check_index_valid(conn)
    ann_ok, ann_lat = _ann_query(conn)

    return ScenarioResult(
        name="Serial build (baseline)",
        build_seconds=build_time,
        rows_before=rows_before,
        rows_after=rows_after,
        ingest_seconds=None,
        ingest_rows=0,
        ingest_errors=[],
        deadlocks=0,
        index_valid=valid,
        ann_query_ok=ann_ok,
        ann_latency_ms=ann_lat,
    )


def scenario_parallel(conn: psycopg.Connection, workers: int) -> ScenarioResult:
    """Scenario 2: Parallel HNSW build, no concurrent writes."""
    logger.info("=" * 60)
    logger.info("SCENARIO 2: Parallel HNSW build (%d workers)", workers)
    logger.info("=" * 60)

    rows_before = _count(conn, TABLE)
    build_time = _build_hnsw(workers=workers)
    rows_after = _count(conn, TABLE)
    valid = _check_index_valid(conn)
    ann_ok, ann_lat = _ann_query(conn)

    return ScenarioResult(
        name=f"Parallel build ({workers} workers)",
        build_seconds=build_time,
        rows_before=rows_before,
        rows_after=rows_after,
        ingest_seconds=None,
        ingest_rows=0,
        ingest_errors=[],
        deadlocks=0,
        index_valid=valid,
        ann_query_ok=ann_ok,
        ann_latency_ms=ann_lat,
    )


def scenario_parallel_concurrent(
    conn: psycopg.Connection, workers: int, batch_size: int
) -> ScenarioResult:
    """Scenario 3: Parallel HNSW build + concurrent ingestion."""
    logger.info("=" * 60)
    logger.info("SCENARIO 3: Parallel HNSW build (%d workers) + concurrent ingest", workers)
    logger.info("=" * 60)

    # Remove any previously ingested rows
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {TABLE} t USING {TABLE_INGEST} s WHERE t.bibcode = s.bibcode"
        )
    conn.commit()
    _drop_index(conn)

    rows_before = _count(conn, TABLE)

    build_result: dict[str, Any] = {}
    ingest_stats = IngestStats()

    # Use CREATE INDEX CONCURRENTLY so it doesn't block writes
    build_thread = threading.Thread(
        target=_build_hnsw_thread,
        args=(workers, True, build_result),
        name="hnsw-builder",
    )
    ingest_thread = threading.Thread(
        target=_concurrent_ingest,
        args=(batch_size, ingest_stats),
        name="ingest-writer",
    )

    # Start ingestion first, let it get a head start
    ingest_thread.start()
    time.sleep(0.5)
    build_thread.start()

    build_thread.join(timeout=3600)
    ingest_thread.join(timeout=3600)

    rows_after = _count(conn, TABLE)
    valid = _check_index_valid(conn)
    ann_ok, ann_lat = _ann_query(conn)

    errors = list(ingest_stats.errors)
    build_err = build_result.get("error")
    if build_err:
        errors.insert(0, f"BUILD: {build_err}")

    return ScenarioResult(
        name=f"Parallel build ({workers} workers) + concurrent ingest",
        build_seconds=build_result.get("build_seconds", 0.0),
        rows_before=rows_before,
        rows_after=rows_after,
        ingest_seconds=ingest_stats.seconds,
        ingest_rows=ingest_stats.rows_inserted,
        ingest_errors=errors,
        deadlocks=ingest_stats.deadlocks,
        index_valid=valid,
        ann_query_ok=ann_ok,
        ann_latency_ms=ann_lat,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[ScenarioResult]) -> dict[str, Any]:
    """Print summary and return JSON-serializable dict."""
    print("\n" + "=" * 72)
    print("STRESS TEST: Parallel HNSW Rebuild Under Concurrent Ingestion")
    print("=" * 72)

    report: dict[str, Any] = {"scenarios": []}

    for r in results:
        print(f"\n--- {r.name} ---")
        print(f"  Build time:     {r.build_seconds:>8.1f}s")
        print(f"  Rows before:    {r.rows_before:>8,}")
        print(f"  Rows after:     {r.rows_after:>8,}")
        if r.ingest_seconds is not None:
            rate = r.ingest_rows / r.ingest_seconds if r.ingest_seconds > 0 else 0
            print(f"  Ingest time:    {r.ingest_seconds:>8.1f}s")
            print(f"  Ingest rows:    {r.ingest_rows:>8,}")
            print(f"  Ingest rate:    {rate:>8,.0f} rows/s")
        print(f"  Deadlocks:      {r.deadlocks:>8}")
        print(f"  Index valid:    {'YES' if r.index_valid else 'NO':>8}")
        print(f"  ANN query OK:   {'YES' if r.ann_query_ok else 'NO':>8}")
        print(f"  ANN latency:    {r.ann_latency_ms:>7.1f}ms")
        if r.ingest_errors:
            print(f"  Errors ({len(r.ingest_errors)}):")
            for e in r.ingest_errors[:5]:
                print(f"    - {e[:120]}")

        report["scenarios"].append({
            "name": r.name,
            "build_seconds": r.build_seconds,
            "rows_before": r.rows_before,
            "rows_after": r.rows_after,
            "ingest_seconds": r.ingest_seconds,
            "ingest_rows": r.ingest_rows,
            "deadlocks": r.deadlocks,
            "index_valid": r.index_valid,
            "ann_query_ok": r.ann_query_ok,
            "ann_latency_ms": r.ann_latency_ms,
            "errors": r.ingest_errors,
        })

    # Speedup comparison
    if len(results) >= 2 and results[0].build_seconds > 0:
        speedup = results[0].build_seconds / results[1].build_seconds
        print(f"\n  Parallel speedup vs serial: {speedup:.2f}x")
        report["speedup"] = round(speedup, 2)

    # Verdict
    print("\n--- VERDICT ---")
    s3 = results[-1] if len(results) >= 3 else None
    issues: list[str] = []
    if s3 is not None:
        if not s3.index_valid:
            issues.append("Index INVALID after concurrent build+ingest")
        if not s3.ann_query_ok:
            issues.append("ANN queries FAIL after concurrent build+ingest")
        if s3.deadlocks > 0:
            issues.append(f"{s3.deadlocks} deadlocks during concurrent ingest")
        non_deadlock = [e for e in s3.ingest_errors if "deadlock" not in e.lower()]
        if non_deadlock:
            issues.append(f"{len(non_deadlock)} non-deadlock errors")

    if not issues:
        verdict = "SAFE"
        print("  SAFE: Parallel HNSW build completes correctly under concurrent ingestion.")
        print("  No deadlocks, no corruption, index is valid and queryable.")
    else:
        verdict = "CAUTION"
        print("  CAUTION: Issues detected:")
        for issue in issues:
            print(f"    - {issue}")

    report["verdict"] = verdict
    report["issues"] = issues
    print("=" * 72)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test parallel HNSW rebuild")
    parser.add_argument("--base-rows", type=int, default=150_000)
    parser.add_argument("--ingest-rows", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    conn = get_connection(DSN, autocommit=True)

    try:
        setup_test_tables(conn, args.base_rows, args.ingest_rows)

        results: list[ScenarioResult] = []
        results.append(scenario_serial(conn))
        results.append(scenario_parallel(conn, args.workers))
        results.append(scenario_parallel_concurrent(conn, args.workers, args.batch_size))

        report = print_report(results)

        # Save JSON results
        out_path = Path(__file__).resolve().parent.parent / "results" / "hnsw_stress_test.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    finally:
        if not args.no_cleanup:
            cleanup(conn)
        conn.close()


if __name__ == "__main__":
    main()
