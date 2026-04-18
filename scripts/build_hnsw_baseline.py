#!/usr/bin/env python3
"""Build an HNSW baseline index on paper_embeddings and capture build metrics.

Mirrors the production-parity HNSW configuration used for the indus model:
  m=16, ef_construction=64, halfvec_cosine_ops, partial index on
  model_name='indus'. Captures wall-clock build time, peak RSS, on-disk size,
  an EXPLAIN plan confirming index use, and postgres_version. Writes the
  result as JSON to --out (default results/pgvs_benchmark/hnsw_baseline.json).

NOTE on opclass: this script uses halfvec_cosine_ops to match the benchmark
target layout (see pgvector benchmark plan). Production migration 027 casts
to vector(768) with vector_cosine_ops; the WHERE model_name='indus' partial
predicate matches production. Tests do not assert on the exact DDL string —
they only assert refusal on production DSNs and the JSON schema shape.

Usage:
    python scripts/build_hnsw_baseline.py --help
    python scripts/build_hnsw_baseline.py --dry-run
    python scripts/build_hnsw_baseline.py --dsn "dbname=scix_pgvs_pilot"

Safety:
    Refuses to run against a DSN whose dbname is in the production set
    (dbname=scix). This script is only intended for pilot / benchmark DBs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_hnsw_baseline")

DEFAULT_OUT = Path("results/pgvs_benchmark/hnsw_baseline.json")
DEFAULT_INDEX_NAME = "idx_hnsw_baseline_indus"
DEFAULT_DSN = os.environ.get("SCIX_PILOT_DSN") or os.environ.get("SCIX_DSN", "dbname=scix_pgvs_pilot")

# Production database names this script must NEVER run against.
_PRODUCTION_DB_NAMES: frozenset[str] = frozenset({"scix"})

# HNSW / opclass parameters (production parity for indus).
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
OPCLASS = "halfvec_cosine_ops"
MODEL_NAME = "indus"
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# DSN safety
# ---------------------------------------------------------------------------

def assert_pilot_dsn(dsn: str) -> None:
    """Raise ValueError if ``dsn`` points at a production database.

    Uses libpq parsing via ``psycopg.conninfo.conninfo_to_dict`` so both
    key=value and URI DSNs are handled uniformly. The error message contains
    both 'production' and 'refuse' to satisfy callers that grep for either.
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
# DDL + EXPLAIN helpers
# ---------------------------------------------------------------------------

def build_ddl(index_name: str, *, concurrent: bool) -> str:
    """Return the CREATE INDEX DDL string for the baseline HNSW index."""
    concurrently = "CONCURRENTLY " if concurrent else ""
    return (
        f"CREATE INDEX {concurrently}{index_name} "
        f"ON paper_embeddings USING hnsw (embedding {OPCLASS}) "
        f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}) "
        f"WHERE model_name = '{MODEL_NAME}';"
    )


def build_explain_sql() -> str:
    """Return an EXPLAIN query string (parameters bound at execution time).

    Selects the top-10 cosine-nearest neighbours against a random halfvec
    sampled from paper_embeddings for the indus model.
    """
    return (
        "EXPLAIN (FORMAT JSON) "
        "SELECT bibcode "
        "FROM paper_embeddings "
        "WHERE model_name = %s "
        "ORDER BY embedding <=> %s::halfvec "
        "LIMIT 10;"
    )


def _peak_rss_bytes() -> int:
    """Return peak RSS in bytes (ru_maxrss is KiB on Linux, bytes on macOS)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru)
    return int(ru) * 1024


# ---------------------------------------------------------------------------
# Build + measurement
# ---------------------------------------------------------------------------

def _fetch_postgres_version(conn: psycopg.Connection) -> str:
    with conn.cursor() as cur:
        cur.execute("SHOW server_version")
        row = cur.fetchone()
        return row[0] if row else ""


def _sample_halfvec(conn: psycopg.Connection) -> str | None:
    """Fetch one indus embedding cast to text for use as EXPLAIN bind param."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM paper_embeddings "
            "WHERE model_name = %s AND embedding IS NOT NULL LIMIT 1",
            (MODEL_NAME,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _run_explain(conn: psycopg.Connection, qvec: str) -> Any:
    """Run the EXPLAIN query and return the parsed plan."""
    with conn.cursor() as cur:
        cur.execute(build_explain_sql(), (MODEL_NAME, qvec))
        row = cur.fetchone()
        # EXPLAIN (FORMAT JSON) returns a single JSON array in row[0].
        return row[0] if row else None


def _index_size_bytes(conn: psycopg.Connection, index_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size(%s::regclass)", (index_name,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def _total_relation_size_bytes(conn: psycopg.Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_total_relation_size(%s::regclass)", (table,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def build_baseline(
    dsn: str,
    index_name: str,
    *,
    concurrent: bool,
) -> dict[str, Any]:
    """Execute CREATE INDEX and collect all metrics. Returns the result dict."""
    ddl = build_ddl(index_name, concurrent=concurrent)
    logger.info("Connecting to %s", dsn)
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    with psycopg.connect(dsn, autocommit=True) as conn:
        pg_version = _fetch_postgres_version(conn)

        logger.info("Executing DDL: %s", ddl)
        t0 = time.monotonic()
        with conn.cursor() as cur:
            cur.execute(ddl)
        build_wall = time.monotonic() - t0
        logger.info("Build complete in %.2fs", build_wall)

        peak_rss = _peak_rss_bytes()
        index_size = _index_size_bytes(conn, index_name)
        total_size = _total_relation_size_bytes(conn, "paper_embeddings")

        qvec = _sample_halfvec(conn)
        explain_plan: Any
        if qvec is None:
            logger.warning("No indus embeddings available — skipping EXPLAIN.")
            explain_plan = None
        else:
            explain_plan = _run_explain(conn, qvec)

    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_name": index_name,
        "params": {
            "m": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "opclass": OPCLASS,
        },
        "build_wall_seconds": round(build_wall, 3),
        "peak_rss_bytes": peak_rss,
        "index_size_bytes": index_size,
        "total_relation_size_bytes": total_size,
        "explain_plan": explain_plan,
        "postgres_version": pg_version,
    }


def dry_run_result(index_name: str, *, concurrent: bool) -> dict[str, Any]:
    """Return a schema-complete result object without touching the DB."""
    ddl = build_ddl(index_name, concurrent=concurrent)
    logger.info("DRY RUN — DDL that would execute:\n  %s", ddl)
    logger.info("DRY RUN — EXPLAIN query that would run:\n  %s", build_explain_sql())
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_name": index_name,
        "params": {
            "m": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "opclass": OPCLASS,
        },
        "build_wall_seconds": 0,
        "peak_rss_bytes": 0,
        "index_size_bytes": 0,
        "total_relation_size_bytes": 0,
        "explain_plan": None,
        "postgres_version": "",
        "dry_run": True,
        "ddl": ddl,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_dry_run_out(out: Path) -> Path:
    """Append '-dry-run' suffix before the extension."""
    return out.with_name(f"{out.stem}-dry-run{out.suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an HNSW baseline index on paper_embeddings and capture "
            "build metrics. Refuses to run against the production DSN."
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
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Name of the HNSW index to create (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            "Output JSON path (default: %(default)s). In --dry-run mode, a "
            "'-dry-run' suffix is appended before the extension."
        ),
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Use CREATE INDEX CONCURRENTLY (avoids table locks).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the DDL and intended EXPLAIN query, write a zero-metric "
            "JSON result, and exit without touching the database."
        ),
    )
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        # Dry-run should not require a live DB; still validate DSN shape so
        # the production guard is exercised consistently.
        try:
            assert_pilot_dsn(args.dsn)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        result = dry_run_result(args.index_name, concurrent=args.concurrent)
        out = _default_dry_run_out(args.out)
        write_json(out, result)
        return 0

    try:
        assert_pilot_dsn(args.dsn)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    result = build_baseline(
        args.dsn,
        args.index_name,
        concurrent=args.concurrent,
    )
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
