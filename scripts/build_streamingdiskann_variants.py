#!/usr/bin/env python3
"""Build three pgvectorscale StreamingDiskANN index variants on paper_embeddings.

Creates and benchmarks three index variants on the pilot DB's paper_embeddings
table (filtered to model_name='indus'):

  V1: default params + halfvec (no SBQ)
  V2: default params + SBQ (num_bits_per_dimension = 2)
  V3: tuned params (storage_layout=memory_optimized, num_neighbors=64) + SBQ

For each variant we record:
  - DDL executed
  - Build wall-clock seconds
  - Peak RSS bytes (process-level, resource.getrusage(RUSAGE_SELF).ru_maxrss)
  - Index size bytes (pg_relation_size)
  - Total relation size bytes (pg_total_relation_size — includes TOAST)
  - Params dict
  - Run id + ISO8601 timestamp

Results are merged into results/pgvs_benchmark/streamingdiskann_builds.json:
the file is read (if present), the entry for the current variant is replaced,
and the file is written back.

Usage:
    python scripts/build_streamingdiskann_variants.py \\
        --dsn "dbname=scix_pilot" \\
        --variant all

The script refuses to run against the production DSN (dbname=scix) unless
--allow-prod is passed. That guard is defined inline as ``assert_pilot_dsn``
to match the Layer 0 convention used by sibling scripts.
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

# Make ``src/`` importable when run as a script (mirrors other scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg  # noqa: E402

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_streamingdiskann_variants")

INDEX_NAME_PREFIX = "paper_embeddings_diskann"
RESULTS_PATH = Path("results/pgvs_benchmark/streamingdiskann_builds.json")


# ---------------------------------------------------------------------------
# Variant definitions — exact DDL strings so tests can assert keyword presence.
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict[str, Any]] = {
    "v1": {
        "ddl": (
            f"CREATE INDEX {INDEX_NAME_PREFIX}_v1 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WHERE model_name='indus';"
        ),
        "params": {
            "storage_layout": "plain",
            "num_neighbors": "default",
            "num_bits_per_dimension": None,
        },
    },
    "v2": {
        "ddl": (
            f"CREATE INDEX {INDEX_NAME_PREFIX}_v2 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WITH (num_bits_per_dimension = 2) "
            "WHERE model_name='indus';"
        ),
        "params": {
            "storage_layout": "plain",
            "num_neighbors": "default",
            "num_bits_per_dimension": 2,
        },
    },
    "v3": {
        "ddl": (
            f"CREATE INDEX {INDEX_NAME_PREFIX}_v3 ON paper_embeddings "
            "USING diskann (embedding halfvec_cosine_ops) "
            "WITH (storage_layout = 'memory_optimized', num_neighbors = 64, "
            "num_bits_per_dimension = 2) "
            "WHERE model_name='indus';"
        ),
        "params": {
            "storage_layout": "memory_optimized",
            "num_neighbors": 64,
            "num_bits_per_dimension": 2,
        },
    },
}

VARIANT_CHOICES = ["v1", "v2", "v3", "all"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def assert_pilot_dsn(dsn: str, allow_prod: bool = False) -> None:
    """Refuse to run against a production DSN unless explicitly overridden.

    Raises ``SystemExit(2)`` so the script terminates with the same exit code
    used by the other Layer 0 scripts for this condition.
    """
    if is_production_dsn(dsn) and not allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Index build + sizing
# ---------------------------------------------------------------------------


def _index_name_for(variant: str) -> str:
    return f"{INDEX_NAME_PREFIX}_{variant}"


def _drop_index_if_exists(conn: psycopg.Connection, index_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")
    conn.commit()


def _measure_sizes(conn: psycopg.Connection, index_name: str) -> tuple[int, int]:
    """Return (index_size_bytes, total_relation_size_bytes) for ``index_name``.

    Returns (0, 0) when the index is not present (e.g., dry-run or failed
    build) so downstream JSON always contains the keys.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_relation_size(c.oid), pg_total_relation_size(c.oid) "
            "FROM pg_class c WHERE c.relname = %s",
            (index_name,),
        )
        row = cur.fetchone()
    if row is None:
        return (0, 0)
    return (int(row[0]), int(row[1]))


def build_variant(
    dsn: str,
    variant: str,
    *,
    dry_run: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a single variant and return its result entry.

    The result dict contains every key required by the acceptance criteria:
    variant, ddl, build_wall_seconds, peak_rss_bytes, index_size_bytes,
    total_relation_size_bytes, params, run_id, timestamp.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    spec = VARIANTS[variant]
    ddl: str = spec["ddl"]
    params: dict[str, Any] = dict(spec["params"])
    index_name = _index_name_for(variant)
    effective_run_id = run_id or uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("Variant %s: DDL = %s", variant, ddl)

    if dry_run:
        logger.info("[dry-run] skipping DDL execution for %s", variant)
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "variant": variant,
            "ddl": ddl,
            "build_wall_seconds": 0.0,
            "peak_rss_bytes": int(peak_rss) * 1024,
            "index_size_bytes": 0,
            "total_relation_size_bytes": 0,
            "params": params,
            "run_id": effective_run_id,
            "timestamp": timestamp,
        }

    with psycopg.connect(dsn) as conn:
        _drop_index_if_exists(conn, index_name)

        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
        build_wall_seconds = time.perf_counter() - t0

        index_size, total_size = _measure_sizes(conn, index_name)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB; normalize to bytes.
    peak_rss_bytes = int(peak_rss_kb) * 1024

    result: dict[str, Any] = {
        "variant": variant,
        "ddl": ddl,
        "build_wall_seconds": float(build_wall_seconds),
        "peak_rss_bytes": peak_rss_bytes,
        "index_size_bytes": int(index_size),
        "total_relation_size_bytes": int(total_size),
        "params": params,
        "run_id": effective_run_id,
        "timestamp": timestamp,
    }

    logger.info(
        "Variant %s complete: build=%.2fs, index_size=%d bytes, total=%d bytes",
        variant, build_wall_seconds, index_size, total_size,
    )
    return result


# ---------------------------------------------------------------------------
# Merge-mode JSON persistence
# ---------------------------------------------------------------------------


def load_existing_results(path: Path) -> dict[str, Any]:
    """Read existing results JSON (if any) into an in-memory dict.

    The on-disk shape is ``{"variants": {variant_name: entry, ...}, ...}``.
    """
    if not path.exists():
        return {"variants": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Existing %s is not valid JSON — starting fresh", path)
        return {"variants": {}}
    if not isinstance(data, dict) or "variants" not in data:
        return {"variants": {}}
    if not isinstance(data.get("variants"), dict):
        data["variants"] = {}
    return data


def merge_and_write(path: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``entries`` (keyed by variant) into the file at ``path``.

    Creates parent directories as needed. Returns the final merged document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = load_existing_results(path)
    for entry in entries:
        doc["variants"][entry["variant"]] = entry
    doc["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pgvectorscale StreamingDiskANN index variants on paper_embeddings.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_PILOT_DSN", DEFAULT_DSN),
        help="PostgreSQL DSN for the pilot database (default: $SCIX_PILOT_DSN or $SCIX_DSN).",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        default="all",
        help="Which variant to build (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DDL and write a zero-time JSON entry without touching the DB.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against the production DSN.",
    )
    parser.add_argument(
        "--results-path",
        default=str(RESULTS_PATH),
        help=f"Path to merged JSON output (default: {RESULTS_PATH}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    assert_pilot_dsn(args.dsn, allow_prod=args.allow_prod)

    if args.variant == "all":
        targets = ["v1", "v2", "v3"]
    else:
        targets = [args.variant]

    run_id = uuid.uuid4().hex[:12]
    logger.info(
        "Building variants %s against %s (run_id=%s, dry_run=%s)",
        targets, redact_dsn(args.dsn), run_id, args.dry_run,
    )

    entries: list[dict[str, Any]] = []
    results_path = Path(args.results_path)

    for variant in targets:
        entry = build_variant(
            args.dsn,
            variant,
            dry_run=args.dry_run,
            run_id=run_id,
        )
        entries.append(entry)
        # Merge each entry as we go so a mid-run crash preserves earlier work.
        merge_and_write(results_path, [entry])

    doc = merge_and_write(results_path, entries)
    logger.info(
        "Wrote %d variant entries to %s (total variants on disk: %d)",
        len(entries), results_path, len(doc["variants"]),
    )

    # Emit a compact JSON summary on stdout for callers that want to pipe it.
    print(json.dumps({"variants": [e for e in entries]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
