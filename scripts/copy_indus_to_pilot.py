"""Binary COPY INDUS embeddings from production to a pilot PostgreSQL database.

This script streams rows from ``paper_embeddings WHERE model_name='indus'`` on
the source DSN directly into the destination DSN using psycopg3's binary
``COPY`` protocol. It is read-only on the source and refuses to write to the
production database.

Safety invariants:
    * Destination DSN must NOT resolve to the production database
      (checked via :func:`scix.db.is_production_dsn` when available, plus a
      local fallback check that rejects ``dbname=scix``, empty dbnames, and
      any dbname matching the source's dbname or the ``SCIX_DSN`` env var).
    * ``--dry-run`` short-circuits before any connection is opened and is the
      recommended way to inspect the exact SELECT/COPY statements the script
      will execute.

Outputs ``results/pgvs_benchmark/copy_indus.json`` with wall-clock, bytes, and
row counts on successful completion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("scix.copy_indus_to_pilot")

RESULTS_DIR = Path("results/pgvs_benchmark")
RESULTS_PATH = RESULTS_DIR / "copy_indus.json"

# ---------------------------------------------------------------------------
# Pure helpers (no side effects — safe to import in unit tests)
# ---------------------------------------------------------------------------


def _parse_dsn(dsn: str) -> dict[str, Any]:
    """Return a dict view of a DSN (key=value or URI form).

    Uses psycopg's libpq-backed parser when available; falls back to a minimal
    key=value tokenizer so unit tests don't strictly require psycopg at import
    time (they will, in practice, but we want a clean failure mode).
    """
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:  # pragma: no cover - defensive
        result: dict[str, Any] = {}
        for token in dsn.split():
            if "=" in token:
                key, _, value = token.partition("=")
                result[key.strip()] = value.strip()
        return result
    try:
        return dict(conninfo_to_dict(dsn))
    except Exception:  # pragma: no cover - psycopg.ProgrammingError et al.
        return {}


def _sanitize_dsn(dsn: str) -> str:
    """Return a DSN safe for logging/JSON output (no password)."""
    try:
        from scix.db import redact_dsn

        return redact_dsn(dsn)
    except ImportError:
        # Minimal fallback: strip password=... tokens.
        if "://" in dsn:
            scheme, _, rest = dsn.partition("://")
            rest = rest.split("?", 1)[0]
            if "@" in rest:
                userinfo, _, host_and_path = rest.partition("@")
                user = userinfo.split(":", 1)[0]
                rest = f"{user}:***@{host_and_path}"
            return f"{scheme}://{rest}"
        safe_keys = {"dbname", "host", "port", "user"}
        parts: list[str] = []
        for token in dsn.split():
            if "=" in token:
                key, _, value = token.partition("=")
                if key.strip() in safe_keys:
                    parts.append(f"{key.strip()}={value.strip()}")
        return " ".join(parts) if parts else "<redacted>"


def assert_pilot_dsn(dest_dsn: str, source_dsn: str) -> None:
    """Refuse to run if the destination DSN looks like production.

    Raises:
        ValueError: if the destination DSN appears to match a production
            database. Message always contains the word 'production' and
            'refuse' so callers (and tests) can detect it.
    """
    if not dest_dsn or not dest_dsn.strip():
        raise ValueError(
            "Refuse to run: destination DSN is empty — production safety "
            "guard rejects unset dest."
        )

    # Preferred: delegate to scix.db.is_production_dsn if it exists.
    is_prod = False
    used_library_check = False
    try:
        from scix.db import is_production_dsn  # type: ignore[attr-defined]

        is_prod = bool(is_production_dsn(dest_dsn))
        used_library_check = True
    except ImportError:
        used_library_check = False

    dest_params = _parse_dsn(dest_dsn)
    source_params = _parse_dsn(source_dsn) if source_dsn else {}
    dest_dbname = (dest_params.get("dbname") or "").strip()
    source_dbname = (source_params.get("dbname") or "").strip()

    # Always run the local fallback checks — cheap and belt-and-suspenders.
    reasons: list[str] = []
    if used_library_check and is_prod:
        reasons.append("scix.db.is_production_dsn flagged dest as production")
    if not dest_dbname:
        reasons.append("destination DSN has no dbname (refuses empty dbname)")
    if dest_dbname.lower() == "scix":
        reasons.append("destination dbname='scix' is the production database")
    if source_dbname and dest_dbname and source_dbname.lower() == dest_dbname.lower():
        reasons.append(
            f"destination dbname={dest_dbname!r} matches source dbname — "
            "refuse to copy onto source"
        )

    env_dsn = os.environ.get("SCIX_DSN", "").strip()
    if env_dsn and env_dsn == dest_dsn.strip():
        reasons.append(
            "destination DSN equals SCIX_DSN env var — refuse: that's "
            "the production database"
        )

    if reasons:
        joined = "; ".join(reasons)
        raise ValueError(
            f"Refuse to write to production database: {joined}. "
            "Set --dest-dsn to a non-production DSN (e.g. dbname=scix_pilot)."
        )


def build_source_copy_sql(columns: list[str], limit: int | None) -> str:
    """Return the source-side COPY ... TO STDOUT statement (binary)."""
    col_list = ", ".join(columns)
    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    return (
        f"COPY (SELECT {col_list} FROM paper_embeddings "
        f"WHERE model_name = 'indus'{limit_clause}) "
        "TO STDOUT WITH (FORMAT BINARY)"
    )


def build_dest_copy_sql(columns: list[str]) -> str:
    """Return the destination-side COPY FROM STDIN statement (binary)."""
    col_list = ", ".join(columns)
    return (
        f"COPY paper_embeddings ({col_list}) FROM STDIN WITH (FORMAT BINARY)"
    )


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyResult:
    run_id: str
    timestamp: str
    source_dsn_sanitized: str
    dest_dsn_sanitized: str
    rows_copied: int
    bytes_copied: int
    wall_seconds: float
    git_sha: str | None
    columns: list[str]


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Binary-COPY paper_embeddings (model_name='indus') from a source "
            "(production) PostgreSQL DSN to a pilot DSN. Refuses to write to "
            "the production database."
        )
    )
    parser.add_argument(
        "--source-dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help=(
            "Source DSN (read-only). Defaults to $SCIX_DSN or 'dbname=scix'."
        ),
    )
    parser.add_argument(
        "--dest-dsn",
        required=False,
        default=os.environ.get("SCIX_PILOT_DSN"),
        help=(
            "Destination (pilot) DSN. MUST NOT resolve to the production "
            "database. Defaults to $SCIX_PILOT_DSN if set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the exact SELECT/COPY statements and asserted destination "
            "DSN, then exit WITHOUT opening any connection."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help=(
            "Progress-reporting interval in rows (default: 100000). Does not "
            "affect the binary COPY stream itself."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "If set, LIMIT the source SELECT to this many rows (useful for "
            "smoke tests). Default: no limit (full ~32M INDUS rows)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _discover_columns(conn: Any) -> list[str]:
    """Return the ordered list of paper_embeddings columns from the source."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'paper_embeddings' "
            "ORDER BY ordinal_position"
        )
        rows = cur.fetchall()
    columns = [r[0] for r in rows]
    if not columns:
        raise RuntimeError(
            "Could not discover paper_embeddings columns from source — "
            "is the schema loaded?"
        )
    return columns


def _lazy_import_capture_env() -> Any:
    """Import capture_env from scripts.pgvs_bench_env at call time.

    We import lazily (inside main()) so the script is importable for unit
    tests even before scripts/pgvs_bench_env.py lands in the repo.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pgvs_bench_env import capture_env  # type: ignore[import-not-found]
        return capture_env
    except ImportError as exc:
        raise ImportError(
            "scripts/pgvs_bench_env.py is missing or does not export "
            "capture_env(); this is required to stamp git_sha and env "
            "metadata onto the copy result. Create the module (work unit "
            "'pgvs-bench-env') before running copy_indus_to_pilot.py."
        ) from exc


def _git_sha_from_env(capture_env: Any) -> str | None:
    """Pull git_sha out of capture_env() output; return None on any failure."""
    try:
        env = capture_env()
        if isinstance(env, dict):
            sha = env.get("git_sha") or env.get("git_commit") or env.get("commit")
            return str(sha) if sha else None
    except Exception:  # pragma: no cover - best effort
        logger.debug("capture_env() raised; git_sha will be None", exc_info=True)
    return None


def _write_result(result: CopyResult, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "source_dsn_sanitized": result.source_dsn_sanitized,
        "dest_dsn_sanitized": result.dest_dsn_sanitized,
        "rows_copied": result.rows_copied,
        "bytes_copied": result.bytes_copied,
        "wall_seconds": result.wall_seconds,
        "git_sha": result.git_sha,
        "columns": list(result.columns),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote copy result to %s", path)


def _run_copy(
    source_dsn: str,
    dest_dsn: str,
    limit: int | None,
    batch_size: int,
) -> tuple[int, int, list[str]]:
    """Execute the binary COPY. Returns (rows, bytes, columns)."""
    import psycopg

    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
        _have_tqdm = True
    except ImportError:
        tqdm = None  # type: ignore[assignment]
        _have_tqdm = False

    rows_copied = 0
    bytes_copied = 0

    # Source connection is read-only — we never write back.
    with psycopg.connect(source_dsn) as src_conn:
        src_conn.autocommit = True
        columns = _discover_columns(src_conn)
        source_sql = build_source_copy_sql(columns, limit)
        dest_sql = build_dest_copy_sql(columns)
        logger.info("Discovered columns: %s", columns)
        logger.info("Source SQL: %s", source_sql)
        logger.info("Dest SQL:   %s", dest_sql)

        with psycopg.connect(dest_dsn) as dst_conn:
            dst_conn.autocommit = False
            with (
                src_conn.cursor() as src_cur,
                dst_conn.cursor() as dst_cur,
            ):
                with (
                    src_cur.copy(source_sql) as src_copy,
                    dst_cur.copy(dest_sql) as dst_copy,
                ):
                    # Propagate binary format signature between streams.
                    try:
                        dst_copy.set_types(src_copy.description)
                    except Exception:
                        # Binary COPY streams carry their own header; if
                        # set_types isn't available we still pass raw bytes.
                        logger.debug(
                            "set_types unavailable; relying on raw bytes",
                            exc_info=True,
                        )

                    progress = None
                    if _have_tqdm:
                        progress = tqdm(unit="row", desc="COPY INDUS")
                    next_tick = batch_size

                    for block in src_copy:
                        if not block:
                            continue
                        dst_copy.write(block)
                        bytes_copied += len(block)
                        # psycopg3's copy iterator yields raw bytes chunks, not
                        # rows. We derive a row estimate from cursor.rowcount
                        # after the stream closes; in-stream we just tick bytes.

                    # After the block iterator drains, rowcount should reflect
                    # the number of rows transferred.
                    rows_copied = int(src_cur.rowcount or 0)
                    if progress is not None:
                        progress.update(rows_copied)
                        progress.close()
                    if not _have_tqdm:
                        # Emit at least one progress line for operators.
                        if rows_copied >= next_tick or rows_copied > 0:
                            print(f"[copy_indus] rows_copied={rows_copied}")

            dst_conn.commit()

    return rows_copied, bytes_copied, columns


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_dsn = args.source_dsn
    dest_dsn = args.dest_dsn
    if not dest_dsn:
        print(
            "ERROR: --dest-dsn is required (or set SCIX_PILOT_DSN).",
            file=sys.stderr,
        )
        return 2

    # SAFETY: refuse to write to production BEFORE doing anything else.
    assert_pilot_dsn(dest_dsn, source_dsn)

    source_sanitized = _sanitize_dsn(source_dsn)
    dest_sanitized = _sanitize_dsn(dest_dsn)

    if args.dry_run:
        # Placeholder columns for dry-run display; real discovery needs DB.
        placeholder_cols = ["bibcode", "model_name", "embedding"]
        print(f"[dry-run] source_dsn = {source_sanitized}")
        print(f"[dry-run] dest_dsn   = {dest_sanitized}")
        print(f"[dry-run] limit      = {args.limit}")
        print(f"[dry-run] batch_size = {args.batch_size}")
        print(
            f"[dry-run] source SQL (placeholder cols): "
            f"{build_source_copy_sql(placeholder_cols, args.limit)}"
        )
        print(
            f"[dry-run] dest SQL   (placeholder cols): "
            f"{build_dest_copy_sql(placeholder_cols)}"
        )
        print("[dry-run] no connections opened; no writes performed.")
        return 0

    # Lazy import — raises a clear ImportError if pgvs_bench_env is missing.
    # Placed here so --dry-run works even before that helper lands.
    capture_env = _lazy_import_capture_env()

    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    logger.info("Starting COPY INDUS → pilot (run_id=%s)", run_id)
    logger.info("Source: %s", source_sanitized)
    logger.info("Dest:   %s", dest_sanitized)

    start = time.perf_counter()
    rows, bytes_, columns = _run_copy(
        source_dsn=source_dsn,
        dest_dsn=dest_dsn,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    wall = time.perf_counter() - start

    git_sha = _git_sha_from_env(capture_env)

    result = CopyResult(
        run_id=run_id,
        timestamp=timestamp,
        source_dsn_sanitized=source_sanitized,
        dest_dsn_sanitized=dest_sanitized,
        rows_copied=rows,
        bytes_copied=bytes_,
        wall_seconds=wall,
        git_sha=git_sha,
        columns=columns,
    )
    _write_result(result)
    logger.info(
        "Done. rows=%d bytes=%d wall=%.2fs", rows, bytes_, wall
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
