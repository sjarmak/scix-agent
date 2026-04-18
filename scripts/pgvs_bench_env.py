#!/usr/bin/env python3
"""Capture environment metadata for pgvector/pgvectorscale benchmark runs.

Writes a JSON document describing the machine, git state, and Postgres
extension versions under which a benchmark was executed. Benchmark
scripts import ``capture_env`` and call it once per run to stamp their
output artifacts so results can be correlated with hardware/software
versions later.

The module is designed to degrade gracefully: missing extensions, an
unreachable database, or a hardware probe that fails must never raise
out of ``capture_env``. Each unavailable field becomes ``None`` and the
corresponding error message is attached under a ``_*_error`` key.

Usage as a script:

    python scripts/pgvs_bench_env.py \
        --dsn "dbname=scix" \
        --out results/pgvs_benchmark/env.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("results/pgvs_benchmark/env.json")
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_command(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a command and return stripped stdout, or None on any failure."""
    try:
        result = subprocess.run(  # noqa: S603 - trusted, fixed argv
            args,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("command %s failed: %s", args, exc)
        return None
    if result.returncode != 0:
        logger.debug("command %s exited %s: %s", args, result.returncode, result.stderr)
        return None
    return result.stdout.strip()


def _git_sha() -> str | None:
    return _run_command(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT)


def _git_branch() -> str | None:
    return _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO_ROOT)


def _cpu_model() -> str | None:
    """Return CPU model string, preferring /proc/cpuinfo then platform."""
    cpuinfo = Path("/proc/cpuinfo")
    try:
        if cpuinfo.exists():
            with cpuinfo.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        _, _, value = line.partition(":")
                        value = value.strip()
                        if value:
                            return value
    except OSError as exc:
        logger.debug("/proc/cpuinfo read failed: %s", exc)
    fallback = platform.processor()
    return fallback or None


def _ram_gb() -> float | None:
    """Return total system RAM in GB rounded to 1 decimal."""
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.debug("psutil unavailable: %s", exc)
        return None
    try:
        total = psutil.virtual_memory().total
    except Exception as exc:  # noqa: BLE001 - psutil may raise various OS errors
        logger.debug("psutil.virtual_memory failed: %s", exc)
        return None
    return round(total / (1024**3), 1)


def _disk_info() -> dict[str, Any] | None:
    """Return first physical disk info via lsblk, or None on failure."""
    raw = _run_command(["lsblk", "-d", "-o", "NAME,MODEL,SIZE", "--json"])
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("lsblk JSON parse failed: %s", exc)
        return None
    devices = data.get("blockdevices")
    if not isinstance(devices, list) or not devices:
        return None
    first = devices[0]
    if not isinstance(first, dict):
        return None
    return {
        "name": first.get("name"),
        "model": first.get("model"),
        "size": first.get("size"),
    }


def _postgres_info(dsn: str | None) -> dict[str, Any]:
    """Query Postgres for server version and extension versions.

    Returns a dict with keys ``postgres_version``, ``extensions`` and
    optionally ``_db_error``. Extensions keys ``vector`` and
    ``vectorscale`` are always present; missing ones are ``None``.
    """
    info: dict[str, Any] = {
        "postgres_version": None,
        "extensions": {"vector": None, "vectorscale": None},
    }
    if dsn is None:
        info["_db_error"] = "no dsn provided"
        return info
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError as exc:
        info["_db_error"] = f"psycopg import failed: {exc}"
        return info
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                row = cur.fetchone()
                if row is not None:
                    info["postgres_version"] = row[0]
                cur.execute(
                    "SELECT extname, extversion FROM pg_extension "
                    "WHERE extname IN ('vector', 'vectorscale')"
                )
                for extname, extversion in cur.fetchall():
                    info["extensions"][extname] = extversion
    except Exception as exc:  # noqa: BLE001 - any DB error must not raise
        info["_db_error"] = f"{type(exc).__name__}: {exc}"
    return info


def capture_env(dsn: str | None = None) -> dict[str, Any]:
    """Capture environment metadata for a benchmark run.

    Args:
        dsn: Postgres DSN to probe for server/extension versions. If
            None or unreachable, DB-derived fields are None and a
            ``_db_error`` key is attached.

    Returns:
        Mapping with keys: run_id, timestamp, git_sha, postgres_version,
        extensions, cpu, ram_gb, disk, git_branch.
    """
    pg = _postgres_info(dsn)
    env: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "postgres_version": pg.get("postgres_version"),
        "extensions": pg.get("extensions", {"vector": None, "vectorscale": None}),
        "cpu": _cpu_model(),
        "ram_gb": _ram_gb(),
        "disk": _disk_info(),
    }
    if "_db_error" in pg:
        env["_db_error"] = pg["_db_error"]
    return env


def write_env(env: dict[str, Any], out_path: Path) -> None:
    """Write env dict to out_path as pretty JSON, creating parents."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture benchmark environment metadata to JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=None,
        help="Postgres DSN to probe (default: no DB probe)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    env = capture_env(dsn=args.dsn)
    write_env(env, args.out)
    logger.info("wrote %s (run_id=%s)", args.out, env["run_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
