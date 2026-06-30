#!/usr/bin/env python3
"""Seal ``papers_fulltext`` JSONB for a closed year into a read-only NAS shard.

ADR-016, Phase 1. This script is **read-only against Postgres** — it streams
``papers_fulltext`` rows for a year and writes them to a per-year SQLite shard on
NAS. It does NOT delete anything from Postgres; the destructive prune step is a
separate, gated operation.

Usage::

    # build the shard for one year (or a sample with --limit)
    python scripts/seal_fulltext_to_nas.py build --year 2020 --root /mnt/scix_coldtext/v1
    python scripts/seal_fulltext_to_nas.py build --year 2020 --limit 2000 --root /mnt/scix_coldtext/v1

    # verify shard parity vs Postgres (row count + per-bibcode deep equality)
    python scripts/seal_fulltext_to_nas.py verify --year 2020 --root /mnt/scix_coldtext/v1 --sample 500

Heavy full-year builds must run under ``scix-batch`` (cgroup memory bounds).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

# Allow running as a plain script (scripts/ is not a package), independent of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.coldtext import (
    ColdTextReader,
    ShardStats,
    sections_checksum,
    shard_path,
    write_shard,
)
from scix.db import DEFAULT_DSN, get_connection, redact_dsn

DEFAULT_ROOT = "/mnt/scix_coldtext/v1"

# Columns pulled from papers_fulltext. sections_tsv is intentionally excluded —
# it stays in Postgres so section BM25 keeps serving sealed years.
_SELECT_COLS = (
    "pf.bibcode",
    "pf.source",
    "pf.parser_version",
    "pf.canonical_bibcode",
    "pf.suppressed_by_publisher",
    "pf.sections",
    "pf.figures",
    "pf.tables",
    "pf.equations",
)


def _bound_session(conn: psycopg.Connection) -> None:
    """Defensive resource bounds (PG-side OOM rules from CLAUDE.md)."""
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '128MB'")


def _year_pred(year: int) -> tuple[str, tuple[object, ...]]:
    """WHERE fragment selecting a year. ``year == 0`` is the sentinel for
    NULL-year rows (treated as sealed: unknown year is old/safe)."""
    if year == 0:
        return "p.year IS NULL", ()
    return "p.year = %s", (year,)


def _year_count(conn: psycopg.Connection, year: int) -> int:
    pred, params = _year_pred(year)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM papers_fulltext pf "
            f"JOIN papers p ON p.bibcode = pf.bibcode WHERE {pred}",
            params,
        )
        (n,) = cur.fetchone()
    return int(n)


def _stream_year(
    conn: psycopg.Connection, year: int, limit: int | None
) -> Iterator[Mapping[str, object]]:
    """Server-side cursor over one year's papers_fulltext rows."""
    pred, params = _year_pred(year)
    sql = (
        f"SELECT {', '.join(_SELECT_COLS)} FROM papers_fulltext pf "
        f"JOIN papers p ON p.bibcode = pf.bibcode WHERE {pred}"
    )
    if limit is not None:
        sql += " LIMIT %s"
        params = (*params, limit)
    with conn.cursor(name="seal_stream", row_factory=dict_row) as cur:
        cur.itersize = 1000
        cur.execute(sql, params)
        for row in cur:
            yield row


def _sealed_year_counts(conn: psycopg.Connection, before: int) -> list[tuple[int, int]]:
    """``(year, rows)`` for every sealed year with full-text, NULL-year as 0.

    One grouped scan (bounded by the caller's session settings) so the driver
    only builds non-empty years.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(p.year, 0) AS y, count(*) "
            "FROM papers_fulltext pf JOIN papers p ON p.bibcode = pf.bibcode "
            "WHERE p.year < %s OR p.year IS NULL "
            "GROUP BY COALESCE(p.year, 0) ORDER BY 1",
            (before,),
        )
        return [(int(y), int(n)) for y, n in cur.fetchall()]


def _build_one(conn: psycopg.Connection, year: int, root: str, limit: int | None) -> ShardStats:
    path = shard_path(root, year)
    t0 = time.perf_counter()
    if limit is None:
        total = _year_count(conn, year)
        print(f"[seal] year={year}: {total} rows -> {path}")
    stats = write_shard(path, year, _stream_year(conn, year, limit))
    dt = time.perf_counter() - t0
    size_mb = path.stat().st_size / 1e6
    rate = f"{stats.rows / dt:.0f} rows/s" if dt > 0 else "n/a"
    print(f"[seal] year={year}: wrote {stats.rows} rows, {size_mb:.1f} MB, {dt:.1f}s ({rate})")
    return stats


def cmd_build(args: argparse.Namespace) -> int:
    with get_connection(args.dsn) as conn:
        _bound_session(conn)
        print(f"[seal] DSN={redact_dsn(args.dsn)} -> {args.root}")
        _build_one(conn, args.year, args.root, args.limit)
    return 0


def cmd_build_sealed(args: argparse.Namespace) -> int:
    """Build shards for every sealed year (< --before, plus NULL-year as 0)."""
    if not os.environ.get("INVOCATION_ID"):
        print(
            "[seal] WARNING: not running under scix-batch (INVOCATION_ID unset). "
            "A full sealed build streams millions of rows; run via "
            "`scix-batch` to bound memory (CLAUDE.md OOM-isolation rule).",
            file=sys.stderr,
        )
    with get_connection(args.dsn) as conn:
        _bound_session(conn)
        print(f"[seal] DSN={redact_dsn(args.dsn)} before={args.before} -> {args.root}")
        years = _sealed_year_counts(conn, args.before)
        total = sum(n for _, n in years)
        print(f"[seal] {len(years)} sealed years, {total} rows total")
        built = 0
        for year, n in years:
            if n == 0:
                continue
            stats = _build_one(conn, year, args.root, None)
            built += stats.rows
        print(f"[seal] done: {built} rows across {len(years)} shards")
    return 0


def _verify_one(
    conn: psycopg.Connection, reader: ColdTextReader, year: int, sample_n: int
) -> tuple[int, int, int]:
    """Return ``(shard_rows, checked, mismatches)`` for one shard.

    Parity = shard count vs PG count for the year + per-bibcode sections
    checksum on a deterministic random sample.
    """
    shard_n = reader.row_count()
    pg_n = _year_count(conn, year)
    rows_ok = shard_n == pg_n
    bibcodes = list(reader.iter_bibcodes())
    sample = bibcodes
    if sample_n and len(bibcodes) > sample_n:
        sample = random.Random(year).sample(bibcodes, sample_n)

    mismatches = 0 if rows_ok else 1
    if not rows_ok:
        print(f"[verify] year={year}: ROW COUNT shard={shard_n} pg={pg_n}")
    with conn.cursor(row_factory=dict_row) as cur:
        for bib in sample:
            cur.execute("SELECT sections FROM papers_fulltext WHERE bibcode = %s", (bib,))
            pg_row = cur.fetchone()
            if pg_row is None:
                print(f"[verify] year={year}: MISSING in PG: {bib}")
                mismatches += 1
                continue
            if sections_checksum(pg_row["sections"]) != sections_checksum(
                reader.fetch_sections(bib)
            ):
                print(f"[verify] year={year}: CHECKSUM MISMATCH: {bib}")
                mismatches += 1
    return shard_n, len(sample), mismatches


def cmd_verify(args: argparse.Namespace) -> int:
    path = shard_path(args.root, args.year)
    with ColdTextReader(path) as reader, get_connection(args.dsn) as conn:
        _bound_session(conn)
        shard_n, checked, mismatches = _verify_one(conn, reader, args.year, args.sample)
    if mismatches:
        print(f"[verify] FAIL — year={args.year}: {mismatches} mismatched")
        return 1
    print(f"[verify] PASS — year={args.year}: {checked} checked, shard rows={shard_n}")
    return 0


def cmd_verify_sealed(args: argparse.Namespace) -> int:
    """Verify every sealed-year shard found under --root against PG."""
    with get_connection(args.dsn) as conn:
        _bound_session(conn)
        years = [y for y, n in _sealed_year_counts(conn, args.before) if n > 0]
        total_bad = 0
        for year in years:
            path = shard_path(args.root, year)
            if not path.exists():
                print(f"[verify] year={year}: SHARD MISSING at {path}")
                total_bad += 1
                continue
            with ColdTextReader(path) as reader:
                _, checked, mismatches = _verify_one(conn, reader, year, args.sample)
            if mismatches:
                total_bad += 1
            else:
                print(f"[verify] year={year}: PASS ({checked} checked)")
    if total_bad:
        print(f"[verify] FAIL — {total_bad}/{len(years)} years had problems")
        return 1
    print(f"[verify] PASS — all {len(years)} sealed-year shards match")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build one year shard (read-only on PG)")
    b.add_argument("--year", type=int, required=True)
    b.add_argument("--limit", type=int, default=None, help="cap rows (sampling)")
    b.set_defaults(func=cmd_build)

    bs = sub.add_parser("build-sealed", help="build all sealed-year shards")
    bs.add_argument("--before", type=int, default=2025, help="hot-window start year")
    bs.set_defaults(func=cmd_build_sealed)

    v = sub.add_parser("verify", help="verify one shard vs PG")
    v.add_argument("--year", type=int, required=True)
    v.add_argument("--sample", type=int, default=500, help="bibcodes to deep-check")
    v.set_defaults(func=cmd_verify)

    vs = sub.add_parser("verify-sealed", help="verify all sealed-year shards vs PG")
    vs.add_argument("--before", type=int, default=2025, help="hot-window start year")
    vs.add_argument("--sample", type=int, default=200, help="bibcodes/year to deep-check")
    vs.set_defaults(func=cmd_verify_sealed)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
