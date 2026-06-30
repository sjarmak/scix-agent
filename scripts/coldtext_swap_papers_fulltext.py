#!/usr/bin/env python3
"""ADR-016 Phase 1b: rebuild papers_fulltext hot-only and reclaim the monolith.

DESTRUCTIVE. Three explicit, separately-invoked phases:

  build     CREATE papers_fulltext_hot (hot years only) + indexes + FK, verify.
            Additive and reversible (just drop papers_fulltext_hot to undo).
  swap      One txn: drop the route view, rename papers_fulltext -> _old,
            rename papers_fulltext_hot -> papers_fulltext, recreate the view.
            papers_fulltext_old is RETAINED as the rollback.
  drop-old  DROP papers_fulltext_old -> reclaims ~470 GB. Run only after reads
            are verified. Also renames the _hot-suffixed indexes/constraints back
            to canonical names (collision-free once _old is gone).

Guards: requires --allow-prod and SYSTEMD_SCOPE (run under scix-batch). Bounds
work_mem and disables parallel workers (PG-side OOM rules, CLAUDE.md).

Prerequisites (verified out-of-band before running): the sealed-year NAS shards
exist and verify-sealed has passed, and the cold read route (Phase 1a) is live.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from scix.coldtext import HOT_WINDOW_START_YEAR  # noqa: E402
from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402

SRC = "papers_fulltext"
HOT = "papers_fulltext_hot"
OLD = "papers_fulltext_old"
VIEW = "v_papers_fulltext_route_inputs"

# Indexes/constraints to (re)create on the hot table, then rename to canonical
# in drop-old. (canonical_name, hot_name, create_sql_on_hot).
_HOT_INDEXES = [
    (
        "idx_papers_fulltext_canonical_bibcode",
        "idx_papers_fulltext_hot_canonical_bibcode",
        f"CREATE INDEX idx_papers_fulltext_hot_canonical_bibcode ON {HOT} (canonical_bibcode)",
    ),
    (
        "idx_papers_fulltext_sections_tsv",
        "idx_papers_fulltext_hot_sections_tsv",
        f"CREATE INDEX idx_papers_fulltext_hot_sections_tsv ON {HOT} USING gin (sections_tsv)",
    ),
    (
        "idx_papers_fulltext_source",
        "idx_papers_fulltext_hot_source",
        f"CREATE INDEX idx_papers_fulltext_hot_source ON {HOT} (source)",
    ),
    (
        "idx_papers_fulltext_suppressed_by_publisher",
        "idx_papers_fulltext_hot_suppressed_by_publisher",
        f"CREATE INDEX idx_papers_fulltext_hot_suppressed_by_publisher ON {HOT} "
        "(suppressed_by_publisher) WHERE suppressed_by_publisher = true",
    ),
]


def _guard(args: argparse.Namespace) -> None:
    if not args.allow_prod and is_production_dsn(args.dsn):
        raise SystemExit("refusing to mutate prod without --allow-prod")
    if not os.environ.get("INVOCATION_ID"):
        raise SystemExit("refusing to run outside scix-batch (INVOCATION_ID unset)")


def _bound(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute("SET work_mem = '256MB'")
        cur.execute("SET maintenance_work_mem = '1GB'")


def _scalar(conn: psycopg.Connection, sql: str, params: tuple = ()) -> object:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return row[0] if row else None


def _exists(conn: psycopg.Connection, relname: str) -> bool:
    return bool(_scalar(conn, "SELECT to_regclass(%s)", (f"public.{relname}",)))


def cmd_build(args: argparse.Namespace) -> int:
    _guard(args)
    with get_connection(args.dsn) as conn:
        _bound(conn)
        print(f"[swap] DSN={redact_dsn(args.dsn)} build {HOT} (year >= {HOT_WINDOW_START_YEAR})")
        if _exists(conn, HOT):
            raise SystemExit(f"{HOT} already exists — drop it first or resume at swap")
        hot_n = _scalar(
            conn,
            f"SELECT count(*) FROM {SRC} pf JOIN papers p ON p.bibcode=pf.bibcode "
            "WHERE p.year >= %s",
            (HOT_WINDOW_START_YEAR,),
        )
        print(f"[swap] expected hot rows: {hot_n}")
        with conn.cursor() as cur:
            cur.execute(f"CREATE TABLE {HOT} (LIKE {SRC} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)")
            t0 = time.perf_counter()
            cur.execute(
                f"INSERT INTO {HOT} SELECT pf.* FROM {SRC} pf "
                "JOIN papers p ON p.bibcode = pf.bibcode WHERE p.year >= %s",
                (HOT_WINDOW_START_YEAR,),
            )
            print(f"[swap] inserted {cur.rowcount} rows in {time.perf_counter() - t0:.1f}s")
            cur.execute(f"ALTER TABLE {HOT} ADD CONSTRAINT {HOT}_pkey PRIMARY KEY (bibcode)")
            for _canon, _hot, ddl in _HOT_INDEXES:
                print(f"[swap] {ddl.split(' ON ')[0]}")
                cur.execute(ddl)
            cur.execute(
                f"ALTER TABLE {HOT} ADD CONSTRAINT {HOT}_bibcode_fkey "
                "FOREIGN KEY (bibcode) REFERENCES papers(bibcode)"
            )
            conn.commit()
        got = _scalar(conn, f"SELECT count(*) FROM {HOT}")
        if got != hot_n:
            raise SystemExit(f"row count mismatch: hot has {got}, expected {hot_n}")
        print(
            f"[swap] build OK: {HOT} has {got} rows, indexes+FK created. Reversible (drop {HOT})."
        )
    return 0


def cmd_swap(args: argparse.Namespace) -> int:
    _guard(args)
    with get_connection(args.dsn) as conn:
        _bound(conn)
        if not _exists(conn, HOT):
            raise SystemExit(f"{HOT} missing — run `build` first")
        if _exists(conn, OLD):
            raise SystemExit(f"{OLD} already exists — swap already ran?")
        viewdef = _scalar(conn, f"SELECT pg_get_viewdef('{VIEW}'::regclass, true)")
        if not viewdef:
            raise SystemExit(f"could not read {VIEW} definition")
        with conn.cursor() as cur:
            print("[swap] BEGIN swap txn")
            cur.execute(f"DROP VIEW {VIEW}")
            cur.execute(f"ALTER TABLE {SRC} RENAME TO {OLD}")
            cur.execute(f"ALTER TABLE {HOT} RENAME TO {SRC}")
            cur.execute(f"CREATE VIEW {VIEW} AS {viewdef}")
            conn.commit()
        live_n = _scalar(conn, f"SELECT count(*) FROM {SRC}")
        old_n = _scalar(conn, f"SELECT count(*) FROM {OLD}")
        print(f"[swap] swapped. live {SRC}={live_n} rows, {OLD}={old_n} rows (rollback kept).")
        print("[swap] verify reads, then run `drop-old` to reclaim.")
    return 0


def cmd_drop_old(args: argparse.Namespace) -> int:
    _guard(args)
    with get_connection(args.dsn) as conn:
        _bound(conn)
        if not _exists(conn, OLD):
            raise SystemExit(f"{OLD} missing — nothing to drop")
        free_before = _scalar(conn, "SELECT pg_size_pretty(pg_total_relation_size(%s))", (OLD,))
        with conn.cursor() as cur:
            print(f"[swap] dropping {OLD} ({free_before}) — reclaiming space")
            cur.execute(f"DROP TABLE {OLD}")
            # Canonical index/constraint names are now free.
            cur.execute(f"ALTER INDEX {HOT}_pkey RENAME TO {SRC}_pkey")
            cur.execute(
                f"ALTER TABLE {SRC} RENAME CONSTRAINT {HOT}_bibcode_fkey TO {SRC}_bibcode_fkey"
            )
            for canon, hot, _ddl in _HOT_INDEXES:
                cur.execute(f"ALTER INDEX {hot} RENAME TO {canon}")
            conn.commit()
        print(f"[swap] {OLD} dropped; indexes/constraints renamed to canonical. Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--allow-prod", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    sub.add_parser("swap").set_defaults(func=cmd_swap)
    sub.add_parser("drop-old").set_defaults(func=cmd_drop_old)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
