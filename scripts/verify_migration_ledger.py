#!/usr/bin/env python3
"""Assert that ``schema_migrations`` agrees with the actual catalog for 069-074.

This is the command behind GOAL.md acceptance criterion **A12** ("for each of
069-072, ``schema_migrations`` agrees with whether the objects actually exist in
``scix``"). It encodes, per migration, the objects that migration creates or
removes, looks them up in the live catalog, and compares that against whether the
ledger claims the migration is applied. Any disagreement is a divergence and the
script exits non-zero.

Read-only by construction: every query is a catalog lookup (``to_regclass``,
``information_schema.columns``, ``pg_proc``) plus one ``SELECT`` from the 58-row
``schema_migrations`` table. It never reads user data, never writes, and cannot
be slow, so it carries no ``--allow-prod`` gate or ``scix-batch`` requirement.

Usage::

    python scripts/verify_migration_ledger.py                      # SCIX_DSN, default dbname=scix
    python scripts/verify_migration_ledger.py --dsn "dbname=scix_test"
    python scripts/verify_migration_ledger.py --verbose            # show every probe

Evidence and the reconciliation table this script enforces:
docs/ops/migration_ledger_reconciliation_2026-07-27.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import psycopg

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from scix.db import DEFAULT_DSN, redact_dsn  # noqa: E402

# Free-text marker that migration 074's ledger row for version 70 must carry.
# Version 70 is recorded *only* to stop a future runner attempting an unrunnable
# migration; without this marker the row would read as a plain "applied" claim,
# which is the exact falsehood this script exists to catch.
#
# "not in force" rather than "not applied" because the row has to be truthful on
# both databases: 070 never ran against `scix`, but it did run against
# `scix_test` and migration 074 then dropped its objects. What is true in both
# cases is that its objects are intentionally absent.
SUPERSEDED_MARKER = "SUPERSEDED, NOT IN FORCE"


class Probe:
    """Read-only catalog lookups. One instance per connection."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def relation_exists(self, qualified_name: str) -> bool:
        row = self._conn.execute("select to_regclass(%s)", (qualified_name,)).fetchone()
        return row is not None and row[0] is not None

    def column_exists(self, table: str, column: str) -> bool:
        row = self._conn.execute(
            "select 1 from information_schema.columns "
            "where table_schema = 'public' and table_name = %s and column_name = %s",
            (table, column),
        ).fetchone()
        return row is not None

    def function_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
            "where n.nspname = 'public' and p.proname = %s",
            (name,),
        ).fetchone()
        return row is not None


@dataclass(frozen=True)
class Check:
    """One migration's expected relationship between the ledger and the catalog.

    ``applied_in_catalog`` returns True when the migration's post-condition holds
    in the database — for a create-migration that the objects exist, for a
    drop-migration that they are gone.
    """

    version: int
    filename: str
    summary: str
    applied_in_catalog: Callable[[Probe], bool]


CHECKS: tuple[Check, ...] = (
    Check(
        version=69,
        filename="069_drop_papers_raw.sql",
        summary="drops papers.raw JSONB",
        applied_in_catalog=lambda p: not p.column_exists("papers", "raw"),
    ),
    Check(
        version=70,
        filename="070_embedding_outbox.sql",
        summary="creates embedding_outbox + drain index + enqueue fn + trigger",
        applied_in_catalog=lambda p: (
            p.relation_exists("public.embedding_outbox")
            and p.relation_exists("public.idx_embedding_outbox_drain")
            and p.function_exists("embedding_outbox_enqueue")
        ),
    ),
    Check(
        version=71,
        filename="071_drop_paper_embeddings_indus_indexes.sql",
        summary="drops idx_embed_hnsw_indus{,_hv}",
        applied_in_catalog=lambda p: not p.relation_exists("public.idx_embed_hnsw_indus")
        and not p.relation_exists("public.idx_embed_hnsw_indus_hv"),
    ),
    Check(
        version=72,
        filename="072_indus_qdrant_synced.sql",
        summary="creates indus_qdrant_synced watermark",
        applied_in_catalog=lambda p: p.relation_exists("public.indus_qdrant_synced"),
    ),
    Check(
        version=73,
        filename="073_reconcile_schema_migrations.sql",
        summary="adds schema_migrations.note",
        applied_in_catalog=lambda p: p.column_exists("schema_migrations", "note"),
    ),
    Check(
        version=74,
        filename="074_record_paper_embeddings_retirement.sql",
        summary="records the paper_embeddings + embedding_outbox retirement",
        applied_in_catalog=lambda p: not p.relation_exists("public.paper_embeddings")
        and not p.relation_exists("public.embedding_outbox"),
    ),
)


@dataclass(frozen=True)
class LedgerRow:
    version: int
    filename: str
    note: str | None


def read_ledger(conn: psycopg.Connection, versions: tuple[int, ...]) -> dict[int, LedgerRow]:
    """Return the ledger rows for ``versions``, tolerating a pre-073 schema.

    Before migration 073 runs there is no ``note`` column; selecting it would
    raise. Probe for it rather than swallowing the error, so a genuine failure
    (missing table, no permission) still propagates.
    """
    has_note = Probe(conn).column_exists("schema_migrations", "note")
    note_expr = "note" if has_note else "null::text"
    rows = conn.execute(
        f"select version, filename, {note_expr} from schema_migrations where version = any(%s)",
        (list(versions),),
    ).fetchall()
    return {int(v): LedgerRow(int(v), f, n) for v, f, n in rows}


@dataclass(frozen=True)
class Result:
    check: Check
    recorded: bool
    applied: bool
    note: str | None

    @property
    def marked_superseded(self) -> bool:
        return self.note is not None and SUPERSEDED_MARKER in self.note

    @property
    def diverges(self) -> bool:
        """The ledger disagrees with the catalog.

        A row explicitly annotated ``SUPERSEDED, NOT IN FORCE`` is a deliberate
        tombstone (see migration 074's entry for version 70): it claims the
        migration's objects are intentionally absent, so it agrees with the
        catalog precisely when the post-condition does *not* hold. If those
        objects ever reappear the tombstone becomes false and this reports it.
        """
        if self.marked_superseded:
            return self.applied
        return self.recorded != self.applied

    @property
    def verdict(self) -> str:
        if self.marked_superseded:
            return "ok (tombstone: SUPERSEDED, NOT IN FORCE)"
        if self.recorded and self.applied:
            return "ok (recorded, applied)"
        if not self.recorded and not self.applied:
            return "ok (unrecorded, not applied)"
        if self.applied and not self.recorded:
            return "DIVERGENT: applied in DB but absent from ledger"
        return "DIVERGENT: ledger claims applied but objects say otherwise"


def evaluate(conn: psycopg.Connection) -> list[Result]:
    probe = Probe(conn)
    ledger = read_ledger(conn, tuple(c.version for c in CHECKS))
    results: list[Result] = []
    for check in CHECKS:
        row = ledger.get(check.version)
        results.append(
            Result(
                check=check,
                recorded=row is not None,
                applied=check.applied_in_catalog(probe),
                note=row.note if row else None,
            )
        )
    return results


def report(results: list[Result], dsn: str, verbose: bool) -> int:
    print(f"Migration ledger vs catalog — {redact_dsn(dsn)}\n")
    width = max(len(r.check.filename) for r in results)
    for r in results:
        flag = "FAIL" if r.diverges else "ok  "
        print(f"  [{flag}] {r.check.version}  {r.check.filename:<{width}}  {r.verdict}")
        if verbose:
            print(f"           {r.check.summary}")
            print(f"           recorded={r.recorded} post_condition_holds={r.applied}")
            if r.note:
                print(f"           note: {r.note}")

    divergent = [r for r in results if r.diverges]
    print()
    if not divergent:
        print("Ledger agrees with the catalog for migrations 069-074.")
        return 0
    print(f"{len(divergent)} divergence(s):")
    for r in divergent:
        print(f"  - {r.check.version} {r.check.filename}: {r.verdict}")
    print("\nSee docs/ops/migration_ledger_reconciliation_2026-07-27.md.")
    print("Migrations 073 and 074 reconcile the known gaps; a human must run them.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="libpq DSN (default: $SCIX_DSN)")
    parser.add_argument("--verbose", action="store_true", help="show each probe's inputs")
    args = parser.parse_args(argv)

    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        results = evaluate(conn)
    return report(results, args.dsn, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
