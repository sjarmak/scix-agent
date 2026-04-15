#!/usr/bin/env python3
"""Seed bootstrap for query_log (xz4.1.18).

Classifies ``ambiguity_class='unique'`` entities by dispatching real
``keyword_search`` tool calls in-process and writing each result to
``query_log`` under a deterministic session_id derived from the manifest.
Entities whose canonical name returns zero corpus hits become pass1
candidates in curate_entity_core; entities that return >=1 hit become
pass3 candidates.

Design constraints (see bead xz4.1.18 PLAN-REVIEW note):
  * Exact canonical names only — pass1/pass3 SQL matches
    ``lower(trim(query)) = lower(canonical_name)`` literally, so
    misspellings/paraphrases contribute zero rows.
  * ``scix.query_log.log_query`` writes is_test=False directly — we
    never touch ``mcp_server._log_query`` or ``_is_test_session``.
  * Deterministic session_id + NOT EXISTS guard makes partial-failure
    retries idempotent.
  * Hard-fail if ``SCIX_TEST_DSN`` is set in env while targeting prod —
    that combination is almost always a developer footgun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

# Allow running as ``python scripts/seed_query_log.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn  # noqa: E402
from scix.mcp_server import _dispatch_tool, _extract_result_count  # noqa: E402
from scix.query_log import log_query  # noqa: E402

logger = logging.getLogger(__name__)

SEED_SESSION_PREFIX = "seed-bootstrap-v1-"
DEFAULT_PER_SOURCE_TARGET = 60
DEFAULT_SOURCES: tuple[str, ...] = (
    "vizier",
    "gcmd",
    "pwc",
    "ascl",
    "physh",
    "aas",
    "spase",
    "ssodnet",
    "ads_data",
)
DEFAULT_TOOL = "keyword_search"
DEFAULT_MAX_CALLS = 1000


@dataclass(frozen=True)
class SeedResult:
    session_id: str
    pass1_written: int
    pass3_written: int
    pass1_skipped: int
    pass3_skipped: int
    errors: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_session_id(manifest: dict[str, Any]) -> str:
    """Deterministic session_id derived from a stable JSON encoding of the manifest."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:12]
    return f"{SEED_SESSION_PREFIX}{digest}"


def assert_safe_target_dsn(target_dsn: str) -> None:
    """Refuse to run if SCIX_TEST_DSN is set while target is production.

    This catches the common footgun of inheriting a test DSN env var from
    a prior pytest session and then running the seed script against prod.
    It also prevents ambiguity about which DB is being mutated.

    An empty DSN also hard-fails: libpq would otherwise fall back to its
    own defaults (PGHOST/PGDATABASE/etc.) which can silently connect to a
    production host under an unexpected dbname.
    """
    if not target_dsn:
        sys.stderr.write("ERROR: empty target DSN — refuse to fall back to libpq defaults.\n")
        raise SystemExit(2)
    test_dsn_env = os.environ.get("SCIX_TEST_DSN")
    if test_dsn_env and is_production_dsn(target_dsn):
        sys.stderr.write(
            "ERROR: SCIX_TEST_DSN is set in env but --dsn points at production.\n"
            "Refusing to run — unset SCIX_TEST_DSN or point --dsn at the test DB.\n"
        )
        raise SystemExit(2)


def normalize_query(s: str) -> str:
    """NFC + strip + lower — matches the JOIN normalization in curate_entity_core."""
    return unicodedata.normalize("NFC", s).strip().lower()


def _load_candidates(
    conn: psycopg.Connection,
    sources: list[str],
    per_source_target: int,
) -> list[tuple[int, str, str]]:
    """Load up to *per_source_target* unique entities per source.

    Returns a flat list of ``(entity_id, canonical_name, source)`` tuples.
    Deterministic order within a source (canonical_name ASC) so the manifest
    + DB state alone determine which candidates get classified.
    """
    out: list[tuple[int, str, str]] = []
    with conn.cursor() as cur:
        for source in sources:
            cur.execute(
                """
                SELECT id, canonical_name, source
                  FROM entities
                 WHERE ambiguity_class = 'unique'
                   AND source = %s
                 ORDER BY canonical_name
                 LIMIT %s
                """,
                (source, per_source_target),
            )
            out.extend((int(r[0]), str(r[1]), str(r[2])) for r in cur.fetchall())
    return out


def _already_logged(conn: psycopg.Connection, session_id: str, query: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM query_log WHERE session_id = %s AND query = %s LIMIT 1",
            (session_id, query),
        )
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Seed run
# ---------------------------------------------------------------------------


def run_seed(
    conn: psycopg.Connection,
    *,
    manifest: dict[str, Any],
    dry_run: bool = False,
    max_calls: int = DEFAULT_MAX_CALLS,
) -> SeedResult:
    """Classify candidate entities and write query_log rows under the manifest's session_id.

    The dispatcher (``_dispatch_tool``) is called once per candidate to
    get a real ``result_count``. Rows already present for the session_id
    are skipped — so partial-failure retries converge without duplicates.

    ``dry_run=True`` classifies candidates but writes nothing.
    """
    t0 = time.monotonic()
    session_id = compute_session_id(manifest)
    sources = list(manifest.get("sources") or DEFAULT_SOURCES)
    per_source_target = int(manifest.get("per_source_target", DEFAULT_PER_SOURCE_TARGET))
    tool = str(manifest.get("tool", DEFAULT_TOOL))

    candidates = _load_candidates(conn, sources, per_source_target)
    if len(candidates) > max_calls:
        logger.warning(
            "candidate pool (%d) exceeds --max-calls (%d); truncating",
            len(candidates),
            max_calls,
        )
        candidates = candidates[:max_calls]

    logger.info(
        "session_id=%s sources=%d per_source=%d candidates=%d dry_run=%s",
        session_id,
        len(sources),
        per_source_target,
        len(candidates),
        dry_run,
    )

    pass1_written = 0
    pass3_written = 0
    pass1_skipped = 0
    pass3_skipped = 0
    errors = 0

    for entity_id, canonical, source in candidates:
        normalized = normalize_query(canonical)
        if not normalized:
            continue
        try:
            result_json = _dispatch_tool(conn, tool, {"terms": canonical, "limit": 1})
        except Exception as exc:
            logger.warning(
                "dispatch failed: entity_id=%d canonical=%r source=%s: %s",
                entity_id,
                canonical,
                source,
                exc,
            )
            errors += 1
            continue

        result_count = _extract_result_count(result_json)
        is_pass1 = result_count == 0

        if dry_run:
            continue

        if _already_logged(conn, session_id, normalized):
            if is_pass1:
                pass1_skipped += 1
            else:
                pass3_skipped += 1
            continue

        # log_query commits per row — mid-run crashes preserve progress
        # and the NOT EXISTS guard handles resume on next invocation.
        try:
            log_query(
                tool=tool,
                query=normalized,
                result_count=result_count,
                session_id=session_id,
                is_test=False,
                conn=conn,
            )
        except Exception as exc:
            logger.warning(
                "log_query failed: entity_id=%d canonical=%r: %s",
                entity_id,
                canonical,
                exc,
            )
            errors += 1
            continue

        if is_pass1:
            pass1_written += 1
        else:
            pass3_written += 1

    elapsed = time.monotonic() - t0
    return SeedResult(
        session_id=session_id,
        pass1_written=pass1_written,
        pass3_written=pass3_written,
        pass1_skipped=pass1_skipped,
        pass3_skipped=pass3_skipped,
        errors=errors,
        elapsed_s=elapsed,
    )


def run_rollback(conn: psycopg.Connection, *, session_id: str) -> int:
    """Delete all query_log rows tagged with ``session_id``. Returns row count.

    Refuses to delete session_ids that don't start with ``SEED_SESSION_PREFIX`` —
    this prevents the rollback path from being used as a generic bulk-delete
    against organic traffic rows.
    """
    if not session_id.startswith(SEED_SESSION_PREFIX):
        raise ValueError(
            f"refusing to rollback session_id={session_id!r}: "
            f"must begin with {SEED_SESSION_PREFIX!r}"
        )
    with conn.cursor() as cur:
        cur.execute("DELETE FROM query_log WHERE session_id = %s", (session_id,))
        deleted = cur.rowcount
    conn.commit()
    return int(deleted)


def compute_bind_rate(conn: psycopg.Connection, *, session_id: str) -> float:
    """Fraction of distinct query strings that bind to an entity canonical name.

    Returns 0.0 if the session has no rows.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(DISTINCT q.query)
              FROM query_log q
             WHERE q.session_id = %s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        distinct = int(row[0]) if row else 0
        if distinct == 0:
            return 0.0
        cur.execute(
            """
            SELECT count(DISTINCT q.query)
              FROM query_log q
             WHERE q.session_id = %s
               AND EXISTS (
                    SELECT 1 FROM entities e
                     WHERE lower(e.canonical_name) = lower(trim(q.query))
               )
            """,
            (session_id,),
        )
        row = cur.fetchone()
        bound = int(row[0]) if row else 0
    return bound / distinct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "tool": DEFAULT_TOOL,
        "per_source_target": DEFAULT_PER_SOURCE_TARGET,
        "sources": list(DEFAULT_SOURCES),
    }


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _default_manifest()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="Target DSN (default: scix.db DEFAULT_DSN)")
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Path to JSON manifest (defaults to built-in astronomy seed plan)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Classify but do not write")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Delete rows tagged with this manifest's session_id",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help="Hard cap on dispatch calls (cost budget)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dsn = args.dsn or DEFAULT_DSN
    assert_safe_target_dsn(dsn)
    manifest = _load_manifest(args.plan)

    conn = get_connection(dsn)
    try:
        if args.rollback:
            session_id = compute_session_id(manifest)
            deleted = run_rollback(conn, session_id=session_id)
            logger.info("Rolled back %d rows for session_id=%s", deleted, session_id)
            return 0

        result = run_seed(conn, manifest=manifest, dry_run=args.dry_run, max_calls=args.max_calls)
        logger.info(
            "session_id=%s pass1=%d pass3=%d skipped_p1=%d skipped_p3=%d errors=%d elapsed=%.1fs",
            result.session_id,
            result.pass1_written,
            result.pass3_written,
            result.pass1_skipped,
            result.pass3_skipped,
            result.errors,
            result.elapsed_s,
        )

        if not args.dry_run:
            bind_rate = compute_bind_rate(conn, session_id=result.session_id)
            logger.info("bind_rate=%.3f", bind_rate)
            if bind_rate < 0.8:
                logger.warning("bind_rate %.3f below 0.80 — curate may skip rows", bind_rate)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
