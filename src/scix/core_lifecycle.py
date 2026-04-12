"""Curated entity core lifecycle (M3.5.2).

Manages ``curated_entity_core`` membership with a hard 10K cap and an
append-only ``core_promotion_log`` event stream.

Promotion at the cap auto-demotes the lowest-query-hit entry to make room.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import psycopg

from scix.db import DEFAULT_DSN, get_connection

logger = logging.getLogger(__name__)

# Hard cap per PRD §M3.5.2. Tests may monkeypatch this module attribute.
CORE_MAX: int = 10_000


def _resolve_dsn(dsn: str | None) -> str:
    if dsn is not None:
        return dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


def _open(conn: psycopg.Connection | None, dsn: str | None) -> tuple[psycopg.Connection, bool]:
    if conn is not None:
        return conn, False
    return get_connection(_resolve_dsn(dsn), autocommit=False), True


def _core_size(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM curated_entity_core")
        return int(cur.fetchone()[0])


def _lowest_hit_entity(conn: psycopg.Connection) -> Optional[int]:
    """Return the entity_id with the lowest query_hits_14d (ties broken by
    oldest promoted_at)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT entity_id
              FROM curated_entity_core
          ORDER BY query_hits_14d ASC, promoted_at ASC
             LIMIT 1
            """)
        row = cur.fetchone()
    return int(row[0]) if row else None


def _log_event(
    conn: psycopg.Connection,
    entity_id: int,
    action: str,
    query_hits_14d: int | None,
    reason: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO core_promotion_log (entity_id, action, query_hits_14d, reason)
            VALUES (%s, %s, %s, %s)
            """,
            (entity_id, action, query_hits_14d, reason),
        )


def promote(
    entity_id: int,
    *,
    query_hits_14d: int = 0,
    reason: str = "manual",
    conn: psycopg.Connection | None = None,
    dsn: str | None = None,
) -> None:
    """Add ``entity_id`` to the curated core.

    If the core is already at ``CORE_MAX``, auto-demotes the lowest-hit
    entry first. Both events are recorded in ``core_promotion_log``.
    Upserts (promoting an already-present entity refreshes its
    ``query_hits_14d``).
    """
    c, owned = _open(conn, dsn)
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM curated_entity_core WHERE entity_id = %s",
                (entity_id,),
            )
            already_present = cur.fetchone() is not None

        if not already_present and _core_size(c) >= CORE_MAX:
            victim = _lowest_hit_entity(c)
            if victim is not None and victim != entity_id:
                with c.cursor() as cur:
                    cur.execute(
                        "DELETE FROM curated_entity_core WHERE entity_id = %s",
                        (victim,),
                    )
                _log_event(c, victim, "demote", None, "auto_demote_cap")

        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO curated_entity_core (entity_id, query_hits_14d, promoted_at)
                VALUES (%s, %s, now())
                ON CONFLICT (entity_id) DO UPDATE
                  SET query_hits_14d = EXCLUDED.query_hits_14d,
                      promoted_at    = now()
                """,
                (entity_id, query_hits_14d),
            )
        _log_event(c, entity_id, "promote", query_hits_14d, reason)
        c.commit()
    except Exception:
        try:
            c.rollback()
        except Exception:
            pass
        raise
    finally:
        if owned:
            c.close()


def demote(
    entity_id: int,
    *,
    reason: str = "manual",
    conn: psycopg.Connection | None = None,
    dsn: str | None = None,
) -> None:
    """Remove ``entity_id`` from the curated core. No-op if absent (still
    logged as a demote event for auditability)."""
    c, owned = _open(conn, dsn)
    try:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM curated_entity_core WHERE entity_id = %s",
                (entity_id,),
            )
        _log_event(c, entity_id, "demote", None, reason)
        c.commit()
    except Exception:
        try:
            c.rollback()
        except Exception:
            pass
        raise
    finally:
        if owned:
            c.close()
