"""Researcher feedback loop (PRD §S5).

Thin write-only API for recording suspected incorrect document->entity
links. Writes go to ``entity_link_disputes`` (created in migration 037).

This table is *not* scanned by the M13 AST resolver lint — the lint only
guards writes against ``document_entities`` and
``document_entities_jit_cache``, and reads against
``document_entities_canonical``. ``entity_link_disputes`` is an
auxiliary feedback table and can be written directly.

Consumers: offline audit jobs and the ``entity_audit`` CLI. Not read on
the hot path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import psycopg

from scix.db import DEFAULT_DSN, get_connection

logger = logging.getLogger(__name__)


def _resolve_dsn(dsn: Optional[str]) -> str:
    """Pick an explicit DSN, otherwise SCIX_TEST_DSN, otherwise the default.

    Mirrors ``scix.query_log._resolve_dsn`` so destructive tests that set
    ``SCIX_TEST_DSN`` automatically route here.
    """
    if dsn is not None:
        return dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


def report_incorrect_link(
    bibcode: str,
    entity_id: int,
    reason: str,
    tier: Optional[int] = None,
    *,
    conn: Optional[psycopg.Connection] = None,
    dsn: Optional[str] = None,
) -> None:
    """Record a researcher's report that a document->entity link is wrong.

    Parameters
    ----------
    bibcode
        Paper whose link is disputed.
    entity_id
        Entity on the suspect side of the link.
    reason
        Free-text explanation. Required — empty reasons are dropped.
    tier
        Optional tier the disputed link belongs to (matches
        ``document_entities.tier``).
    conn
        Optional open psycopg connection. If provided, the caller owns
        commit/rollback. If omitted, this function opens and closes its
        own connection.
    dsn
        Optional explicit DSN override. Ignored if ``conn`` is provided.
    """
    if not reason or not reason.strip():
        logger.warning("report_incorrect_link called with empty reason; dropping")
        return

    sql = (
        "INSERT INTO entity_link_disputes (bibcode, entity_id, reason, tier) "
        "VALUES (%s, %s, %s, %s)"
    )
    params = (bibcode, entity_id, reason.strip(), tier)

    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        return

    resolved = _resolve_dsn(dsn)
    with get_connection(resolved) as owned:
        with owned.cursor() as cur:
            cur.execute(sql, params)
        owned.commit()
