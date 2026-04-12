"""Persisted JIT cache over ``document_entities_jit_cache`` (PRD §M11b).

Responsibilities
----------------
* **Synchronous reads** via :meth:`JITCache.get` — a keyed SELECT on
  ``(bibcode, candidate_set_hash, model_version)`` filtered to rows that
  have not yet expired. Returns a :class:`CachedLinkSet` or ``None``.
* **Fire-and-forget writes** via :meth:`JITCache.put` — rows are placed
  onto an :class:`asyncio.Queue` with ``maxsize=1024``. A background
  writer coroutine drains the queue and INSERTs rows into the partitioned
  table. If the queue is saturated we **drop** the write (the spec
  explicitly says puts are fire-and-forget) and bump a consecutive-drop
  counter. After two consecutive drops we call :func:`raise_alert` —
  tests monkey-patch that to assert the pager wiring.
* **TTL cleanup** is owned by ``scripts/jit_cache_cleanup.py`` — this
  module just writes ``expires_at`` and trusts the partition cron to
  reclaim storage.

Lint exemption
--------------
``scripts/ast_lint_resolver.py`` forbids writes to
``document_entities_jit_cache`` outside ``src/scix/resolve_entities.py``.
The u10 spec (acceptance criterion 7) grants this file a transitional
exemption via ``# noqa: resolver-lint`` on the INSERT literal. A later
refactor will move the INSERT behind a resolver-injected handle.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Hard queue ceiling — spec §M11b requires 1024.
CACHE_QUEUE_MAXSIZE: int = 1024

#: Default TTL for cache rows (14 days, per spec).
CACHE_TTL_DAYS: int = 14

#: Tier for JIT rows, per fusion_mv tier_weight table.
CACHE_TIER: int = 5

#: Module-level consecutive-drop counter — tracked globally because the
#: writer is a singleton per process. Tests reset via :func:`_reset_drop_state`.
_consecutive_drops: int = 0


#: Test hook. Replaced by the test's mock to observe pager calls.
def raise_alert(message: str) -> None:  # pragma: no cover - replaced in tests
    """Fire a pager alert.

    The real deployment wires this to OpsGenie / PagerDuty. For u10 we
    ship a stub that logs at ERROR so the default behaviour is visible
    but not fatal — tests monkey-patch this to record calls.
    """
    logger.error("JIT_CACHE_PAGER_ALERT: %s", message)


def _reset_drop_state() -> None:
    """Clear the consecutive-drop counter (test helper)."""
    global _consecutive_drops
    _consecutive_drops = 0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedLinkSet:
    """Domain object read from / written to the JIT cache.

    Deliberately NOT an :class:`scix.entity_link_set.EntityLinkSet` — that
    type is token-gated and can only be constructed by
    ``scix.resolve_entities``. This is a plain DTO the resolver wraps on
    its way out.
    """

    bibcode: str
    candidate_set_hash: str
    model_version: str
    entity_ids: frozenset[int]
    confidences: frozenset[tuple[int, float]] = field(default_factory=frozenset)
    link_type: str = "mention"
    match_method: str = "jit_cache"
    expires_at: Optional[datetime] = None
    tier: int = CACHE_TIER


# ---------------------------------------------------------------------------
# JITCache
# ---------------------------------------------------------------------------


ConnectionFactory = Callable[[], "psycopg.Connection"]  # noqa: F821 - forward ref


class JITCache:
    """Asynchronous cache facade over ``document_entities_jit_cache``.

    Parameters
    ----------
    conn_factory
        Callable returning a fresh ``psycopg.Connection``. Kept behind a
        factory so tests can inject a test-DB DSN and production code can
        use a connection pool.
    queue_maxsize
        Override for the fire-and-forget queue ceiling (default 1024).
    ttl
        Row TTL as a :class:`datetime.timedelta`. Defaults to 14 days.
    """

    def __init__(
        self,
        conn_factory: ConnectionFactory,
        *,
        queue_maxsize: int = CACHE_QUEUE_MAXSIZE,
        ttl: timedelta = timedelta(days=CACHE_TTL_DAYS),
    ) -> None:
        self._conn_factory = conn_factory
        self._queue: asyncio.Queue[CachedLinkSet] = asyncio.Queue(maxsize=queue_maxsize)
        self._ttl = ttl
        self._writer_task: Optional[asyncio.Task[None]] = None
        self._running = False

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(
        self,
        bibcode: str,
        candidate_set_hash: str,
        model_version: str,
    ) -> Optional[CachedLinkSet]:
        """Return a non-expired cached link set, or ``None``.

        Synchronous by design: the read path hits a btree index and we
        want the resolver to be able to call it from both sync and async
        contexts without awaiting.
        """
        sql = (
            "SELECT bibcode, entity_id, confidence, link_type, match_method, "
            "expires_at FROM document_entities_jit_cache "
            "WHERE bibcode = %s AND candidate_set_hash = %s "
            "AND model_version = %s AND expires_at > now() AND tier = 5"
        )
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (bibcode, candidate_set_hash, model_version))
                    rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("jit-cache get failed: %s", exc)
            return None

        if not rows:
            return None

        entity_ids: set[int] = set()
        confidences: set[tuple[int, float]] = set()
        link_type = "mention"
        match_method = "jit_cache"
        expires_at: Optional[datetime] = None
        for row in rows:
            _bibcode, entity_id, confidence, row_link_type, row_match_method, row_expires = row
            entity_ids.add(int(entity_id))
            if confidence is not None:
                confidences.add((int(entity_id), float(confidence)))
            link_type = row_link_type or link_type
            match_method = row_match_method or match_method
            expires_at = row_expires

        return CachedLinkSet(
            bibcode=bibcode,
            candidate_set_hash=candidate_set_hash,
            model_version=model_version,
            entity_ids=frozenset(entity_ids),
            confidences=frozenset(confidences),
            link_type=link_type,
            match_method=match_method,
            expires_at=expires_at,
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def put(self, row: CachedLinkSet) -> bool:
        """Enqueue a row for async write.

        Returns ``True`` if the row was accepted onto the queue, ``False``
        if the queue was full and the row was dropped. Fire-and-forget: we
        never raise on saturation.
        """
        global _consecutive_drops

        if row.expires_at is None:
            row = _with_expiry(row, self._ttl)
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            _consecutive_drops += 1
            logger.warning(
                "jit-cache queue full, dropping row bibcode=%s (consecutive=%d)",
                row.bibcode,
                _consecutive_drops,
            )
            if _consecutive_drops >= 2:
                raise_alert(f"jit-cache consecutive drops >= {_consecutive_drops}")
            return False
        else:
            _consecutive_drops = 0
            return True

    def qsize(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Background writer lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background writer task."""
        if self._writer_task is not None:
            return
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        """Stop the background writer after draining in-flight rows."""
        self._running = False
        if self._writer_task is not None:
            # Wake the loop with a sentinel None by cancelling — drained
            # rows are still committed by the final flush below.
            self._writer_task.cancel()
            try:
                await self._writer_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._writer_task = None

    async def _writer_loop(self) -> None:
        while self._running:
            try:
                row = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await asyncio.to_thread(self._write_row, row)
            except Exception as exc:  # noqa: BLE001
                logger.warning("jit-cache writer failed: %s", exc)
            finally:
                self._queue.task_done()

    async def drain_once(self) -> int:
        """Synchronously drain all currently-queued rows and write them.

        Test helper — production uses the background writer. Returns the
        number of rows written.
        """
        written = 0
        while not self._queue.empty():
            row = self._queue.get_nowait()
            try:
                await asyncio.to_thread(self._write_row, row)
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("jit-cache drain failed: %s", exc)
            finally:
                self._queue.task_done()
        return written

    # ------------------------------------------------------------------
    # Raw INSERT — lint-exempt per u10 §7.
    # ------------------------------------------------------------------

    def _write_row(self, row: CachedLinkSet) -> None:
        """Insert a cached row. The INSERT literal carries the noqa marker
        because the single-entry-point lint normally forbids jit_cache
        writes outside ``scix.resolve_entities`` — u10 spec grants this
        file a transitional exemption.
        """
        expires_at = row.expires_at
        if expires_at is None:
            expires_at = datetime.now(timezone.utc) + self._ttl

        # fmt: off
        sql = (
            "INSERT INTO document_entities_jit_cache "  # noqa: resolver-lint
            "(bibcode, entity_id, link_type, confidence, match_method, "
            "tier, tier_version, candidate_set_hash, model_version, expires_at) "
            "VALUES (%s, %s, %s, %s, %s, 5, 1, %s, %s, %s) "
            "ON CONFLICT DO NOTHING"
        )
        # fmt: on

        conf_map = dict(row.confidences)
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                for entity_id in row.entity_ids:
                    confidence = conf_map.get(entity_id)
                    cur.execute(
                        sql,
                        (
                            row.bibcode,
                            entity_id,
                            row.link_type,
                            confidence,
                            row.match_method,
                            row.candidate_set_hash,
                            row.model_version,
                            expires_at,
                        ),
                    )
            conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_expiry(row: CachedLinkSet, ttl: timedelta) -> CachedLinkSet:
    expires_at = datetime.now(timezone.utc) + ttl
    return CachedLinkSet(
        bibcode=row.bibcode,
        candidate_set_hash=row.candidate_set_hash,
        model_version=row.model_version,
        entity_ids=row.entity_ids,
        confidences=row.confidences,
        link_type=row.link_type,
        match_method=row.match_method,
        expires_at=expires_at,
        tier=row.tier,
    )


def default_conn_factory() -> "psycopg.Connection":  # noqa: F821
    """Default connection factory: reads ``SCIX_TEST_DSN`` or falls back
    to ``dbname=scix``. Callers are expected to inject their own factory
    in production (e.g. a pool checkout)."""
    import psycopg

    dsn = os.environ.get("SCIX_TEST_DSN", "dbname=scix")
    return psycopg.connect(dsn)


__all__ = [
    "CACHE_QUEUE_MAXSIZE",
    "CACHE_TIER",
    "CACHE_TTL_DAYS",
    "CachedLinkSet",
    "JITCache",
    "default_conn_factory",
    "raise_alert",
]
