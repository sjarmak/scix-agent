#!/usr/bin/env python3
"""Drain the PG→Qdrant embedding outbox and apply writes to Qdrant.

Companion to ``migrations/070_embedding_outbox.sql``. The migration installs a
trigger that enqueues one ``embedding_outbox`` row per ``paper_embeddings``
write; this worker drains that queue into the serving Qdrant collection
(``scix_indus_v2_papers_s1``, ADR-013) so the dense lane never goes stale after
a daily embed run.

Delivery model
--------------
Per ``(model_name → collection)`` route, the worker claims a batch of outbox
rows with ``FOR UPDATE SKIP LOCKED``, upserts the current vectors to Qdrant,
then deletes the claimed rows — all in one transaction, and only after the
Qdrant write succeeds. That gives at-least-once delivery; the uuid5(bibcode)
point id makes the upsert idempotent, so a retried batch is harmless. If Qdrant
fails the transaction rolls back and the rows stay queued for the next run.

This is run as ``daily_sync.sh`` step 7 — a batch step, not a long-lived
service. So on Qdrant failure it fails fast (after a few short retries) rather
than blocking the pipeline with the PRD's "back off up to 1h" loop; the daily
cadence *is* the retry, and the lag metric surfaces a stuck queue.

Multi-collection routing (PRD P1)
---------------------------------
``ROUTING`` maps an embedding ``model_name`` to its Qdrant collection. v1 wires
only INDUS. When the sections/chunks collection lands it gets one more entry
here and the same drain loop covers it — the trigger already enqueues every
model, and unmapped models are reported (not silently dropped) by the lag
metric. Paper-level (``paper_embeddings``) and section-level
(``section_embeddings``) live in different source tables, so a sections lane
would add its own trigger + a route here, not reuse this table.

Backfill
--------
The trigger only captures writes that happen *after* it is installed. Papers
embedded before that (e.g. the 1,266-paper 2026-06-11 batch) are reconciled
with ``--backfill-since <date>``, which enqueues ``BACKFILL`` rows for INDUS
embeddings whose paper ``entry_date >= date`` (``entry_date`` is exactly the
field the daily harvest selects on, so it cleanly bounds "what arrived since").

Usage
-----
    # daily drain (daily_sync.sh step 7)
    python scripts/qdrant_outbox_sync.py --allow-prod

    # one-time reconcile of the pre-trigger gap
    python scripts/qdrant_outbox_sync.py --allow-prod --backfill-since 2026-06-11
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger("qdrant_outbox_sync")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Embedding model_name → Qdrant collection. The single source of truth for
# routing; extend here (plus a source-table trigger) when a new lane ships.
ROUTING: dict[str, str] = {"indus": "scix_indus_v2_papers_s1"}

# REST only: qdrant-client's gRPC path fails to deserialize qdrant 1.18.2
# responses on this stack (bead pkcd; mirrored in src/scix/search.py).
DEFAULT_QDRANT_URL = "http://127.0.0.1:6633"

DEFAULT_BATCH_SIZE = 1024

# Network-op retry for transient Qdrant blips. Bounded and short — a hard
# Qdrant outage should fail the daily step, not stall it (see module docstring).
QDRANT_RETRIES = 3
QDRANT_RETRY_BACKOFF_S = 2.0

# Lag is log-only for now (single operator, no pager — PRD OQ7). These mirror
# the PRD's pager thresholds so the warning lines are already calibrated when a
# pager is wired up later.
LAG_WARN_AGE_S = 3600
LAG_WARN_ROWS = 100_000


def point_id(bibcode: str) -> str:
    """Qdrant point id for a bibcode — uuid5, identical to the bulk load.

    Must match ``scripts/qdrant_full_load.py`` and the payload contract in
    ``src/scix/search.py`` so a forward write upserts the *same* point the
    bulk load created (idempotent, no duplicates).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DrainStats:
    """Outcome of draining one model's queue."""

    model_name: str
    collection: str
    upserted: int = 0
    deleted: int = 0
    missing_vector: int = 0
    batches: int = 0


@dataclass
class ClaimedRow:
    id: int
    bibcode: str
    op: str
    enqueued_at: Any


@dataclass
class _Resolved:
    """Per-bibcode resolution of a claimed batch (latest op wins)."""

    upsert_bibcodes: list[str] = field(default_factory=list)
    delete_bibcodes: list[str] = field(default_factory=list)
    claimed_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Qdrant helpers (the client is injected so tests run without a live Qdrant)
# ---------------------------------------------------------------------------


def make_qdrant_client(url: str, timeout: float = 30.0):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, prefer_grpc=False, timeout=timeout)


def _with_retry(fn, *, what: str):
    """Call ``fn`` with a few short retries on transient Qdrant errors."""
    last: Exception | None = None
    for attempt in range(1, QDRANT_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry then re-raise
            last = exc
            if attempt < QDRANT_RETRIES:
                logger.warning(
                    "Qdrant %s failed (attempt %d/%d): %s — retrying",
                    what, attempt, QDRANT_RETRIES, exc,
                )
                time.sleep(QDRANT_RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Qdrant {what} failed after {QDRANT_RETRIES} attempts") from last


def _apply_to_qdrant(
    client: Any,
    collection: str,
    *,
    points: list[Any],
    delete_ids: list[str],
) -> None:
    """Upsert ``points`` then delete ``delete_ids`` in ``collection``."""
    from qdrant_client import models as qm

    if points:
        _with_retry(
            lambda: client.upsert(collection_name=collection, points=points, wait=True),
            what="upsert",
        )
    if delete_ids:
        _with_retry(
            lambda: client.delete(
                collection_name=collection,
                points_selector=qm.PointIdsList(points=delete_ids),
                wait=True,
            ),
            what="delete",
        )


# ---------------------------------------------------------------------------
# Drain logic
# ---------------------------------------------------------------------------


def _claim_batch(conn: psycopg.Connection, model_name: str, batch_size: int) -> list[ClaimedRow]:
    """Claim up to ``batch_size`` oldest rows for ``model_name``.

    ``FOR UPDATE SKIP LOCKED`` holds the rows for the life of the transaction
    so the post-upsert DELETE removes exactly what was processed and concurrent
    workers never double-process. The caller must NOT commit until after the
    Qdrant write succeeds.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, bibcode, op, enqueued_at
            FROM embedding_outbox
            WHERE model_name = %s
            ORDER BY enqueued_at, id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (model_name, batch_size),
        )
        return [ClaimedRow(**r) for r in cur.fetchall()]


def _resolve_batch(rows: list[ClaimedRow]) -> _Resolved:
    """Collapse claimed rows to one decision per bibcode (latest op wins).

    Multiple rows can exist for one bibcode (it was written several times
    before draining). Only the newest matters, and a trailing DELETE must not
    be undone by an earlier UPDATE in the same batch.
    """
    rows_sorted = sorted(rows, key=lambda r: (r.enqueued_at, r.id))
    latest_op: dict[str, str] = {}
    for r in rows_sorted:
        latest_op[r.bibcode] = r.op
    resolved = _Resolved(claimed_ids=[r.id for r in rows])
    for bibcode, op in latest_op.items():
        if op == "DELETE":
            resolved.delete_bibcodes.append(bibcode)
        else:
            resolved.upsert_bibcodes.append(bibcode)
    return resolved


def _fetch_vectors(
    conn: psycopg.Connection, model_name: str, bibcodes: list[str]
) -> dict[str, list[float]]:
    """Return {bibcode: vector} for current INDUS embeddings of ``bibcodes``."""
    if not bibcodes:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, embedding
            FROM paper_embeddings
            WHERE model_name = %s AND bibcode = ANY(%s) AND embedding IS NOT NULL
            """,
            (model_name, bibcodes),
        )
        return {bibcode: vec.tolist() for bibcode, vec in cur.fetchall()}


def drain_model(
    conn: psycopg.Connection,
    client: Any,
    *,
    model_name: str,
    collection: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DrainStats:
    """Drain one model's queue into its Qdrant collection until empty.

    Each batch is claim → resolve → fetch vectors → upsert/delete to Qdrant →
    delete claimed outbox rows → commit. The commit happens only after Qdrant
    confirms (wait=True), so a crash anywhere before it leaves the rows queued.
    """
    from qdrant_client import models as qm

    stats = DrainStats(model_name=model_name, collection=collection)
    while True:
        rows = _claim_batch(conn, model_name, batch_size)
        if not rows:
            conn.rollback()  # release the (empty) snapshot
            break
        try:
            resolved = _resolve_batch(rows)
            vectors = _fetch_vectors(conn, model_name, resolved.upsert_bibcodes)

            points = []
            missing: list[str] = []
            for bibcode in resolved.upsert_bibcodes:
                vec = vectors.get(bibcode)
                if vec is None:
                    # Source row vanished between enqueue and drain; a DELETE
                    # enqueue (if any) removes the point. Drop the stale claim.
                    missing.append(bibcode)
                    continue
                points.append(
                    qm.PointStruct(
                        id=point_id(bibcode),
                        vector=vec,
                        payload={"bibcode": bibcode},
                    )
                )
            delete_ids = [point_id(b) for b in resolved.delete_bibcodes]

            _apply_to_qdrant(client, collection, points=points, delete_ids=delete_ids)

            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM embedding_outbox WHERE id = ANY(%s)",
                    (resolved.claimed_ids,),
                )
            conn.commit()

            stats.upserted += len(points)
            stats.deleted += len(delete_ids)
            stats.missing_vector += len(missing)
            stats.batches += 1
            if missing:
                logger.warning(
                    "%s: %d claimed upsert(s) had no current vector — outbox "
                    "rows dropped (source likely deleted): %s",
                    model_name, len(missing), ", ".join(sorted(missing)[:20]),
                )
        except Exception:
            conn.rollback()
            raise
    return stats


# ---------------------------------------------------------------------------
# Backfill + lag metric
# ---------------------------------------------------------------------------


def enqueue_backfill(conn: psycopg.Connection, model_name: str, since: str) -> int:
    """Enqueue BACKFILL rows for ``model_name`` embeddings since ``entry_date``.

    Reconciles the pre-trigger gap. ``entry_date`` is the ADS index date the
    daily harvest selects on (ISO text — lexical ``>=`` is correct ordering).
    Skips bibcodes already queued so a re-run doesn't bloat the table.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embedding_outbox (bibcode, model_name, op)
            SELECT pe.bibcode, pe.model_name, 'BACKFILL'
            FROM paper_embeddings pe
            JOIN papers p ON p.bibcode = pe.bibcode
            WHERE pe.model_name = %s
              AND pe.embedding IS NOT NULL
              AND p.entry_date >= %s
              AND NOT EXISTS (
                  SELECT 1 FROM embedding_outbox o
                  WHERE o.bibcode = pe.bibcode AND o.model_name = pe.model_name
              )
            """,
            (model_name, since),
        )
        n = cur.rowcount
    conn.commit()
    return n


def report_lag(conn: psycopg.Connection) -> list[tuple[str, int, float]]:
    """Return and log per-model (model_name, rows, oldest_age_seconds).

    The lag metric the PRD wants paged on. Log-only for now; warns past the
    pager thresholds. Rows for a model with no route are surfaced too — they
    would otherwise accumulate invisibly.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_name, count(*) AS rows,
                   COALESCE(EXTRACT(EPOCH FROM now() - min(enqueued_at)), 0) AS age_s
            FROM embedding_outbox
            GROUP BY model_name
            ORDER BY model_name
            """
        )
        result = [(m, int(rows), float(age)) for m, rows, age in cur.fetchall()]

    for model_name, rows, age_s in result:
        unrouted = "" if model_name in ROUTING else " [NO ROUTE]"
        line = "outbox lag %s%s: rows=%d oldest=%.0fs" % (model_name, unrouted, rows, age_s)
        if rows > LAG_WARN_ROWS or age_s > LAG_WARN_AGE_S or unrouted:
            logger.warning("%s (threshold rows>%d or age>%ds)", line, LAG_WARN_ROWS, LAG_WARN_AGE_S)
        else:
            logger.info(line)
    if not result:
        logger.info("outbox lag: empty")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    conn: psycopg.Connection,
    client: Any,
    *,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    backfill_since: str | None = None,
) -> list[DrainStats]:
    """Drain every routed model (or just ``model``), then report lag."""
    models = [model] if model else list(ROUTING)
    all_stats: list[DrainStats] = []
    for m in models:
        collection = ROUTING.get(m)
        if collection is None:
            logger.error("no Qdrant route for model %r — skipping", m)
            continue
        if backfill_since:
            n = enqueue_backfill(conn, m, backfill_since)
            logger.info("backfill: enqueued %d %s row(s) since entry_date>=%s", n, m, backfill_since)
        stats = drain_model(conn, client, model_name=m, collection=collection, batch_size=batch_size)
        logger.info(
            "drained %s → %s: upserted=%d deleted=%d missing_vector=%d batches=%d",
            stats.model_name, stats.collection, stats.upserted,
            stats.deleted, stats.missing_vector, stats.batches,
        )
        all_stats.append(stats)
    report_lag(conn)
    return all_stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help=f"Qdrant URL (default: $QDRANT_URL or {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Drain only this model_name (default: all routed models)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--backfill-since",
        default=None,
        metavar="YYYY-MM-DD",
        help="One-time reconcile: enqueue embeddings whose paper entry_date >= this",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against the production DSN",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(args.dsn),
        )
        return 2

    url = args.qdrant_url or os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL
    client = make_qdrant_client(url)

    t0 = time.monotonic()
    # Non-autocommit: drain_model holds the claim→upsert→delete in one txn.
    with psycopg.connect(args.dsn) as conn:
        register_vector(conn)
        try:
            run(
                conn,
                client,
                model=args.model,
                batch_size=args.batch_size,
                backfill_since=args.backfill_since,
            )
        except Exception as exc:  # noqa: BLE001 — surface, fail the daily step
            logger.error("outbox sync failed: %s", exc)
            return 1
    logger.info("outbox sync done in %.1fs", time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
