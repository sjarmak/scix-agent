"""Backfill filter fields into the scix_indus_v2_papers_s1 Qdrant collection.

Adds the seven indexed payload fields and four non-indexed metadata fields
defined in ADR-008 to the production collection, which was loaded with only
`bibcode` payload (qdrant_full_load.py).

Indexed fields (payload indexes created idempotently):
    year                  integer   papers.year
    doctype               keyword   papers.doctype
    arxiv_class           keyword   papers.arxiv_class (list)
    bibstem               keyword   papers.bibstem (list)
    community_semantic_coarse  integer  paper_metrics.community_semantic_coarse
    community_semantic_medium  integer  paper_metrics.community_semantic_medium
    is_retracted          bool      papers.retracted_at IS NOT NULL

Non-indexed metadata (no index, skip NULLs):
    title                 papers.title
    first_author          papers.first_author
    citation_count        papers.citation_count
    pagerank              paper_metrics.pagerank

The script streams papers in `--batch` chunks ordered by bibcode (stable,
resumable), sets payload via UUID5 point-ID lookup (keyed by
uuid5(NAMESPACE_URL, bibcode)), and respects `--limit` to cap the pilot slice.

Usage:
    # dry-run (no writes):
    scix-batch python scripts/backfill_qdrant_filter_fields.py --dry-run --limit 100

    # pilot: ≤100k points
    scix-batch python scripts/backfill_qdrant_filter_fields.py --limit 100000

    # full corpus (operator-gated — report throughput from pilot first):
    scix-batch python scripts/backfill_qdrant_filter_fields.py

Env:
    QDRANT_URL   default http://127.0.0.1:6633
    SCIX_DSN     Postgres DSN (falls back to scix.db.DEFAULT_DSN)

Idempotent: set_payload overwrites; re-running a batch applies the same
values again — safe.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from scix.db import DEFAULT_DSN


# Point IDs in scix_indus_v2_papers_s1 are UUID5s derived from bibcode,
# matching scripts/qdrant_full_load.py and scripts/qdrant_outbox_sync.py.
# Must stay byte-identical to both.
def _bibcode_to_point_id(bibcode: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode))


log = logging.getLogger("backfill_qdrant_filter_fields")

COLLECTION_DEFAULT = "scix_indus_v2_papers_s1"

# Indexed payload fields: (name, Qdrant schema type)
INDEXED_FIELDS: list[tuple[str, qm.PayloadSchemaType]] = [
    ("year", qm.PayloadSchemaType.INTEGER),
    ("doctype", qm.PayloadSchemaType.KEYWORD),
    ("arxiv_class", qm.PayloadSchemaType.KEYWORD),
    ("bibstem", qm.PayloadSchemaType.KEYWORD),
    ("community_semantic_coarse", qm.PayloadSchemaType.INTEGER),
    ("community_semantic_medium", qm.PayloadSchemaType.INTEGER),
    ("is_retracted", qm.PayloadSchemaType.BOOL),
]

# SQL for paginated batch fetch (stable ordering by bibcode)
_BATCH_SQL = """
    SELECT
        p.bibcode,
        p.year,
        p.doctype,
        p.arxiv_class,
        p.bibstem,
        p.title,
        p.first_author,
        p.citation_count,
        (p.retracted_at IS NOT NULL) AS is_retracted,
        pm.community_semantic_coarse,
        pm.community_semantic_medium,
        pm.pagerank
    FROM papers p
    LEFT JOIN paper_metrics pm USING (bibcode)
    WHERE p.bibcode > %s
    ORDER BY p.bibcode
    LIMIT %s
"""


def ensure_indexes(client: QdrantClient, collection: str, *, dry_run: bool) -> None:
    existing_schema = client.get_collection(collection).payload_schema or {}
    for field_name, schema_type in INDEXED_FIELDS:
        if field_name in existing_schema:
            log.info("payload index %s already exists on %s — skipped", field_name, collection)
            continue
        if dry_run:
            log.info("[dry-run] would create payload index %s (%s)", field_name, schema_type)
            continue
        client.create_payload_index(collection, field_name=field_name, field_schema=schema_type)
        log.info("created payload index %s (%s) on %s", field_name, schema_type, collection)


def _build_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Construct the full payload dict from a PG row.

    Omits fields whose source is NULL so set_payload does not overwrite
    existing non-NULL values with None (absence-semantics for is_retracted
    per ADR-008; also avoids noise on nullable metadata fields).
    """
    payload: dict[str, Any] = {}

    if row["year"] is not None:
        payload["year"] = int(row["year"])
    if row["doctype"] is not None:
        payload["doctype"] = row["doctype"]
    if row["arxiv_class"]:
        payload["arxiv_class"] = list(row["arxiv_class"])
    if row["bibstem"]:
        payload["bibstem"] = list(row["bibstem"])
    if row["community_semantic_coarse"] is not None:
        payload["community_semantic_coarse"] = int(row["community_semantic_coarse"])
    if row["community_semantic_medium"] is not None:
        payload["community_semantic_medium"] = int(row["community_semantic_medium"])
    # is_retracted: write True explicitly; omit False (absence = not known retracted)
    if row["is_retracted"]:
        payload["is_retracted"] = True

    # Non-indexed metadata
    if row["title"] is not None:
        payload["title"] = row["title"]
    if row["first_author"] is not None:
        payload["first_author"] = row["first_author"]
    if row["citation_count"] is not None:
        payload["citation_count"] = int(row["citation_count"])
    if row["pagerank"] is not None:
        payload["pagerank"] = float(row["pagerank"])

    return payload


def stream_pg_batches(
    conn: psycopg.Connection,
    *,
    batch: int,
    limit: int | None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of rows from Postgres, paginating via bibcode cursor."""
    cursor = ""
    total = 0
    while True:
        fetch = batch if limit is None else min(batch, limit - total)
        if fetch <= 0:
            break
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_BATCH_SQL, (cursor, fetch))
            rows = cur.fetchall()
        if not rows:
            break
        yield rows
        total += len(rows)
        cursor = rows[-1]["bibcode"]
        if len(rows) < fetch:
            break


def apply_batch(
    client: QdrantClient,
    collection: str,
    rows: list[dict[str, Any]],
    *,
    dry_run: bool,
    call_interval_ms: float = 5.0,
    samples: list[tuple[str, dict[str, Any]]] | None = None,
    sample_cap: int = 10,
) -> int:
    """Apply payload for one batch of rows.

    Each row gets its own set_payload call keyed by UUID5 point ID.
    ``call_interval_ms`` throttles individual calls within the batch to
    avoid bursting the Qdrant HTTP connection (connection reset at >~200/s
    fire-and-forget on the live 32.4M serving collection).

    ``samples`` accumulates up to ``sample_cap`` (bibcode, payload) pairs
    across the run for post-run verification via ``verify_sample``.

    Returns number of rows attempted.
    """
    if dry_run:
        # Mirror the live skip below: only rows with a non-empty payload would
        # produce a set_payload call, so count those — not every fetched row.
        return sum(1 for row in rows if _build_payload(row))

    written = 0
    for row in rows:
        payload = _build_payload(row)
        if not payload:
            continue
        client.set_payload(
            collection_name=collection,
            payload=payload,
            points=qm.PointIdsList(points=[_bibcode_to_point_id(row["bibcode"])]),
            wait=False,
        )
        written += 1
        if samples is not None and len(samples) < sample_cap:
            samples.append((row["bibcode"], payload))
        if call_interval_ms > 0:
            time.sleep(call_interval_ms / 1000.0)
    return written


def verify_sample(
    client: QdrantClient,
    collection: str,
    samples: list[tuple[str, dict[str, Any]]],
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 5.0,
) -> bool:
    """Confirm sampled payloads actually landed on their points.

    set_payload runs with wait=False: Qdrant acks the op into the WAL and
    applies it asynchronously. A stalled update pipeline acks every write
    and applies none (observed on the live collection 2026-06-12, bead
    nnim) — without this check the backfill reports success while writing
    nothing. Polls until every sampled point carries all written keys, or
    the timeout expires.
    """
    expected = dict(samples)
    # Pre-build id→bibcode reverse map so verify loop doesn't re-compute per-poll.
    id_to_bibcode = {_bibcode_to_point_id(b): b for b in expected}
    deadline = time.monotonic() + timeout_s
    while True:
        ids = list(id_to_bibcode)
        points = client.retrieve(collection, ids=ids, with_payload=True)
        # Key on point.id (always present) — not on payload bibcode, which may
        # be absent if the collection was loaded without bibcode in payload.
        got = {p.id: p.payload for p in points if p.payload}
        unapplied = {
            b
            for point_id, b in id_to_bibcode.items()
            if point_id not in got or any(got[point_id].get(k) != v for k, v in expected[b].items())
        }
        if not unapplied:
            log.info(
                "verification passed: all %d sampled points carry their payload", len(expected)
            )
            return True
        if time.monotonic() >= deadline:
            log.error(
                "verification FAILED: %d/%d sampled points missing payload after %.0fs "
                "(unapplied: %s) — Qdrant acked the writes but did not apply them; "
                "check the collection's update pipeline before re-running",
                len(unapplied),
                len(expected),
                timeout_s,
                sorted(unapplied)[:5],
            )
            return False
        time.sleep(poll_interval_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Backfill filter-field payloads into Qdrant")
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--batch", type=int, default=100, help="PG fetch + Qdrant write batch size")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total points processed (pilot mode; omit for full corpus)",
    )
    ap.add_argument(
        "--call-interval-ms",
        type=float,
        default=5.0,
        help="Sleep ms between individual set_payload calls within a batch (default: 5)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6633")
    client = QdrantClient(url=url, timeout=120)

    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        raise SystemExit(
            f"collection {args.collection!r} not found — available: {sorted(existing)}"
        )

    log.info(
        "starting backfill on %s (batch=%d, limit=%s, dry_run=%s)",
        args.collection,
        args.batch,
        args.limit,
        args.dry_run,
    )
    ensure_indexes(client, args.collection, dry_run=args.dry_run)

    dsn = os.environ.get("SCIX_DSN", DEFAULT_DSN)
    t0 = time.monotonic()
    total_attempted = 0
    batches = 0
    samples: list[tuple[str, dict[str, Any]]] = []

    with psycopg.connect(dsn) as conn:
        for rows in stream_pg_batches(conn, batch=args.batch, limit=args.limit):
            attempted = apply_batch(
                client,
                args.collection,
                rows,
                dry_run=args.dry_run,
                call_interval_ms=args.call_interval_ms,
                samples=samples,
            )
            total_attempted += attempted
            batches += 1
            if batches % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = total_attempted / elapsed if elapsed > 0 else 0
                log.info("progress: %d rows in %.1fs (%.0f rows/s)", total_attempted, elapsed, rate)
    elapsed = time.monotonic() - t0
    rate = total_attempted / elapsed if elapsed > 0 else 0
    mode = "dry-run" if args.dry_run else "written"
    log.info(
        "done — %d rows %s in %.1fs (%.0f rows/s)",
        total_attempted,
        mode,
        elapsed,
        rate,
    )
    if total_attempted > 0:
        projected_full = 32_383_535 / rate / 3600 if rate > 0 else float("inf")
        log.info(
            "throughput: %.0f rows/s → full-corpus (32.4M) projected %.1f h at this rate",
            rate,
            projected_full,
        )

    if not args.dry_run and samples:
        if not verify_sample(client, args.collection, samples):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
