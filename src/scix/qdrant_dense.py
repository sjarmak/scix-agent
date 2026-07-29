"""Direct-to-Qdrant writes for the INDUS dense lane.

The serving collection ``scix_indus_v2_papers_s1`` (ADR-013) is the dense lane
``search``/``vector_search`` reads from. New papers are embedded and upserted
here directly by the daily pipeline (``scix.embed``); there is no PostgreSQL
staging table (``paper_embeddings`` was retired — see ADR-015 / bead s7cy).

The point-id and payload contract MUST match the original bulk load, or new
points land under ids the existing 32.4M points can't match:
  * id      = ``uuid5(NAMESPACE_URL, bibcode)`` — string
  * vector  = the 768-d INDUS mean-pooled embedding (single unnamed vector;
              the collection stores float16, Qdrant down-converts on upsert)
  * payload = ``{"bibcode": bibcode}``
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

# ADR-013 serving collection for the INDUS dense lane.
INDUS_COLLECTION = "scix_indus_v2_papers_s1"


def point_id(bibcode: str) -> str:
    """Qdrant point id for a bibcode — uuid5, identical to the bulk load."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode))


def dense_client(timeout: float = 30.0) -> Any:
    """Build a QdrantClient from ``QDRANT_URL``; raise if the lane is unconfigured."""
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError(
            "QDRANT_URL is unset; the INDUS dense lane requires Qdrant (ADR-013). "
            "Set QDRANT_URL before embedding."
        )
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=timeout)


def build_points(vectors: Mapping[str, list[float]]) -> list[Any]:
    """Build PointStructs honouring the serving-collection id/payload contract."""
    from qdrant_client import models as qm

    return [
        qm.PointStruct(id=point_id(bibcode), vector=list(vec), payload={"bibcode": bibcode})
        for bibcode, vec in vectors.items()
    ]


def upsert_dense(client: Any, collection: str, vectors: Mapping[str, list[float]]) -> int:
    """Upsert ``{bibcode: vector}`` into ``collection``, waiting for durability.

    ``wait=True`` is load-bearing: callers mark a paper as synced (and commit)
    only after this returns, so a crash before the durable write leaves the
    paper unsynced and it is re-embedded next run (at-least-once, idempotent).

    NOTE: ``upsert`` replaces the whole point (vector + payload), so the payload
    is reset to ``{"bibcode": ...}``. This matches the current collection (every
    point carries only ``bibcode``). If ``scripts/backfill_qdrant_filter_fields.py``
    is ever run to enrich points with year/doctype/etc., this write path must
    become payload-preserving (merge) or re-run the backfill for re-upserted
    bibcodes — otherwise a recovery re-upsert would strip the enrichment.
    """
    if not vectors:
        return 0
    points = build_points(vectors)
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


# ---------------------------------------------------------------------------
# Read path
#
# ``paper_embeddings`` was the only fetch-a-vector-by-bibcode source until
# ADR-015 dropped it, leaving every non-embed reader broken (beads 6ou, 5z5,
# w7m). These helpers are the replacement.
#
# ``point_id`` is uuid5, which is one-way: a returned point cannot be mapped
# back to its bibcode by inverting the id. The bibcode therefore always comes
# from the payload, and a point missing it is a contract violation rather than
# something to skip — silently dropping points would under-report coverage
# while looking like a clean run.
# ---------------------------------------------------------------------------

DEFAULT_FETCH_BATCH = 1_000


def _bibcode_and_vector(record: Any) -> tuple[str, list[float]]:
    """Extract ``(bibcode, vector)`` from a Qdrant record, or raise."""
    payload = record.payload or {}
    bibcode = payload.get("bibcode")
    if not bibcode:
        raise ValueError(
            f"Qdrant point {record.id!r} in the dense lane has no 'bibcode' payload; "
            "the point-id scheme (uuid5) is one-way, so it cannot be attributed. "
            "This breaks the collection contract documented in scix.qdrant_dense."
        )
    vector = record.vector
    if vector is None:
        raise ValueError(
            f"Qdrant point {record.id!r} (bibcode {bibcode}) returned no vector "
            "despite with_vectors=True."
        )
    return bibcode, list(vector)


def fetch_dense(
    client: Any,
    collection: str,
    bibcodes: Iterable[str],
    batch_size: int = DEFAULT_FETCH_BATCH,
) -> dict[str, list[float]]:
    """Return ``{bibcode: vector}`` for those ``bibcodes`` present in ``collection``.

    Bibcodes with no point are omitted rather than raising — this preserves the
    contract the dropped SQL had ("vectors for all bibcodes that have them"), so
    callers keep their existing partial-result handling. Input order is not
    preserved; duplicates collapse.
    """
    wanted = list(dict.fromkeys(bibcodes))
    if not wanted:
        return {}

    out: dict[str, list[float]] = {}
    for i in range(0, len(wanted), batch_size):
        batch = wanted[i : i + batch_size]
        records = client.retrieve(
            collection_name=collection,
            ids=[point_id(b) for b in batch],
            with_payload=True,
            with_vectors=True,
        )
        for record in records:
            bibcode, vector = _bibcode_and_vector(record)
            out[bibcode] = vector
    return out


def scroll_dense(
    client: Any,
    collection: str,
    batch_size: int = DEFAULT_FETCH_BATCH,
    limit: int | None = None,
) -> Iterator[list[tuple[str, list[float]]]]:
    """Yield ``[(bibcode, vector), ...]`` batches over the whole collection.

    Streams via Qdrant's offset-paged scroll so the client never holds more
    than ``batch_size`` points. ``limit`` caps the total number of points
    yielded (for validation runs); ``None`` scrolls the full collection.

    Vectors are returned in the collection's storage precision (float16
    down-converted to float on the wire), not the float32 the retired
    ``paper_embeddings`` column held.
    """
    if limit is not None and limit <= 0:
        return

    offset: Any = None
    emitted = 0
    while True:
        page_size = batch_size
        if limit is not None:
            page_size = min(batch_size, limit - emitted)
        records, offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if records:
            batch = [_bibcode_and_vector(r) for r in records]
            emitted += len(batch)
            yield batch
        if offset is None or (limit is not None and emitted >= limit):
            return


def count_dense(client: Any, collection: str) -> int:
    """Exact point count for ``collection`` — replaces the retired row count."""
    return int(client.count(collection_name=collection, exact=True).count)
