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
from collections.abc import Mapping
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
