"""Qdrant chunk uploader + ``ingest_log`` checkpointing.

This module is the write-side counterpart to
:mod:`scix.extract.chunk_pass.collection` (which defines the schema) and
:mod:`scix.extract.chunk_pass.chunker` (which produces :class:`Chunk` objects).
It takes a batch of ``(Chunk, vector, payload_extras)`` and:

1. assembles the per-point Qdrant payload (indexed + non-indexed fields),
2. upserts the points into ``scix_chunks_v1`` with deterministic ids derived
   from :func:`scix.extract.chunk_pass.collection.chunk_point_id`,
3. records a checkpoint in the ``ingest_log`` table keyed on
   ``chunk_pass:{first_bibcode}`` so the pipeline is resumable.

Idempotency is structural: the point id is a stable hash of
``(bibcode, parser_version, chunk_id)``, so re-uploading the same chunk
overwrites the existing point — never duplicates it.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import psycopg

try:
    from qdrant_client.http import models as qm
except ImportError:  # pragma: no cover
    qm = None  # type: ignore[assignment]

from .chunker import Chunk
from .collection import chunk_point_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

#: Exact key set written into each Qdrant point payload. The first six are
#: payload-indexed (see :data:`scix.extract.chunk_pass.collection.INDEXED_PAYLOAD_FIELDS`);
#: the remaining four ride along as non-indexed metadata.
PAYLOAD_KEYS: tuple[str, ...] = (
    "bibcode",
    "year",
    "arxiv_class",
    "community_id_med",
    "section_heading_norm",
    "doctype",
    "section_idx",
    "char_offset",
    "chunk_id",
    "n_tokens",
)


def assemble_payload(
    chunk: Chunk,
    *,
    year: int | None,
    arxiv_class: str | None,
    community_id_med: int | None,
    doctype: str | None,
    section_heading_norm: str,
) -> dict[str, Any]:
    """Build the per-point Qdrant payload dict for a single ``Chunk``.

    Joins paper-level metadata (``year``, ``arxiv_class``, ``community_id_med``,
    ``doctype``) and the canonicalized section heading with the chunk's own
    structural fields (``bibcode``, ``section_idx``, ``char_offset``,
    ``chunk_id``, ``n_tokens``).

    The output always has exactly the keys listed in :data:`PAYLOAD_KEYS`.
    Callers are responsible for resolving ``year`` etc. from ``papers`` rows
    upstream — this function does no IO.
    """
    return {
        "bibcode": chunk.bibcode,
        "year": year,
        "arxiv_class": arxiv_class,
        "community_id_med": community_id_med,
        "section_heading_norm": section_heading_norm,
        "doctype": doctype,
        "section_idx": chunk.section_idx,
        "char_offset": chunk.char_offset,
        "chunk_id": chunk.chunk_id,
        "n_tokens": chunk.n_tokens,
    }


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------


def upsert_chunks(
    client: Any,
    collection_name: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[dict[str, Any]],
    *,
    parser_version: str,
) -> int:
    """Batch-upsert ``chunks`` into ``collection_name`` and return the count.

    Each point id is derived deterministically from
    :func:`chunk_point_id(chunk.bibcode, parser_version, chunk.chunk_id)`,
    so re-running with the same inputs overwrites in place — never
    duplicates.

    The three input sequences must have the same length; their elements are
    zipped index-wise into one ``PointStruct`` per chunk.
    """
    if qm is None:
        raise RuntimeError(
            "qdrant-client is not installed — install it via the 'search' extra"
        )
    if not (len(chunks) == len(vectors) == len(payloads)):
        raise ValueError(
            "chunks, vectors, and payloads must all have the same length "
            f"(got {len(chunks)}, {len(vectors)}, {len(payloads)})"
        )
    if not chunks:
        return 0

    points = [
        qm.PointStruct(
            id=chunk_point_id(chunk.bibcode, parser_version, chunk.chunk_id),
            vector=list(vector),
            payload=dict(payload),
        )
        for chunk, vector, payload in zip(chunks, vectors, payloads, strict=True)
    ]

    client.upsert(collection_name=collection_name, points=points, wait=False)
    return len(points)


# ---------------------------------------------------------------------------
# ingest_log checkpointing
# ---------------------------------------------------------------------------


def _checkpoint_key(first_bibcode: str) -> str:
    """Return the ``ingest_log.filename`` key for a chunk-pass batch.

    The first bibcode in the batch (deterministic because batches stream in
    bibcode order — see :func:`scix.extract.ner_pass.iter_paper_batches`) is
    used as the watermark.
    """
    return f"chunk_pass:{first_bibcode}"


def is_batch_done(conn: psycopg.Connection, key: str) -> bool:
    """Return True iff ``ingest_log`` has ``status='complete'`` for ``key``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM ingest_log WHERE filename = %s",
            (key,),
        )
        row = cur.fetchone()
    return bool(row and row[0] == "complete")


_CHECKPOINT_UPSERT_SQL = """
    INSERT INTO ingest_log
        (filename, records_loaded, edges_loaded, status, finished_at)
    VALUES (%s, %s, %s, %s, now())
    ON CONFLICT (filename) DO UPDATE
        SET records_loaded = EXCLUDED.records_loaded,
            edges_loaded   = EXCLUDED.edges_loaded,
            status         = EXCLUDED.status,
            finished_at    = now()
"""


def record_checkpoint(
    conn: psycopg.Connection,
    key: str,
    *,
    records_loaded: int,
    status: str = "complete",
) -> None:
    """Upsert the chunk-pass checkpoint row for ``key`` into ``ingest_log``.

    Mirrors the shape of :func:`scix.extract.ner_pass._record_checkpoint`,
    but ``edges_loaded`` is hard-wired to ``0`` because this pass only
    writes Qdrant points (no graph edges). The caller is responsible for
    committing the surrounding transaction.
    """
    with conn.cursor() as cur:
        cur.execute(
            _CHECKPOINT_UPSERT_SQL,
            (key, records_loaded, 0, status),
        )
