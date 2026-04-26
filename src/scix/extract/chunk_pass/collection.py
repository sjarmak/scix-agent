"""Qdrant collection schema for the chunk-embeddings index (``scix_chunks_v1``).

This module is the source of truth for the chunk-collection schema described in
``docs/prd/prd_chunk_embeddings_build.md``. It exposes:

* ``CHUNKS_COLLECTION`` / ``VECTOR_SIZE`` / ``INDEXED_PAYLOAD_FIELDS`` constants
* ``ensure_collection(client, ...)`` — idempotent create-collection +
  create-payload-index helper
* ``chunk_point_id(bibcode, parser_version, chunk_id)`` — deterministic 63-bit
  point id derived from the chunk identity tuple

The module deliberately does not import from ``chunker.py`` — it only describes
how chunks are *stored*, not how they are produced.

Postgres remains the source of truth for chunk text, metadata, and provenance;
Qdrant holds the dense vectors plus a small filterable payload subset.
"""
from __future__ import annotations

import hashlib
import logging
import struct
from typing import Any

try:
    from qdrant_client import QdrantClient  # noqa: F401  (re-exported for typing)
    from qdrant_client.http import models as qm
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    qm = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants (PRD: docs/prd/prd_chunk_embeddings_build.md)
# ---------------------------------------------------------------------------

CHUNKS_COLLECTION = "scix_chunks_v1"
VECTOR_SIZE = 768

# Payload-indexed fields for filtered search. Order matches the PRD listing.
# Tuple-of-tuples so the value is hashable / immutable at module level.
INDEXED_PAYLOAD_FIELDS: tuple[tuple[str, Any], ...] = (
    ("bibcode", qm.PayloadSchemaType.KEYWORD if qm else None),
    ("year", qm.PayloadSchemaType.INTEGER if qm else None),
    ("arxiv_class", qm.PayloadSchemaType.KEYWORD if qm else None),
    ("community_id_med", qm.PayloadSchemaType.INTEGER if qm else None),
    ("section_heading_norm", qm.PayloadSchemaType.KEYWORD if qm else None),
    ("doctype", qm.PayloadSchemaType.KEYWORD if qm else None),
)


# ---------------------------------------------------------------------------
# Point-id derivation
# ---------------------------------------------------------------------------


def chunk_point_id(bibcode: str, parser_version: str, chunk_id: int) -> int:
    """Deterministic 63-bit point id for a (bibcode, parser_version, chunk_id) tuple.

    Mirrors the pattern in :func:`scix.qdrant_tools.bibcode_to_point_id` —
    blake2b → 8-byte big-endian unsigned int → right-shift 1 so the value
    always fits in a signed 63-bit integer (Qdrant's accepted positive-int
    range).

    Same inputs always yield the same id; different inputs collide only with
    the cryptographic-hash collision probability (~2^-32 in 4B points).
    """
    payload = f"{bibcode}:{parser_version}:{chunk_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return struct.unpack(">Q", digest)[0] >> 1


# ---------------------------------------------------------------------------
# Collection bootstrap
# ---------------------------------------------------------------------------


def _collection_exists(client: Any, name: str) -> bool:
    """Return True if ``name`` already exists on ``client``.

    Treats every Qdrant error as "does not exist" — the create call is
    idempotent enough on its own that we'd rather try-and-create than
    falsely skip.
    """
    try:
        client.get_collection(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("collection %s lookup failed (%s); will create", name, exc)
        return False
    return True


def _build_vectors_config() -> Any:
    return qm.VectorParams(
        size=VECTOR_SIZE,
        distance=qm.Distance.COSINE,
        on_disk=True,
    )


def _build_quantization_config() -> Any:
    return qm.ScalarQuantization(
        scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    )


def _build_hnsw_config() -> Any:
    return qm.HnswConfigDiff(
        m=16,
        ef_construct=128,
        on_disk=True,
    )


def _build_optimizers_config() -> Any:
    return qm.OptimizersConfigDiff(
        indexing_threshold=10_000_000,
    )


def ensure_collection(
    client: Any,
    *,
    collection_name: str = CHUNKS_COLLECTION,
) -> dict[str, Any]:
    """Idempotently create the chunks collection and its payload indexes.

    Behaviour:
      * If the collection does not exist, create it with the PRD-mandated
        vector / quantization / HNSW / optimizers config and ``on_disk_payload=True``.
      * Always (re-)issue ``create_payload_index`` for each field in
        :data:`INDEXED_PAYLOAD_FIELDS`. Qdrant treats this as a no-op when the
        index already exists — re-running the bootstrap is safe.

    Returns:
        ``{"created": bool, "payload_indexes_created": list[str]}``. The
        ``payload_indexes_created`` list contains every field for which
        ``create_payload_index`` returned without raising — it is the
        post-condition guarantee, not a "newly created" claim.
    """
    if qm is None:
        raise RuntimeError(
            "qdrant-client is not installed — install it via the 'search' extra"
        )

    created = False
    if not _collection_exists(client, collection_name):
        logger.info("creating Qdrant collection %s", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=_build_vectors_config(),
            quantization_config=_build_quantization_config(),
            hnsw_config=_build_hnsw_config(),
            on_disk_payload=True,
            optimizers_config=_build_optimizers_config(),
        )
        created = True
    else:
        logger.info("collection %s already exists; skipping create", collection_name)

    payload_indexes: list[str] = []
    for field_name, field_schema in INDEXED_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(collection_name, field_name, field_schema)
            payload_indexes.append(field_name)
            logger.debug("ensured payload index on %s (%s)", field_name, field_schema)
        except Exception as exc:  # noqa: BLE001
            # Qdrant raises on duplicate index in some server versions; treat
            # as success so the bootstrap stays idempotent. We still record
            # the field as "ensured" because the post-condition holds.
            logger.debug(
                "create_payload_index(%s) raised %s; treating as already-present",
                field_name,
                exc,
            )
            payload_indexes.append(field_name)

    return {"created": created, "payload_indexes_created": payload_indexes}
