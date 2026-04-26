"""Tests for ``scix.extract.chunk_pass.collection``.

The collection module is a thin idempotent wrapper around
``QdrantClient.create_collection`` / ``create_payload_index``. We don't reach
out to a running Qdrant — every test uses :class:`unittest.mock.MagicMock` so
the assertions stay focused on the schema parameters the module passes
through.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scix.extract.chunk_pass.collection import (
    CHUNKS_COLLECTION,
    INDEXED_PAYLOAD_FIELDS,
    VECTOR_SIZE,
    chunk_point_id,
    ensure_collection,
)

# qdrant_client is a hard dep of this module — if it's missing we should fail
# loudly rather than skip silently. The module's try/except is only there to
# allow ``import scix.extract.chunk_pass.collection`` in environments where
# the search extra hasn't been installed (e.g. lint runs).
qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client.http import models as qm  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_match_prd() -> None:
    assert CHUNKS_COLLECTION == "scix_chunks_v1"
    assert VECTOR_SIZE == 768

    # Exact field-name + schema-type set from the PRD.
    expected = {
        "bibcode": qm.PayloadSchemaType.KEYWORD,
        "year": qm.PayloadSchemaType.INTEGER,
        "arxiv_class": qm.PayloadSchemaType.KEYWORD,
        "community_id_med": qm.PayloadSchemaType.INTEGER,
        "section_heading_norm": qm.PayloadSchemaType.KEYWORD,
        "doctype": qm.PayloadSchemaType.KEYWORD,
    }
    assert dict(INDEXED_PAYLOAD_FIELDS) == expected
    # Stable ordering matters for the script output / test below.
    assert [name for name, _ in INDEXED_PAYLOAD_FIELDS] == list(expected.keys())


# ---------------------------------------------------------------------------
# ensure_collection — fresh-install path
# ---------------------------------------------------------------------------


def _missing_collection_client() -> MagicMock:
    """A QdrantClient mock that reports the collection as missing."""
    client = MagicMock()
    # get_collection raises -> ensure_collection treats as "not present".
    client.get_collection.side_effect = RuntimeError("not found")
    return client


def test_ensure_collection_creates_with_prd_params() -> None:
    client = _missing_collection_client()

    result = ensure_collection(client)

    assert result["created"] is True
    assert result["payload_indexes_created"] == [
        name for name, _ in INDEXED_PAYLOAD_FIELDS
    ]

    # create_collection called exactly once with the PRD params.
    assert client.create_collection.call_count == 1
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == CHUNKS_COLLECTION
    assert kwargs["on_disk_payload"] is True

    # Vectors: 768-dim COSINE on disk.
    vectors = kwargs["vectors_config"]
    assert isinstance(vectors, qm.VectorParams)
    assert vectors.size == VECTOR_SIZE
    assert vectors.distance == qm.Distance.COSINE
    assert vectors.on_disk is True

    # Quantization: int8 scalar, quantile 0.99, always_ram.
    quant = kwargs["quantization_config"]
    assert isinstance(quant, qm.ScalarQuantization)
    assert quant.scalar.type == qm.ScalarType.INT8
    assert quant.scalar.quantile == pytest.approx(0.99)
    assert quant.scalar.always_ram is True

    # HNSW: m=16, ef_construct=128, on_disk.
    hnsw = kwargs["hnsw_config"]
    assert isinstance(hnsw, qm.HnswConfigDiff)
    assert hnsw.m == 16
    assert hnsw.ef_construct == 128
    assert hnsw.on_disk is True

    # Optimizers: indexing_threshold=10M.
    opt = kwargs["optimizers_config"]
    assert isinstance(opt, qm.OptimizersConfigDiff)
    assert opt.indexing_threshold == 10_000_000


def test_ensure_collection_creates_each_payload_index() -> None:
    client = _missing_collection_client()

    ensure_collection(client)

    # Every (field, schema) tuple from the PRD list is forwarded to
    # create_payload_index — once each, in declared order.
    expected_calls = [
        ((CHUNKS_COLLECTION, name, schema), {})
        for name, schema in INDEXED_PAYLOAD_FIELDS
    ]
    actual_calls = [
        (call.args, call.kwargs) for call in client.create_payload_index.call_args_list
    ]
    assert actual_calls == expected_calls


def test_ensure_collection_honours_custom_collection_name() -> None:
    client = _missing_collection_client()

    ensure_collection(client, collection_name="scix_chunks_test")

    assert client.create_collection.call_args.kwargs["collection_name"] == (
        "scix_chunks_test"
    )
    for call in client.create_payload_index.call_args_list:
        assert call.args[0] == "scix_chunks_test"


# ---------------------------------------------------------------------------
# ensure_collection — idempotent path
# ---------------------------------------------------------------------------


def test_ensure_collection_is_idempotent_when_present() -> None:
    client = MagicMock()
    # get_collection succeeds -> create_collection should be skipped.
    client.get_collection.return_value = MagicMock(status="green")

    result = ensure_collection(client)

    assert result["created"] is False
    client.create_collection.assert_not_called()
    # Payload indexes are still ensured (no-op on the server) so the
    # post-condition guarantee holds even on re-run.
    assert client.create_payload_index.call_count == len(INDEXED_PAYLOAD_FIELDS)
    assert result["payload_indexes_created"] == [
        name for name, _ in INDEXED_PAYLOAD_FIELDS
    ]


def test_ensure_collection_payload_index_swallows_already_exists() -> None:
    """If a payload index already exists, the bootstrap still succeeds."""
    client = MagicMock()
    client.get_collection.return_value = MagicMock(status="green")
    client.create_payload_index.side_effect = RuntimeError("index already exists")

    result = ensure_collection(client)

    assert result["created"] is False
    # Every field is still recorded — the post-condition is what we promise,
    # not "newly created" vs "already there".
    assert result["payload_indexes_created"] == [
        name for name, _ in INDEXED_PAYLOAD_FIELDS
    ]


# ---------------------------------------------------------------------------
# chunk_point_id
# ---------------------------------------------------------------------------


def test_chunk_point_id_is_deterministic() -> None:
    a = chunk_point_id("2024ApJ...999..001A", "v1.2.3", 0)
    b = chunk_point_id("2024ApJ...999..001A", "v1.2.3", 0)
    assert a == b


def test_chunk_point_id_varies_per_input() -> None:
    base = chunk_point_id("2024ApJ...999..001A", "v1.2.3", 0)
    assert chunk_point_id("2024ApJ...999..001B", "v1.2.3", 0) != base
    assert chunk_point_id("2024ApJ...999..001A", "v1.2.4", 0) != base
    assert chunk_point_id("2024ApJ...999..001A", "v1.2.3", 1) != base


def test_chunk_point_id_fits_in_63_bits() -> None:
    # Sample a handful of inputs — every result must fit in an unsigned 63-bit
    # int (Qdrant's accepted positive-int range).
    samples = [
        ("2024ApJ...999..001A", "v1.0.0", 0),
        ("2024ApJ...999..001A", "v1.0.0", 99_999),
        ("2099XYZ..123..456Z", "v999.999.999", 2**31 - 1),
        ("", "", 0),
    ]
    for bibcode, parser_version, chunk_id in samples:
        pid = chunk_point_id(bibcode, parser_version, chunk_id)
        assert 0 <= pid < 2**63
