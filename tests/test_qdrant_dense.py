"""Unit tests for the direct-to-Qdrant dense write path (bead s7cy).

No live Qdrant or DB: the point-id scheme is asserted against the value the
serving collection actually uses, and upsert is exercised through a fake client
that records the call.
"""

from scix import qdrant_dense as qd


def test_point_id_matches_serving_collection_scheme():
    # Golden value verified by retrieving this bibcode from the live
    # scix_indus_v2_papers_s1 collection (uuid5 over NAMESPACE_URL). If this
    # changes, new points would land under ids the 32.4M bulk-load can't match.
    assert qd.point_id("2020arXiv200407180C") == "9c0c38ac-bb4e-5260-9526-b7b73a773176"


def test_build_points_shape():
    points = qd.build_points({"2020arXiv200407180C": [0.1, 0.2, 0.3]})
    assert len(points) == 1
    p = points[0]
    assert p.id == "9c0c38ac-bb4e-5260-9526-b7b73a773176"
    assert p.vector == [0.1, 0.2, 0.3]
    assert p.payload == {"bibcode": "2020arXiv200407180C"}


class _FakeClient:
    def __init__(self):
        self.calls = []

    def upsert(self, *, collection_name, points, wait):
        self.calls.append({"collection": collection_name, "points": points, "wait": wait})


def test_upsert_dense_calls_client_with_wait_and_returns_count():
    client = _FakeClient()
    n = qd.upsert_dense(client, qd.INDUS_COLLECTION, {"a": [1.0], "b": [2.0]})
    assert n == 2
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["collection"] == qd.INDUS_COLLECTION
    assert call["wait"] is True  # commit-after-durable-write ordering depends on this
    assert {p.payload["bibcode"] for p in call["points"]} == {"a", "b"}


def test_upsert_dense_empty_is_noop():
    client = _FakeClient()
    assert qd.upsert_dense(client, qd.INDUS_COLLECTION, {}) == 0
    assert client.calls == []
