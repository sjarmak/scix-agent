"""Unit tests for the direct-to-Qdrant dense read/write path (beads s7cy, 6ou).

No live Qdrant or DB: the point-id scheme is asserted against the value the
serving collection actually uses, and the read/write helpers are exercised
through fake clients that record the call. The fakes mirror behaviour verified
against the live ``scix_indus_v2_papers_s1`` collection: ``retrieve`` omits
misses rather than erroring, and vectors come back as plain ``list[float]``.
"""

import pytest

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


# ---------------------------------------------------------------------------
# Read path (ADR-015 fallout: beads 6ou / 5z5 / w7m)
#
# The dropped ``paper_embeddings`` was the only fetch-by-bibcode source. These
# helpers replace it, so the invariants they must hold are the ones the SQL
# gave for free: bibcode identity, partial results on a miss, and no silent
# drops. uuid5 is one-way, so bibcode always comes from the payload.
# ---------------------------------------------------------------------------


class _FakeRecord:
    def __init__(self, id, payload, vector):
        self.id = id
        self.payload = payload
        self.vector = vector


def _rec(bibcode: str, vector: list[float]) -> _FakeRecord:
    return _FakeRecord(qd.point_id(bibcode), {"bibcode": bibcode}, vector)


class _FakeReadClient:
    """Fake with a live-verified shape: retrieve omits misses, scroll pages."""

    def __init__(self, records=(), pages=None, count=0):
        self._by_id = {r.id: r for r in records}
        self._pages = pages or []
        self._count = count
        self.retrieve_calls = []
        self.scroll_calls = []

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        self.retrieve_calls.append(
            {"collection": collection_name, "ids": list(ids), "with_vectors": with_vectors}
        )
        # Live behaviour: a miss is simply absent from the result, not an error.
        return [self._by_id[i] for i in ids if i in self._by_id]

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls.append({"collection": collection_name, "limit": limit, "offset": offset})
        idx = 0 if offset is None else offset
        if idx >= len(self._pages):
            return [], None
        # Qdrant never returns more than `limit` points per call.
        page = self._pages[idx][:limit]
        nxt = idx + 1 if idx + 1 < len(self._pages) else None
        return page, nxt

    def count(self, *, collection_name, exact):
        class _R:
            pass

        r = _R()
        r.count = self._count
        return r


def test_fetch_dense_returns_bibcode_keyed_vectors():
    client = _FakeReadClient(records=[_rec("a", [1.0, 2.0]), _rec("b", [3.0, 4.0])])
    got = qd.fetch_dense(client, qd.INDUS_COLLECTION, ["a", "b"])
    assert got == {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    assert client.retrieve_calls[0]["with_vectors"] is True


def test_fetch_dense_omits_missing_bibcodes_without_raising():
    # Replaces the SQL contract "for all bibcodes that have embeddings".
    client = _FakeReadClient(records=[_rec("a", [1.0])])
    got = qd.fetch_dense(client, qd.INDUS_COLLECTION, ["a", "missing"])
    assert got == {"a": [1.0]}


def test_fetch_dense_batches_large_id_lists():
    records = [_rec(f"b{i}", [float(i)]) for i in range(250)]
    client = _FakeReadClient(records=records)
    got = qd.fetch_dense(client, qd.INDUS_COLLECTION, [f"b{i}" for i in range(250)], batch_size=100)
    assert len(got) == 250
    assert [len(c["ids"]) for c in client.retrieve_calls] == [100, 100, 50]


def test_fetch_dense_empty_input_makes_no_call():
    client = _FakeReadClient()
    assert qd.fetch_dense(client, qd.INDUS_COLLECTION, []) == {}
    assert client.retrieve_calls == []


def test_fetch_dense_deduplicates_repeated_bibcodes():
    client = _FakeReadClient(records=[_rec("a", [1.0])])
    got = qd.fetch_dense(client, qd.INDUS_COLLECTION, ["a", "a", "a"])
    assert got == {"a": [1.0]}
    assert len(client.retrieve_calls[0]["ids"]) == 1


def test_scroll_dense_yields_all_pages_in_batches():
    pages = [[_rec("a", [1.0]), _rec("b", [2.0])], [_rec("c", [3.0])]]
    client = _FakeReadClient(pages=pages)
    batches = list(qd.scroll_dense(client, qd.INDUS_COLLECTION, batch_size=2))
    assert batches == [[("a", [1.0]), ("b", [2.0])], [("c", [3.0])]]


def test_scroll_dense_requests_vectors():
    client = _FakeReadClient(pages=[[_rec("a", [1.0])]])
    list(qd.scroll_dense(client, qd.INDUS_COLLECTION, batch_size=2))
    assert client.scroll_calls[0]["limit"] == 2


def test_scroll_dense_honours_limit_across_pages():
    pages = [[_rec("a", [1.0]), _rec("b", [2.0])], [_rec("c", [3.0])]]
    client = _FakeReadClient(pages=pages)
    batches = list(qd.scroll_dense(client, qd.INDUS_COLLECTION, batch_size=2, limit=3))
    assert [p for b in batches for p in b] == [("a", [1.0]), ("b", [2.0]), ("c", [3.0])]

    client = _FakeReadClient(pages=pages)
    got = [
        p for b in qd.scroll_dense(client, qd.INDUS_COLLECTION, batch_size=2, limit=1) for p in b
    ]
    assert got == [("a", [1.0])]


def test_read_helpers_raise_on_payload_without_bibcode():
    # The payload contract is load-bearing: uuid5 is one-way, so a point with
    # no bibcode cannot be attributed. Dropping it silently would under-report
    # coverage and look like a clean run.
    bad = _FakeRecord(qd.point_id("a"), {}, [1.0])
    client = _FakeReadClient(records=[bad])
    with pytest.raises(ValueError, match="bibcode"):
        qd.fetch_dense(client, qd.INDUS_COLLECTION, ["a"])

    client = _FakeReadClient(pages=[[bad]])
    with pytest.raises(ValueError, match="bibcode"):
        list(qd.scroll_dense(client, qd.INDUS_COLLECTION, batch_size=2))


def test_read_helpers_raise_when_vector_missing():
    # with_vectors=True was requested; a None vector means the call was made
    # wrong or the point is degenerate. Either way it must not read as absent.
    bad = _FakeRecord(qd.point_id("a"), {"bibcode": "a"}, None)
    client = _FakeReadClient(records=[bad])
    with pytest.raises(ValueError, match="vector"):
        qd.fetch_dense(client, qd.INDUS_COLLECTION, ["a"])


def test_count_dense_returns_exact_count():
    client = _FakeReadClient(count=35_473_784)
    assert qd.count_dense(client, qd.INDUS_COLLECTION) == 35_473_784
