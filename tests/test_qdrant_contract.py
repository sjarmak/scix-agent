"""qdrant-client contract tests (PRD MH-16, bead 5jtf).

The dense lane (ADR-013) depends on qdrant-client behavior that drifted
across minor versions in the premortem narrative: score semantics, result
ordering, ``SearchParams.exact``, and scroll cursor semantics. This suite
pins those behaviors against the *installed* client using its in-process
local mode (``:memory:``) — no server required — so any version bump that
changes the contract fails CI here before it reaches the serving path.

Bump procedure: install the candidate version, re-run this file, then
update PINNED_VERSION below AND the ``qdrant-client`` pin in pyproject.toml
in the same commit.

Deviation from the MH-16 text: the acceptance asks for "response shape with
named vectors", but the production collection (scix_indus_v2_papers_s1)
uses a single unnamed vector — the shape asserted here matches actual usage
in scix.search._vector_search_qdrant.
"""

from __future__ import annotations

import importlib.metadata
import math

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

# Must match the exact pin in pyproject.toml [project.optional-dependencies].
PINNED_VERSION = "1.17.1"

COLLECTION = "contract_test"


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# Fixture vectors with known cosine relationships to the query [1, 0, 0, 0]:
# identical (1.0), near (≈0.994), and two orthogonal (0.0) vectors.
_FIXTURES = [
    ("BIB.IDENTICAL", [1.0, 0.0, 0.0, 0.0], 2020),
    ("BIB.NEAR", [0.9, 0.1, 0.0, 0.0], 2021),
    ("BIB.ORTHO", [0.0, 1.0, 0.0, 0.0], 2022),
    ("BIB.ORTHO2", [0.0, 0.0, 1.0, 0.0], 2023),
]
_QUERY = _normalize([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def client() -> QdrantClient:
    c = QdrantClient(":memory:")
    c.create_collection(
        COLLECTION,
        vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE),
    )
    c.upsert(
        COLLECTION,
        points=[
            qm.PointStruct(
                id=i,
                vector=_normalize(vec),
                payload={"bibcode": bibcode, "year": year},
            )
            for i, (bibcode, vec, year) in enumerate(_FIXTURES)
        ],
    )
    return c


def test_version_is_pinned() -> None:
    """Tripwire: a qdrant-client bump must re-validate this contract suite.

    Update PINNED_VERSION and the pyproject pin together after re-running
    this file against the new version (PRD MH-16 acceptance).
    """
    assert importlib.metadata.version("qdrant-client") == PINNED_VERSION


class TestQueryPointsShape:
    """(a) Response shape as consumed by scix.search._vector_search_qdrant."""

    def test_points_carry_id_score_payload(self, client: QdrantClient) -> None:
        result = client.query_points(COLLECTION, query=_QUERY, limit=4)
        assert hasattr(result, "points")
        for point in result.points:
            assert isinstance(point.score, float)
            assert isinstance(point.payload["bibcode"], str)
        # The serving path builds {payload["bibcode"]: score} — bibcodes
        # must round-trip uniquely.
        bibcodes = [p.payload["bibcode"] for p in result.points]
        assert len(bibcodes) == len(set(bibcodes)) == 4

    def test_limit_caps_result_count(self, client: QdrantClient) -> None:
        result = client.query_points(COLLECTION, query=_QUERY, limit=2)
        assert len(result.points) == 2


class TestScoreSemantics:
    """(c) Cosine score range and ordering for known fixtures.

    The pgvector path computes ``1 - (vec <=> query)``; the Qdrant lane
    relies on ``point.score`` meaning the same cosine similarity.
    """

    def test_scores_match_cosine_similarity(self, client: QdrantClient) -> None:
        result = client.query_points(COLLECTION, query=_QUERY, limit=4)
        by_bibcode = {p.payload["bibcode"]: p.score for p in result.points}
        assert by_bibcode["BIB.IDENTICAL"] == pytest.approx(1.0, abs=1e-6)
        assert by_bibcode["BIB.NEAR"] == pytest.approx(0.9939, abs=1e-3)
        assert by_bibcode["BIB.ORTHO"] == pytest.approx(0.0, abs=1e-6)

    def test_scores_descend_and_stay_in_cosine_range(self, client: QdrantClient) -> None:
        scores = [p.score for p in client.query_points(COLLECTION, query=_QUERY, limit=4).points]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= s <= 1.0 for s in scores)


class TestSearchParamsExact:
    """(b) SearchParams.exact behavior, with and without payload filters."""

    def test_exact_accepted_and_ordering_unchanged(self, client: QdrantClient) -> None:
        default = client.query_points(COLLECTION, query=_QUERY, limit=4)
        exact = client.query_points(
            COLLECTION,
            query=_QUERY,
            limit=4,
            search_params=qm.SearchParams(exact=True),
        )
        assert [p.id for p in exact.points] == [p.id for p in default.points]
        for e, d in zip(exact.points, default.points):
            assert e.score == pytest.approx(d.score, abs=1e-6)

    def test_filtered_query_respects_filter_under_exact(self, client: QdrantClient) -> None:
        result = client.query_points(
            COLLECTION,
            query=_QUERY,
            limit=4,
            query_filter=qm.Filter(must=[qm.FieldCondition(key="year", range=qm.Range(gte=2022))]),
            search_params=qm.SearchParams(exact=True),
        )
        bibcodes = {p.payload["bibcode"] for p in result.points}
        assert bibcodes == {"BIB.ORTHO", "BIB.ORTHO2"}


class TestScrollCursorSemantics:
    """(d) Scroll pagination contract for batch reads (outbox sync et al.)."""

    def test_offset_cursor_visits_every_point_once(self, client: QdrantClient) -> None:
        seen: list[int | str] = []
        offset = None
        pages = 0
        while True:
            points, offset = client.scroll(COLLECTION, limit=2, offset=offset)
            seen.extend(p.id for p in points)
            pages += 1
            if offset is None:
                break
            assert pages < 10, "scroll cursor failed to terminate"
        assert sorted(seen) == list(range(len(_FIXTURES)))

    def test_exhausted_scroll_returns_none_cursor(self, client: QdrantClient) -> None:
        points, offset = client.scroll(COLLECTION, limit=len(_FIXTURES) + 1)
        assert len(points) == len(_FIXTURES)
        assert offset is None
