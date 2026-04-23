"""Tests for the ego-network viz endpoint (``GET /viz/api/ego/{bibcode}``).

The tests use ``fastapi.testclient.TestClient`` with a stub fetcher injected
via ``app.dependency_overrides`` — no DB, no network.
"""

from __future__ import annotations

from typing import Callable, Optional

import pytest
from fastapi.testclient import TestClient

from scix.viz.api import get_ego_fetcher
from scix.viz.server import app

client = TestClient(app)


def _make_stub_fetcher(
    payload: Optional[dict],
    record_calls: Optional[list] = None,
) -> Callable[[str, int, int, int], Optional[dict]]:
    def _stub(bibcode: str, max_refs: int, max_cites: int, max_second_hop: int):
        if record_calls is not None:
            record_calls.append((bibcode, max_refs, max_cites, max_second_hop))
        return payload

    return _stub


@pytest.fixture(autouse=True)
def _reset_overrides():
    """Ensure dependency overrides don't bleed between tests."""
    yield
    app.dependency_overrides.clear()


def test_ego_returns_structured_neighborhood() -> None:
    """Endpoint returns the full {center, direct_refs, direct_cites,
    second_hop_sample, edges} envelope for a known bibcode."""
    payload = {
        "center": {"bibcode": "2023APS..MART15006K", "title": "Test", "community_id": 3},
        "direct_refs": [
            {"bibcode": "2020Ref..001", "title": "R1", "community_id": 3},
            {"bibcode": "2020Ref..002", "title": "R2", "community_id": 5},
        ],
        "direct_cites": [
            {"bibcode": "2024Cite.001", "title": "C1", "community_id": 3},
        ],
        "second_hop_sample": [
            {"bibcode": "2019Hop..001", "title": "H1", "community_id": 7, "weight": 3},
            {"bibcode": "2019Hop..002", "title": "H2", "community_id": 5, "weight": 2},
        ],
        "edges": [
            {"source": "2023APS..MART15006K", "target": "2020Ref..001", "kind": "ref"},
            {"source": "2023APS..MART15006K", "target": "2020Ref..002", "kind": "ref"},
            {"source": "2024Cite.001", "target": "2023APS..MART15006K", "kind": "cite"},
            {"source": "2020Ref..001", "target": "2019Hop..001", "kind": "hop"},
            {"source": "2020Ref..002", "target": "2019Hop..001", "kind": "hop"},
        ],
        "counts": {"direct_refs": 2, "direct_cites": 1, "second_hop": 2, "edges": 5},
    }

    app.dependency_overrides[get_ego_fetcher] = lambda: _make_stub_fetcher(payload)

    response = client.get("/viz/api/ego/2023APS..MART15006K")
    assert response.status_code == 200
    body = response.json()

    assert body["center"]["bibcode"] == "2023APS..MART15006K"
    assert len(body["direct_refs"]) == 2
    assert len(body["direct_cites"]) == 1
    assert len(body["second_hop_sample"]) == 2
    assert body["second_hop_sample"][0]["weight"] == 3
    # Edge kinds are the three expected values.
    kinds = {e["kind"] for e in body["edges"]}
    assert kinds == {"ref", "cite", "hop"}
    # Latency is attached by the route, not the fetcher.
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0


def test_ego_missing_bibcode_returns_404() -> None:
    """Unknown bibcodes propagate a 404 with a helpful message."""
    app.dependency_overrides[get_ego_fetcher] = lambda: _make_stub_fetcher(None)
    response = client.get("/viz/api/ego/1999Unknown..001")
    assert response.status_code == 404
    assert "Paper not found" in response.json()["detail"]


def test_ego_rejects_malformed_bibcode() -> None:
    """Bibcode pattern validation blocks path-traversal and garbage inputs."""
    # Override still needs to be set; FastAPI validates the path before the
    # dependency runs, but we set a stub so a slipped-through request would
    # still produce a clean response instead of touching the DB.
    app.dependency_overrides[get_ego_fetcher] = lambda: _make_stub_fetcher({})
    response = client.get("/viz/api/ego/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404  # starlette treats traversal as no-route


def test_ego_passes_cap_query_params_to_fetcher() -> None:
    """max_refs/max_cites/max_second_hop are forwarded verbatim."""
    calls: list = []
    payload = {
        "center": {"bibcode": "2023A.test..1", "title": None, "community_id": None},
        "direct_refs": [],
        "direct_cites": [],
        "second_hop_sample": [],
        "edges": [],
        "counts": {"direct_refs": 0, "direct_cites": 0, "second_hop": 0, "edges": 0},
    }
    app.dependency_overrides[get_ego_fetcher] = lambda: _make_stub_fetcher(
        payload, record_calls=calls
    )

    response = client.get(
        "/viz/api/ego/2023A.test..1",
        params={"max_refs": 10, "max_cites": 20, "max_second_hop": 50},
    )
    assert response.status_code == 200
    assert calls == [("2023A.test..1", 10, 20, 50)]


def test_ego_cap_params_are_bounded() -> None:
    """FastAPI rejects cap values outside the documented range."""
    app.dependency_overrides[get_ego_fetcher] = lambda: _make_stub_fetcher({})
    # max_refs=0 < ge=1 -> 422
    r1 = client.get("/viz/api/ego/2023A.test..1", params={"max_refs": 0})
    assert r1.status_code == 422
    # max_second_hop=5000 > le=1000 -> 422
    r2 = client.get("/viz/api/ego/2023A.test..1", params={"max_second_hop": 5000})
    assert r2.status_code == 422


def test_ego_payload_shape_for_isolated_paper() -> None:
    """A paper with no refs, no cites, no hops still returns a well-formed
    envelope (empty arrays, not null)."""
    payload = {
        "center": {"bibcode": "2023Iso..001", "title": "Alone", "community_id": 9},
        "direct_refs": [],
        "direct_cites": [],
        "second_hop_sample": [],
        "edges": [],
        "counts": {"direct_refs": 0, "direct_cites": 0, "second_hop": 0, "edges": 0},
    }
    app.dependency_overrides[get_ego_fetcher] = lambda: _make_stub_fetcher(payload)
    response = client.get("/viz/api/ego/2023Iso..001")
    assert response.status_code == 200
    body = response.json()
    assert body["direct_refs"] == []
    assert body["direct_cites"] == []
    assert body["second_hop_sample"] == []
    assert body["edges"] == []
    assert body["counts"] == {
        "direct_refs": 0,
        "direct_cites": 0,
        "second_hop": 0,
        "edges": 0,
    }
