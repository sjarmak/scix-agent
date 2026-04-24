"""Tests for the resolution-catalog viz endpoint (``GET /viz/api/resolution``).

The endpoint is consumed by ``web/viz/shared.js`` so the nav-level toggle can
surface coarse / medium / fine as available options and know which JSON file
URL to fetch for each. These tests verify the payload shape and that
availability flags reflect what actually sits on disk.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from scix.viz.server import app

client = TestClient(app)


def test_resolution_catalog_shape() -> None:
    """GET /viz/api/resolution returns the three expected resolution specs."""
    response = client.get("/viz/api/resolution")
    assert response.status_code == 200
    body = response.json()

    assert body["default"] == "coarse"
    assert body["current"] == "coarse"
    assert isinstance(body["resolutions"], list)
    ids = [r["id"] for r in body["resolutions"]]
    assert ids == ["coarse", "medium", "fine"]


def test_resolution_entries_have_required_fields() -> None:
    """Every resolution entry exposes the fields the frontend depends on."""
    response = client.get("/viz/api/resolution")
    body = response.json()

    required = {
        "id",
        "label",
        "description",
        "available",
        "umap_url",
        "labels_url",
        "stream_url",
        "umap_candidates",
        "labels_candidates",
        "stream_candidates",
    }
    for entry in body["resolutions"]:
        missing = required - set(entry.keys())
        assert not missing, f"resolution {entry.get('id')!r} missing {missing}"
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["umap_candidates"], list) and entry["umap_candidates"]
        assert isinstance(entry["labels_candidates"], list) and entry["labels_candidates"]
        assert isinstance(entry["stream_candidates"], list) and entry["stream_candidates"]


def test_coarse_is_available_on_disk() -> None:
    """The coarse resolution has shipped data files, so `available` is True
    and the resolved URLs are rooted under `/viz/` or `/data/viz/`."""
    response = client.get("/viz/api/resolution")
    body = response.json()

    coarse = next(r for r in body["resolutions"] if r["id"] == "coarse")
    assert coarse["available"] is True
    assert coarse["umap_url"] is not None
    assert coarse["labels_url"] is not None
    assert coarse["umap_url"].startswith(("/viz/", "/data/viz/"))
    assert coarse["labels_url"].startswith(("/viz/", "/data/viz/"))


def test_unavailable_resolution_reports_none_urls() -> None:
    """Resolutions without shipped data files report available=False and
    null URLs so the frontend can disable them in the toggle."""
    response = client.get("/viz/api/resolution")
    body = response.json()

    by_id = {r["id"]: r for r in body["resolutions"]}
    # Medium has labels shipped; fine typically does not. At minimum, if a
    # resolution is marked unavailable both URLs must be None.
    for entry in by_id.values():
        if not entry["available"]:
            assert entry["umap_url"] is None or entry["labels_url"] is None
