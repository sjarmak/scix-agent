"""Tests for the SciX viz FastAPI server.

These tests use :class:`fastapi.testclient.TestClient` which runs the ASGI app
in-process via httpx. No live server, no DB, no network.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from scix.viz.server import app

client = TestClient(app)


def test_health_ok() -> None:
    """GET /viz/health returns 200 with {'status': 'ok'}."""
    response = client.get("/viz/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_static_shared_css() -> None:
    """GET /viz/shared.css returns 200 with a text/css content type."""
    response = client.get("/viz/shared.css")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("text/css"), content_type
    assert len(response.content) > 0


def test_unknown_viz_path_404() -> None:
    """Unknown static paths under /viz/ return 404."""
    response = client.get("/viz/does-not-exist.html")
    assert response.status_code == 404
