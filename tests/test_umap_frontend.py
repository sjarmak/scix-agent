"""Static + integration tests for the UMAP frontend (unit-v3-umap-frontend).

Three layers are exercised:

* Static HTML/JS layout checks via BeautifulSoup — ensures the bundle
  wires in deck.gl from unpkg and declares the required DOM hooks.
* Integration of the static asset via :class:`fastapi.testclient.TestClient`
  against the existing viz ``app``.
* JSON API contract for ``/viz/api/paper/{bibcode}`` — the DB fetcher is
  stubbed via FastAPI dependency overrides so no database is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from scix.viz.api import get_fetcher
from scix.viz.server import app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UMAP_HTML = _REPO_ROOT / "web" / "viz" / "umap_browser.html"
_UMAP_JS = _REPO_ROOT / "web" / "viz" / "umap_browser.js"


def _load_soup() -> BeautifulSoup:
    assert _UMAP_HTML.is_file(), f"Missing {_UMAP_HTML}"
    return BeautifulSoup(_UMAP_HTML.read_text(encoding="utf-8"), "html.parser")


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------


def test_umap_html_references_deckgl() -> None:
    """umap_browser.html must pull deck.gl from unpkg via a <script src=...>."""
    soup = _load_soup()
    script_srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any("unpkg.com" in src and "deck.gl" in src for src in script_srcs), (
        "Expected a <script src=...> containing 'unpkg.com' and 'deck.gl'; "
        f"found: {script_srcs!r}"
    )


def test_umap_html_has_required_containers() -> None:
    """umap_browser.html must provide the DOM hooks the JS renderer expects."""
    soup = _load_soup()
    root = soup.find("div", id="umap-root")
    assert root is not None, "Missing <div id='umap-root'>"
    tooltip = soup.find(id="umap-tooltip")
    assert tooltip is not None, "Missing element with id='umap-tooltip'"

    script_srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any(
        src.endswith("umap_browser.js") or src == "./umap_browser.js"
        for src in script_srcs
    ), f"Expected a <script src='./umap_browser.js'>; found: {script_srcs!r}"

    # Shared stylesheet must be referenced so visual style stays consistent.
    link_hrefs = [link.get("href") or "" for link in soup.find_all("link")]
    assert any(href.endswith("shared.css") for href in link_hrefs), (
        f"Expected shared.css link; found: {link_hrefs!r}"
    )


def test_umap_js_has_render_symbol() -> None:
    """umap_browser.js must define an identifiable ``renderUMAP`` symbol."""
    assert _UMAP_JS.is_file(), f"Missing {_UMAP_JS}"
    js_text = _UMAP_JS.read_text(encoding="utf-8")
    assert "renderUMAP" in js_text, (
        "umap_browser.js must contain the string 'renderUMAP' "
        "(function definition or window assignment)"
    )


# ---------------------------------------------------------------------------
# Static-mount integration via the FastAPI app
# ---------------------------------------------------------------------------


def test_umap_html_served_by_viz_app() -> None:
    """GET /viz/umap_browser.html via the FastAPI app returns 200 text/html."""
    client = TestClient(app)
    response = client.get("/viz/umap_browser.html")
    assert response.status_code == 200, response.text
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("text/html"), (
        f"Expected text/html content-type, got: {content_type!r}"
    )
    assert b"umap-root" in response.content, (
        "Response body should include the umap-root container"
    )


# ---------------------------------------------------------------------------
# /viz/api/paper/{bibcode} contract — DB call stubbed via dependency override
# ---------------------------------------------------------------------------


def _install_stub(
    stub: Callable[[str], Optional[dict]],
) -> Callable[[], Callable[[str], Optional[dict]]]:
    def _provider() -> Callable[[str], Optional[dict]]:
        return stub

    return _provider


def test_paper_api_returns_404_for_unknown_bibcode() -> None:
    """A fetcher that returns None surfaces as a 404 HTTP response."""
    app.dependency_overrides[get_fetcher] = _install_stub(lambda _bib: None)
    try:
        client = TestClient(app)
        response = client.get("/viz/api/paper/1999ApJ...000..000X")
        assert response.status_code == 404, response.text
        body = response.json()
        assert "detail" in body
        assert "Paper not found" in body["detail"]
    finally:
        app.dependency_overrides.pop(get_fetcher, None)


def test_paper_api_returns_200_for_known_bibcode() -> None:
    """A fetcher that returns a dict surfaces as a 200 with the same payload."""
    sample = {
        "bibcode": "2020ApJ...900..123A",
        "title": "A Sample Title",
        "abstract": "A sample abstract.",
        "community_id": 42,
    }
    app.dependency_overrides[get_fetcher] = _install_stub(lambda _bib: dict(sample))
    try:
        client = TestClient(app)
        response = client.get("/viz/api/paper/2020ApJ...900..123A")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["bibcode"] == sample["bibcode"]
        assert body["title"] == sample["title"]
        assert body["abstract"] == sample["abstract"]
        assert body["community_id"] == sample["community_id"]
    finally:
        app.dependency_overrides.pop(get_fetcher, None)


def test_paper_api_rejects_malformed_bibcode() -> None:
    """Path-pattern validation rejects obviously illegal inputs (422 or 404).

    We don't assert a specific status code because FastAPI may route-miss on
    inputs containing slashes. The key invariant is that we never reach the
    fetcher with garbage — so no 200 OK can ever come back.
    """
    hits: list[str] = []

    def _trap(bibcode: str) -> Optional[dict]:
        hits.append(bibcode)
        return {"bibcode": bibcode, "title": "hit", "abstract": None, "community_id": 0}

    app.dependency_overrides[get_fetcher] = _install_stub(_trap)
    try:
        client = TestClient(app)
        # '$' is outside the allowed pattern, should 422.
        resp = client.get("/viz/api/paper/bad$bibcode")
        assert resp.status_code != 200, resp.text
        assert hits == [], f"fetcher must not be called; got {hits!r}"
    finally:
        app.dependency_overrides.pop(get_fetcher, None)
