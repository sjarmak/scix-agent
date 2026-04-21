"""Static + integration tests for the Sankey frontend (unit-v2-sankey-frontend).

These tests are deliberately lightweight:

* The static tests assert the HTML has the right CDN imports and container
  hooks so future edits cannot silently regress the wiring.
* The JS test is a pure substring check — we are not shipping a JS test
  runner, and static presence is enough to satisfy the acceptance spec.
* The integration test mounts the existing FastAPI viz app in-process via
  :class:`fastapi.testclient.TestClient` and confirms the new HTML is
  served at ``/viz/sankey.html`` with a ``text/html`` content-type.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from scix.viz.server import app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SANKEY_HTML = _REPO_ROOT / "web" / "viz" / "sankey.html"
_SANKEY_JS = _REPO_ROOT / "web" / "viz" / "sankey.js"


def _load_soup() -> BeautifulSoup:
    assert _SANKEY_HTML.is_file(), f"Missing {_SANKEY_HTML}"
    return BeautifulSoup(_SANKEY_HTML.read_text(encoding="utf-8"), "html.parser")


def test_sankey_html_references_d3_sankey() -> None:
    """sankey.html must pull d3-sankey from a CDN via a <script src=...> tag."""
    soup = _load_soup()
    script_srcs = [
        (s.get("src") or "") for s in soup.find_all("script") if s.get("src")
    ]
    assert any("d3-sankey" in src for src in script_srcs), (
        "Expected a <script src=...> containing 'd3-sankey'; found: "
        f"{script_srcs!r}"
    )
    # Also confirm plain d3 (v7) is imported so d3-sankey has its dependency.
    assert any("d3@7" in src for src in script_srcs), (
        f"Expected a <script src=...> containing 'd3@7'; found: {script_srcs!r}"
    )


def test_sankey_html_has_required_containers() -> None:
    """sankey.html must provide #sankey-root and reference ./sankey.js."""
    soup = _load_soup()
    root = soup.find("div", id="sankey-root")
    assert root is not None, "Missing <div id='sankey-root'>"

    script_srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any(
        src.endswith("sankey.js") or src == "./sankey.js" for src in script_srcs
    ), f"Expected a <script src='./sankey.js'>; found: {script_srcs!r}"

    # Shared stylesheet must be referenced so visual style stays consistent.
    link_hrefs = [link.get("href") or "" for link in soup.find_all("link")]
    assert any(href.endswith("shared.css") for href in link_hrefs), (
        f"Expected shared.css link; found: {link_hrefs!r}"
    )


def test_sankey_js_contains_render_symbol() -> None:
    """sankey.js must define an identifiable ``renderSankey`` symbol."""
    assert _SANKEY_JS.is_file(), f"Missing {_SANKEY_JS}"
    js_text = _SANKEY_JS.read_text(encoding="utf-8")
    assert "renderSankey" in js_text, (
        "sankey.js must contain the string 'renderSankey' "
        "(function definition or window assignment)"
    )


def test_sankey_served_by_viz_app() -> None:
    """GET /viz/sankey.html via the FastAPI app returns 200 text/html."""
    client = TestClient(app)
    response = client.get("/viz/sankey.html")
    assert response.status_code == 200, response.text
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("text/html"), (
        f"Expected text/html content-type, got: {content_type!r}"
    )
    assert b"sankey-root" in response.content, (
        "Response body should include the sankey-root container"
    )
