"""Static + integration tests for the streamgraph frontend (V7 lens).

Mirrors test_umap_frontend.py: BeautifulSoup HTML structure checks plus a
TestClient assertion that the FastAPI viz app serves the bundle. The
renderer's actual D3 stack is exercised by the Playwright smoke test, not
here — we only validate that the wiring is in place.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from scix.viz.server import app


_REPO_ROOT = Path(__file__).resolve().parents[1]
_HTML = _REPO_ROOT / "web" / "viz" / "streamgraph.html"
_JS = _REPO_ROOT / "web" / "viz" / "streamgraph.js"


def _load_soup() -> BeautifulSoup:
    assert _HTML.is_file(), f"Missing {_HTML}"
    return BeautifulSoup(_HTML.read_text(encoding="utf-8"), "html.parser")


def test_streamgraph_html_loads_d3_from_cdn() -> None:
    soup = _load_soup()
    srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any("d3" in s for s in srcs), f"expected a d3 CDN script tag; found {srcs!r}"


def test_streamgraph_html_has_required_containers() -> None:
    soup = _load_soup()
    assert soup.find("div", id="stream-root") is not None
    assert soup.find(id="stream-tooltip") is not None
    assert soup.find(id="stream-panel") is not None
    assert soup.find(id="stream-stats") is not None
    # Shared CSS + JS bundles.
    links = [l.get("href") or "" for l in soup.find_all("link")]
    assert any(h.endswith("shared.css") for h in links), f"missing shared.css link; {links!r}"
    srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any(s.endswith("shared.js") for s in srcs), f"missing shared.js; {srcs!r}"
    assert any(s.endswith("streamgraph.js") for s in srcs), f"missing streamgraph.js; {srcs!r}"


def test_streamgraph_js_has_render_symbol() -> None:
    assert _JS.is_file(), f"Missing {_JS}"
    js = _JS.read_text(encoding="utf-8")
    assert "renderStreamgraph" in js
    # Color must come through the shared resolution-aware palette so the
    # streamgraph stays in sync with the UMAP and ego views.
    assert "scx.colorForCommunity" in js, (
        "streamgraph.js should resolve colors via window.scixViz.colorForCommunity"
    )
    # Streamgraph offset + insideOut order are the two D3 calls that
    # distinguish a streamgraph from an ordinary stacked area.
    assert "stackOffsetWiggle" in js
    assert "stackOrderInsideOut" in js


def test_streamgraph_html_served_by_viz_app() -> None:
    client = TestClient(app)
    resp = client.get("/viz/streamgraph.html")
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("content-type", "").startswith("text/html")
    assert b"stream-root" in resp.content


def test_shared_js_lists_streamgraph_in_nav() -> None:
    """The nav injection in shared.js should advertise the V7 page so the
    streamgraph isn't an orphan that users can only reach by typing the URL."""
    shared = (_REPO_ROOT / "web" / "viz" / "shared.js").read_text(encoding="utf-8")
    assert "streamgraph.html" in shared, "shared.js nav must link streamgraph.html"
    assert "V7" in shared, "shared.js nav should label streamgraph as V7"
