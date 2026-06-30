"""Static + integration tests for the section-coverage frontend (V11 lens).

Mirrors test_streamgraph_frontend.py: BeautifulSoup HTML structure checks plus
a TestClient assertion that the FastAPI viz app serves the bundle + data. The
renderer's actual D3 draw is exercised by the Playwright smoke test, not here —
we only validate that the wiring is in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from scix.viz.server import app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HTML = _REPO_ROOT / "web" / "viz" / "section_coverage.html"
_JS = _REPO_ROOT / "web" / "viz" / "section_coverage.js"
_DATA = _REPO_ROOT / "web" / "viz" / "section_coverage.json"


def _load_soup() -> BeautifulSoup:
    assert _HTML.is_file(), f"Missing {_HTML}"
    return BeautifulSoup(_HTML.read_text(encoding="utf-8"), "html.parser")


def test_html_loads_d3_from_cdn() -> None:
    soup = _load_soup()
    srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any("d3" in s for s in srcs), f"expected a d3 CDN script tag; found {srcs!r}"


def test_html_has_required_containers_and_bundles() -> None:
    soup = _load_soup()
    assert soup.find("div", id="sc-roles") is not None
    assert soup.find("div", id="sc-decades") is not None
    assert soup.find(id="sc-stats") is not None
    assert soup.find(id="sc-caveat") is not None
    assert soup.find(id="sc-status") is not None
    links = [tag.get("href") or "" for tag in soup.find_all("link")]
    assert any(h.endswith("shared.css") for h in links), f"missing shared.css; {links!r}"
    srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any(s.endswith("shared.js") for s in srcs), f"missing shared.js; {srcs!r}"
    assert any(
        s.endswith("section_coverage.js") for s in srcs
    ), f"missing section_coverage.js; {srcs!r}"


def test_js_has_render_symbol() -> None:
    assert _JS.is_file(), f"Missing {_JS}"
    js = _JS.read_text(encoding="utf-8")
    assert "renderSectionCoverage" in js
    # Both charts must be wired in the public entrypoint.
    assert "renderRoles" in js and "renderDecades" in js


def test_data_payload_shape() -> None:
    assert _DATA.is_file(), f"Missing {_DATA} — run build_section_coverage.py"
    payload = json.loads(_DATA.read_text(encoding="utf-8"))
    assert {"corpus_papers", "fulltext_papers", "fulltext_fraction", "roles", "decades"} <= set(
        payload
    )
    assert len(payload["roles"]) == 5
    for role in payload["roles"]:
        assert {"role", "label", "present_pct", "papers_with"} <= set(role)
        assert 0.0 <= role["present_pct"] <= 1.0
    for dec in payload["decades"]:
        assert dec["fulltext"] <= dec["total"]
        assert 0.0 <= dec["fulltext_pct"] <= 1.0


def test_app_serves_section_coverage_bundle() -> None:
    client = TestClient(app)
    for path in (
        "/viz/section_coverage.html",
        "/viz/section_coverage.js",
        "/viz/section_coverage.json",
    ):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}"


def test_index_and_nav_link_the_page() -> None:
    index = (_REPO_ROOT / "web" / "viz" / "index.html").read_text(encoding="utf-8")
    shared = (_REPO_ROOT / "web" / "viz" / "shared.js").read_text(encoding="utf-8")
    assert "section_coverage.html" in index, "index.html should link the V11 card"
    assert "section_coverage.html" in shared, "shared.js nav should list V11 Sections"
