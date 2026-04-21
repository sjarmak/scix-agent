# Test — unit-v3-umap-frontend

## Commands run

```
.venv/bin/python -m pytest tests/test_umap_frontend.py -q
# 7 passed in 0.39s

.venv/bin/python -m pytest tests/test_viz_server.py tests/test_sankey_frontend.py -q
# 7 passed in 0.33s  (regression check — no prior tests broken)
```

## Coverage

### `tests/test_umap_frontend.py`
1. `test_umap_html_references_deckgl` — BeautifulSoup: deck.gl script src
   containing `unpkg.com` is present in `umap_browser.html`.
2. `test_umap_html_has_required_containers` — `#umap-root`, `#umap-tooltip`,
   `./umap_browser.js` script tag, `./shared.css` link tag all present.
3. `test_umap_js_has_render_symbol` — substring `"renderUMAP"` present in
   `umap_browser.js`.
4. `test_umap_html_served_by_viz_app` — `TestClient(app).get(
   "/viz/umap_browser.html")` returns 200 `text/html`; body contains
   `umap-root`.
5. `test_paper_api_returns_404_for_unknown_bibcode` — stub fetcher returns
   None via `app.dependency_overrides[get_fetcher]`; GET
   `/viz/api/paper/1999ApJ...000..000X` returns 404 with `detail` containing
   `"Paper not found"`.
6. `test_paper_api_returns_200_for_known_bibcode` — stub fetcher returns
   sample dict; GET `/viz/api/paper/2020ApJ...900..123A` returns 200 with
   the expected JSON shape.
7. `test_paper_api_rejects_malformed_bibcode` — path-pattern validation
   prevents `bad$bibcode` from reaching the fetcher (FastAPI returns 422).

## Acceptance criteria check

1. `/viz/umap_browser.html` returns 200 — covered by
   `test_umap_html_served_by_viz_app`. PASS
2. deck.gl v9 prebuilt bundle referenced via unpkg CDN, plus
   `./umap_browser.js` and `./shared.css` — covered by
   `test_umap_html_references_deckgl` and
   `test_umap_html_has_required_containers`. PASS
3. `renderUMAP(points, container)` builds a deck.gl Deck with a
   ScatterplotLayer — covered structurally by the `renderUMAP` symbol test,
   and inspected via direct source read. PASS
4. `GET /viz/api/paper/{bibcode}` returns 200 with the expected shape or 404
   on miss — covered by the 200 and 404 tests. PASS
5. All tests under `tests/test_umap_frontend.py` pass, and preexisting
   viz/sankey tests still pass. PASS
