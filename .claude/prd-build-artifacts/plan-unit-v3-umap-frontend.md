# Plan — unit-v3-umap-frontend

## Files to create / edit

1. **`src/scix/viz/api.py`** (new)
   - `from fastapi import APIRouter, Depends, HTTPException, Path`
   - `router = APIRouter(prefix="/viz/api", tags=["viz"])`
   - `def _fetch_paper(bibcode: str) -> dict | None` — real DB query
     - Opens a short-lived psycopg connection via `scix.db.get_connection()`.
     - `SELECT p.bibcode, p.title, p.abstract, pm.community_id_coarse
         FROM papers p
         LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode
         WHERE p.bibcode = %s LIMIT 1`
     - Returns `{"bibcode", "title", "abstract", "community_id"}` or `None`.
   - `def get_fetcher() -> Callable[[str], dict | None]` — dependency that
     returns `_fetch_paper`. Tests override this via
     `app.dependency_overrides[get_fetcher] = lambda: <stub>`.
   - `@router.get("/paper/{bibcode}")` uses `Depends(get_fetcher)` and returns
     the dict, or raises `HTTPException(404)`.
   - Bibcode validation: accept `^[A-Za-z0-9.:&'+%-]{4,32}$` via `Path(...)`
     pattern; mismatches return 422 automatically. This is permissive enough
     for all real bibcodes and strict enough to prevent path traversal.

2. **`src/scix/viz/server.py`** (edit)
   - Add `from scix.viz.api import router as viz_api_router`.
   - Call `app.include_router(viz_api_router)` BEFORE the `app.mount(...)`
     call so the `/viz/api/...` routes win over the static mount at `/viz/`.
   - Leave the existing health route and static mount intact.

3. **`web/viz/umap_browser.html`** (new)
   - `<head>`: meta charset/viewport, title, `<link rel="stylesheet"
     href="./shared.css" />`, `<script src="https://unpkg.com/deck.gl@9/dist.min.js"></script>`.
   - `<body>`:
     - `<h1>UMAP Embedding Browser</h1>`
     - `<div id="umap-root">` — container for the deck.gl canvas
     - `<div id="umap-tooltip">` — hover tooltip overlay
     - Aside/panel: `<aside id="umap-panel">` placeholder for click target
     - `<script src="./umap_browser.js"></script>`
     - Inline bootstrap IIFE that fetches `./umap.json` then
       `/data/viz/umap.json`, on success calls
       `window.renderUMAP(points, document.getElementById('umap-root'))`.

4. **`web/viz/umap_browser.js`** (new)
   - IIFE wrapper, vanilla ES.
   - 20-color categorical palette (hand-rolled, indexed by
     `community_id % PALETTE.length`).
   - `function renderUMAP(points, container)` — creates a
     `new deck.Deck({ parent: container, ... })` with a
     `deck.ScatterplotLayer({ id:'umap-scatter', data: points, ... })`.
   - OrthographicView with `initialViewState` fit to data bounds; pan/zoom
     via `controller: true`.
   - `onHover({ object, x, y })`: show `#umap-tooltip` at cursor, lazy-fetch
     paper title from `/viz/api/paper/{bibcode}` (with a per-render memo).
   - `onClick({ object })`: `console.log(...)` the bibcode and write a tiny
     placeholder into `#umap-panel`.
   - Export: `window.renderUMAP = renderUMAP`.

5. **`tests/test_umap_frontend.py`** (new)
   - `test_umap_html_references_deckgl`: BeautifulSoup check — deck.gl
     script via `unpkg.com/deck.gl` in a `<script src=...>`.
   - `test_umap_html_has_required_containers`: `#umap-root`, `#umap-tooltip`,
     `shared.css` link, `./umap_browser.js` script ref.
   - `test_umap_js_has_render_symbol`: substring `"renderUMAP"` in
     `umap_browser.js`.
   - `test_umap_html_served_by_viz_app`: `TestClient(app).get(
     "/viz/umap_browser.html")` → 200, text/html, body contains `umap-root`.
   - `test_paper_api_returns_404_for_unknown_bibcode`: override
     `get_fetcher` to return a stub that always returns `None`;
     `client.get("/viz/api/paper/1999ApJ...000..000X")` → 404.
   - `test_paper_api_returns_200_for_known_bibcode`: override
     `get_fetcher` to return a sample dict; assert 200 with correct
     JSON shape.
   - `test_paper_api_malformed_bibcode_rejected`: verify a clearly illegal
     path like slashes does not leak through (FastAPI will route-miss, so
     this is just a sanity check for 404/422).

## Route-ordering verification
After `include_router` runs before `app.mount`, order is:
1. `/viz/health` (existing explicit route)
2. `/viz/api/paper/{bibcode}` (new router)
3. `/viz/*` static mount

A request to `/viz/umap_browser.html` still matches the static mount because
neither explicit route collides on its path. `test_viz_server.py` will
confirm no regression.

## Test command
```
.venv/bin/python -m pytest tests/test_umap_frontend.py tests/test_viz_server.py tests/test_sankey_frontend.py -q
```

## Commit
```
git add src/scix/viz/api.py src/scix/viz/server.py \
        web/viz/umap_browser.html web/viz/umap_browser.js \
        tests/test_umap_frontend.py \
        .claude/prd-build-artifacts/research-unit-v3-umap-frontend.md \
        .claude/prd-build-artifacts/plan-unit-v3-umap-frontend.md \
        .claude/prd-build-artifacts/test-unit-v3-umap-frontend.md
git commit -m "prd-build: unit-v3-umap-frontend — UMAP browser HTML/JS (deck.gl) + paper API"
```
