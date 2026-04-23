# Research — unit-v3-umap-frontend

## Objective
Build a deck.gl-powered UMAP browser frontend at `web/viz/umap_browser.html` plus
its JS entry `web/viz/umap_browser.js`, and add a small FastAPI route
`GET /viz/api/paper/{bibcode}` so the frontend can resolve paper titles on hover.

## Existing infrastructure discovered

### FastAPI viz server — `src/scix/viz/server.py`
- Repo-anchored path: `web/viz/` mounted at `/viz/`.
- One pre-existing route: `@app.get("/viz/health")`.
- Static mount is done via `app.mount("/viz", StaticFiles(...))`.
- Because FastAPI matches routes in registration order and explicit routes win
  over mounted ASGI apps at the same prefix, any `@router.get(...)` we register
  BEFORE the `app.mount("/viz", ...)` call will take precedence over a static
  file with a colliding path. To be safe, `app.include_router(...)` must be
  called BEFORE `app.mount(...)`.

### Existing search helper — `src/scix/search.py:get_paper`
Signature:
```python
def get_paper(conn: psycopg.Connection, bibcode: str) -> SearchResult:
```
Returns a `SearchResult` with `papers=[dict]` or `papers=[]`. The full paper
dict contains `bibcode`, `title`, `abstract`, and many other columns. For the
viz API we only need `bibcode`, `title`, `abstract`, `community_id`.

This is too heavy for our needs (it joins against `citation_edges` and returns
dozens of fields). For the API endpoint we define our own tiny helper
`_fetch_paper(dsn, bibcode)` that issues a single joined SELECT against
`papers` and `paper_metrics`. That keeps the endpoint fast and, more
importantly, gives us a clean injection point for tests via
`FastAPI.dependency_overrides`.

### DB DSN — `src/scix/db.py`
- `DEFAULT_DSN = os.environ.get("SCIX_DSN", "dbname=scix")`
- `get_connection(dsn, autocommit=False)` opens a psycopg connection.
- `is_production_dsn(dsn)` is used elsewhere to gate destructive ops. Not
  relevant here (read-only SELECT).

### Existing sibling frontend — `web/viz/sankey.html` + `sankey.js`
Pattern we should mirror:
- `<link rel="stylesheet" href="./shared.css" />`
- `<script src="https://cdn.jsdelivr.net/..."></script>` for third-party libs.
- `<div id="sankey-root">` container.
- Inline bootstrap in `<script>` tags at the end of `<body>` that fetches the
  dataset (`./sankey.json` → `/data/viz/sankey.json`) and calls the globally
  exposed `window.renderSankey(...)`.
- `sankey.js` uses an IIFE, exports via `window.renderSankey = renderSankey`.

We follow the same shape for `umap_browser.*`.

### Tests to mirror
- `tests/test_viz_server.py` — `TestClient(app)` usage pattern.
- `tests/test_sankey_frontend.py` — BeautifulSoup static checks +
  `TestClient(app).get("/viz/sankey.html")` integration.

### deck.gl prebuilt bundle
- URL: `https://unpkg.com/deck.gl@9/dist.min.js`
- Exposes the global `deck` with `deck.Deck` and `deck.ScatterplotLayer`.

## Data schema — `data/viz/umap.json`
Array of `{bibcode, x, y, community_id, resolution}` rows, produced by
`scripts/viz/project_embeddings_umap.py`. For coloring we bucket by
`community_id` (coarse resolution); a 20-color categorical palette is applied
by `community_id % 20`.

## Paper API shape
```
GET /viz/api/paper/{bibcode}
200 -> {"bibcode": "...", "title": "...", "abstract": "..." | null,
        "community_id": <int|null>}
404 -> standard FastAPI HTTPException body when paper not found.
```

Bibcode path-parameter validation kept pragmatic: we accept any string of a
small character set (alphanumerics + a few punctuation marks). The 19-char
ADS bibcode format is not strictly enforced at the path — the DB lookup is the
ground truth. Obviously malformed bibcodes just 404 through the normal miss path.

## Test strategy notes
- For the 404 test: override the module-level `_fetch_paper` callable via
  FastAPI's `app.dependency_overrides` or `monkeypatch.setattr`. Dependency
  overrides are cleanest but require the code to express the DB fetch as a
  `Depends(...)` in the route signature. We go with that.
- For the 200 test: same mechanism, just return a sample dict.
- For the HTML/JS static tests: load file contents directly + BeautifulSoup.
