# Research — unit-v2-sankey-frontend

## Goal

Build a d3.js v7 + d3-sankey v0.12 static HTML frontend (no npm, no build step)
served by the existing SciX viz FastAPI app. The page loads a JSON dataset
produced by `scripts/viz/build_temporal_sankey_data.py` and renders an
interactive Sankey diagram of decade-to-decade community flow.

## Relevant existing files

- `src/scix/viz/server.py` — FastAPI app. Mounts `web/viz/` at `/viz/` as
  StaticFiles (`html=True`) after registering the `/viz/health` route. No
  extra mount for `data/`, so served content is strictly whatever lives in
  `web/viz/`.
- `web/viz/shared.css` — baseline reset + typography. Body uses the system
  font stack, `#fafafa` background, `#222` text, `#1a6fd9` accent color.
  h1/h2/h3 are `font-weight: 600`.
- `web/viz/shared.js` — attaches a `window.scixViz` namespace. Loaded via
  `<script src="./shared.js"></script>` pattern in sibling pages.
- `tests/test_viz_server.py` — demonstrates `TestClient(app)` in-process
  testing pattern. Verifies 200s and content-types without a live server.
- `scripts/viz/build_temporal_sankey_data.py` — writes
  `data/viz/sankey.json` with schema
  `{nodes: [{id, decade, community_id, paper_count}], links: [{source, target, value}]}`.

## Data-serving constraint

`data/viz/sankey.json` is NOT served by the FastAPI app (only `web/viz/` is
mounted). Two pragmatic options:

1. Copy the JSON under `web/viz/` at build time (out of scope for this unit).
2. Try multiple candidate URLs at fetch time, falling back gracefully.

The acceptance-criteria spec calls out option 2 ("try BOTH `./sankey.json`
and `/data/viz/sankey.json`, falling back between them"). We adopt that
pattern — frontend-only change, no backend mount edit.

## Style / convention decisions (JS)

- Vanilla ES2020, 2-space indent, single quotes, semicolons present.
- CDN imports via jsdelivr (`https://cdn.jsdelivr.net/npm/d3@7` and
  `https://cdn.jsdelivr.net/npm/d3-sankey@0.12`). Both expose globals
  (`d3`, `d3.sankey` on the `d3` namespace when using the combined UMD
  bundle). We use the UMD bundles explicitly.
- `renderSankey(data, container)` is assigned to `window.renderSankey` so it
  is greppable by the static test AND callable from the bootstrap script
  embedded in the HTML.
- Color nodes via `d3.scaleOrdinal(d3.schemeCategory10)` keyed by
  `community_id`.
- Hover: dim non-incident links/nodes. Click on a link: `console.log`
  target community info.

## Test plan (Phase 4)

Three pytest tests in `tests/test_sankey_frontend.py`:

1. `test_sankey_html_has_d3_sankey` — BeautifulSoup parse; find a `<script>`
   whose `src` contains `d3-sankey`.
2. `test_sankey_html_has_root_container_and_js` — ensure
   `<div id="sankey-root">` exists and `<script src="./sankey.js">` is
   referenced.
3. `test_sankey_js_defines_render_function` — plain substring check for
   `renderSankey` in `web/viz/sankey.js`.
4. `test_sankey_served_by_viz_app` — `TestClient` GET `/viz/sankey.html`
   returns 200 and `content-type` starts with `text/html`.

## Risks / gotchas

- `d3-sankey` UMD bundle depends on `d3`. Load `d3` first, then
  `d3-sankey`. Verified path is
  `https://cdn.jsdelivr.net/npm/d3-sankey@0.12` which serves `dist/d3-sankey.min.js`.
- If data never loads (both fetch URLs 404), render a user-visible error
  message rather than a blank canvas.
- Keep code modular: one `renderSankey` entrypoint, helpers for color,
  tooltip, and event wiring.
