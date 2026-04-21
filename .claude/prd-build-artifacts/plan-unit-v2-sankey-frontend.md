# Plan — unit-v2-sankey-frontend

## HTML structure (`web/viz/sankey.html`)

```
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SciX — Temporal Community Flow (Sankey)</title>
    <link rel="stylesheet" href="./shared.css" />
    <style> page-specific styles (svg, .sankey-node, .sankey-link, etc.) </style>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12"></script>
  </head>
  <body>
    <main class="viz-container">
      <h1>Temporal Community Flow</h1>
      <p class="viz-sub">...</p>
      <div id="sankey-root"></div>
      <div id="sankey-status"></div>
    </main>
    <script src="./shared.js"></script>
    <script src="./sankey.js"></script>
    <script>
      // Bootstrap: fetch ./sankey.json, fall back to /data/viz/sankey.json,
      // render into #sankey-root, surface error in #sankey-status.
    </script>
  </body>
</html>
```

## JS structure (`web/viz/sankey.js`)

```
window.renderSankey = function renderSankey(data, container) {
  // 1. Validate data shape.
  // 2. Resolve container (selector string or DOM node).
  // 3. Compute width/height from container's bounding box (fallback 1100x600).
  // 4. Build sankey layout via d3.sankey()
  //      .nodeId(d => d.id)
  //      .nodeWidth(14)
  //      .nodePadding(12)
  //      .extent([[1,1],[W-1,H-6]]).
  // 5. Clone data (nodes, links) so d3 can mutate the copies.
  // 6. Run layout.
  // 7. Append svg > g for links, g for nodes.
  //      - links: path with stroke-width = Math.max(1, d.width)
  //      - nodes: rect + text label (decade + community id + paper_count)
  // 8. Color scale: d3.scaleOrdinal(d3.schemeCategory10) keyed on community_id.
  // 9. Hover handler on nodes: highlight incident links (raise opacity),
  //    dim the rest.
  // 10. Click handler on links: console.log({source, target, value, community}).
  // 11. Return void.
};
```

Helpers (private to the file, not on window):

- `_colorFor(communityId)` — memoized ordinal scale.
- `_clearNode(container)` — remove previous chart.
- `_formatNodeLabel(node)` — e.g. `"1990 · c3 (42)"`.
- `_mountError(container, message)` — swap in an error message if data is bad.

## Data loading

The bootstrap block in the HTML:

```
(async () => {
  const root = document.getElementById('sankey-root');
  const status = document.getElementById('sankey-status');
  const urls = ['./sankey.json', '/data/viz/sankey.json'];
  let data = null;
  let lastErr = null;
  for (const u of urls) {
    try {
      const r = await fetch(u);
      if (!r.ok) { lastErr = new Error(`${u} -> ${r.status}`); continue; }
      data = await r.json();
      break;
    } catch (e) { lastErr = e; }
  }
  if (!data) {
    status.textContent = `Could not load sankey.json: ${lastErr}`;
    return;
  }
  window.renderSankey(data, root);
})();
```

## Tests (`tests/test_sankey_frontend.py`)

Four tests:

1. `test_sankey_html_references_d3_sankey` — BeautifulSoup, find a
   `<script src=...>` where the src contains `d3-sankey`.
2. `test_sankey_html_has_required_containers` — ensure
   `<div id="sankey-root">` + `<script src="./sankey.js">` both present.
3. `test_sankey_js_contains_render_symbol` — read `web/viz/sankey.js`,
   assert `'renderSankey'` substring present.
4. `test_sankey_served_by_viz_app` — `TestClient(app).get('/viz/sankey.html')`
   returns 200 and content-type starts with `text/html`.

All read-only; no DB, no network.

## Acceptance-criteria mapping

| AC | Satisfied by |
|----|--------------|
| 1. 200 text/html at /viz/sankey.html | test 4 + static mount |
| 2. d3@7 + d3-sankey@0.12 CDN imports | HTML head |
| 3. renderSankey symbol | sankey.js + test 3 |
| 4a. d3-sankey in <script src> | test 1 |
| 4b. #sankey-root container | test 2 |
| 4c. renderSankey in JS file | test 3 |
| 5. FastAPI integration test | test 4 |
| 6. pytest passes | verified in Phase 4 |
