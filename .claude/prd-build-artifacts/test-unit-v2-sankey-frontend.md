# Test — unit-v2-sankey-frontend

## Command

```
.venv/bin/python -m pytest tests/test_sankey_frontend.py -q
```

## Result

```
....                                                                     [100%]
4 passed in 0.35s
```

## Regression check

Ran together with the pre-existing viz server tests to confirm no regression:

```
.venv/bin/python -m pytest tests/test_viz_server.py tests/test_sankey_frontend.py -q
.......                                                                  [100%]
7 passed in 0.35s
```

## Tests enumerated

1. `test_sankey_html_references_d3_sankey` — BeautifulSoup-based assertion
   that the HTML contains both a `d3@7` and a `d3-sankey` CDN `<script>`.
2. `test_sankey_html_has_required_containers` — asserts presence of
   `<div id="sankey-root">`, a `<script src="./sankey.js">` reference, and
   a `shared.css` stylesheet link.
3. `test_sankey_js_contains_render_symbol` — plain substring check for
   `renderSankey` in `web/viz/sankey.js`.
4. `test_sankey_served_by_viz_app` — in-process `TestClient` GET for
   `/viz/sankey.html`, asserting 200 + `text/html` content-type + presence
   of `sankey-root` in the body.

## Environment note

The project's venv (`.venv/`) was missing `fastapi` despite it being listed
in `pyproject.toml`. Installed via `pip install 'fastapi>=0.110'` to satisfy
the already-declared dependency — no source change required.

## Acceptance criteria → evidence

| AC | Evidence |
|----|----------|
| 1. 200 text/html at /viz/sankey.html | `test_sankey_served_by_viz_app` |
| 2. d3@7 + d3-sankey@0.12 CDN imports | `test_sankey_html_references_d3_sankey` + HTML head |
| 3. `renderSankey` symbol in sankey.js | `test_sankey_js_contains_render_symbol` |
| 4a. `d3-sankey` in `<script src>` | `test_sankey_html_references_d3_sankey` |
| 4b. `#sankey-root` container | `test_sankey_html_has_required_containers` |
| 4c. `renderSankey` in JS file | `test_sankey_js_contains_render_symbol` |
| 5. FastAPI integration test | `test_sankey_served_by_viz_app` |
| 6. `pytest tests/test_sankey_frontend.py -q` passes | Result above |
