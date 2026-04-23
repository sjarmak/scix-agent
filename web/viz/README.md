# web/viz

Static demo assets served by the SciX visualization server (`src/scix/viz/server.py`).
Files here are mounted at `/viz/` and include shared stylesheets (`shared.css`),
shared JavaScript helpers (`shared.js`), and per-view HTML pages added by later
work units. The frontend is CDN-loaded (d3, deck.gl, etc.) — no npm, no build step.
