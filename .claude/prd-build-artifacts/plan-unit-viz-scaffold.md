# Plan — unit-viz-scaffold

## Files to create / edit (in order)

1. `web/viz/shared.css` — 10+ lines: basic reset, body font, heading/link defaults.
2. `web/viz/shared.js` — tiny IIFE with a `console.log` so the file is non-empty.
3. `web/viz/README.md` — one-paragraph plain markdown noting this dir hosts static demo assets consumed by the FastAPI viz server.
4. `src/scix/viz/__init__.py` — module docstring + `__all__ = ["server"]`.
5. `src/scix/viz/server.py` — FastAPI `app`, `/viz/health` route (registered before mount), StaticFiles mount at `/viz`.
6. `pyproject.toml` — add `viz = ["fastapi>=0.110", "uvicorn>=0.29", "orjson>=3.9"]` to `[project.optional-dependencies]`.
7. `tests/test_viz_server.py` — TestClient-based tests for health, static asset, 404.

## FastAPI app structure

```
app = FastAPI(title="SciX Viz", docs_url=None, redoc_url=None)

@app.get("/viz/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

app.mount("/viz", StaticFiles(directory=str(WEB_VIZ_DIR), html=True), name="viz")
```

Route-before-mount order is what lets `/viz/health` resolve to the FastAPI handler instead of hitting StaticFiles (StaticFiles at `/viz/health` would return 404 anyway, but explicit ordering is clearer and guaranteed).

## WEB_VIZ_DIR resolution

- `__file__` is `<repo>/src/scix/viz/server.py`.
- `Path(__file__).resolve().parents[3]` = repo root.
- `WEB_VIZ_DIR = Path(__file__).resolve().parents[3] / "web" / "viz"`.
- Works from any CWD because it's anchored to `__file__`.

## Test strategy

Using `fastapi.testclient.TestClient(app)`:

1. `test_health_ok` — GET `/viz/health`; assert 200, JSON body `{"status":"ok"}`.
2. `test_static_shared_css` — GET `/viz/shared.css`; assert 200, `content-type` starts with `text/css`, body non-empty.
3. `test_unknown_viz_path_404` — GET `/viz/does-not-exist.html`; assert 404.

StaticFiles reads from the filesystem, so the test's correctness depends on `web/viz/shared.css` existing at the resolved path. Because we commit the file to the repo, both local dev and CI get it for free — no fixture / temp-dir plumbing needed.

## Risk / edge cases

- If a user runs tests from a weird working directory: mitigated by anchoring to `__file__`.
- If StaticFiles returns a redirect for a directory request: we only hit leaf files and the health route, no directory hits.
- `html=True` is harmless here: it makes `index.html` the default doc for `/viz/` but we don't rely on that behaviour for the acceptance criteria.
