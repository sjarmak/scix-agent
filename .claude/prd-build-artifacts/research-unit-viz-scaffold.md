# Research — unit-viz-scaffold

## Repo conventions noted

- `pyproject.toml` uses setuptools with `src/` layout (`[tool.setuptools.packages.find] where = ["src"]`).
- `[project.optional-dependencies]` already defines groups: dev, embed, search, mcp, analysis, docling, graph, ner_eval. Add a new `viz` group alongside — do not disturb existing.
- black `line-length = 100`, ruff `line-length = 100`, ruff lint selects `E, F, I, W`.
- pytest config: `testpaths = ["tests"]`, `pythonpath = ["src", "tests", "scripts"]`. So `from scix.viz.server import app` will resolve from `src/scix/viz/server.py`.
- `src/scix/__init__.py` is empty (0 lines) — package-init conventions are minimal; we can safely create a similarly minimal `src/scix/viz/__init__.py`.
- Existing modules (e.g. `src/scix/search.py`) use `from __future__ import annotations`, module-level docstring, standard stdlib-then-third-party-then-local import ordering, type annotations on all function signatures.

## Dependency status

- `fastapi` 0.135.3 already importable in worktree Python.
- `httpx` 0.28.1 already importable — needed by `fastapi.testclient.TestClient`.
- So `pytest tests/test_viz_server.py` should run with no further install.

## FastAPI mount pattern

- `StaticFiles(directory=..., html=True)` mounted at `/viz` will serve `web/viz/*` at paths `/viz/<file>`.
- FastAPI resolves registered routes in order; a route defined with `@app.get("/viz/health")` BEFORE the `/viz` mount will win over StaticFiles for that exact path. This is the approach the unit spec dictates.
- Directory resolution: `pathlib.Path(__file__).resolve().parents[2] / "web" / "viz"` — from `src/scix/viz/server.py`, parents[0]=viz, parents[1]=scix, parents[2]=src; that's wrong. Count again: `__file__` is `.../src/scix/viz/server.py`, so `parents[0]=viz`, `parents[1]=scix`, `parents[2]=src`, `parents[3]=repo_root`. Repo root is `parents[3]`. We want `repo_root / "web" / "viz"`.

## Test strategy

- `fastapi.testclient.TestClient` wraps the ASGI app in httpx and runs requests in-process. No live server, no DB.
- `TestClient(app).get("/viz/health")` -> 200, JSON body.
- `TestClient(app).get("/viz/shared.css")` -> 200; content-type must start with `text/css`. StaticFiles sets this from mimetypes.
- Unknown path `/viz/does-not-exist.html` -> 404 via StaticFiles.
