# Test results — unit-viz-scaffold

## Command

```
pytest tests/test_viz_server.py -q
```

## Output

```
...                                                                      [100%]
3 passed in 0.19s
```

## Tests run

| Test | Result | Verifies |
|---|---|---|
| `test_health_ok` | PASS | GET `/viz/health` -> 200, JSON `{"status":"ok"}` (acceptance criterion 3) |
| `test_static_shared_css` | PASS | GET `/viz/shared.css` -> 200 with `text/css` content-type (acceptance criterion 4, 7) |
| `test_unknown_viz_path_404` | PASS | GET `/viz/does-not-exist.html` -> 404 (acceptance criterion 7) |

## Environment notes

- fastapi 0.135.3 and httpx 0.28.1 already present in the worktree's Python env — no install needed.
- Tests ran via the repo pytest config which adds `src/` to `pythonpath` (so `from scix.viz.server import app` resolves).
- Manual smoke via `python -c "from scix.viz.server import app, WEB_VIZ_DIR; ..."`:
  - `WEB_VIZ_DIR` resolves to `<repo>/web/viz` and exists.
  - Registered routes include `/viz/health` (route) before `/viz` (static mount), confirming the intended resolution order.

## Acceptance-criteria cross-check

1. `src/scix/viz/__init__.py` exists, defines `__all__ = ["server"]`. OK
2. `src/scix/viz/server.py` exports FastAPI `app` and mounts `web/viz/` at `/viz/`. OK
3. `/viz/health` returns 200 + `{"status":"ok"}`. OK (test_health_ok)
4. `/viz/shared.css` returns 200 with `text/css` content-type. OK (test_static_shared_css)
5. `pyproject.toml` has `viz` optional-deps group with fastapi>=0.110, uvicorn>=0.29, orjson>=3.9. OK
6. `pytest tests/test_viz_server.py -q` passes with zero failures. OK (3 passed)
7. Tests cover health, static file serving, and 404 for unknown viz path. OK
