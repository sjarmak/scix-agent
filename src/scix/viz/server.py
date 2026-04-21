"""FastAPI app that serves the SciX visualization frontend.

The frontend is a plain static bundle under ``web/viz/`` — CDN-loaded libraries
(d3, deck.gl, ...) with no npm/build step. This module wires the directory to
``/viz/`` and exposes a minimal health endpoint. It is deliberately independent
of the MCP server so the two processes can evolve separately.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Repo-anchored path to web/viz/.
# __file__ -> <repo>/src/scix/viz/server.py
#   parents[0]=viz, [1]=scix, [2]=src, [3]=repo_root
WEB_VIZ_DIR: Path = Path(__file__).resolve().parents[3] / "web" / "viz"

app = FastAPI(title="SciX Viz", docs_url=None, redoc_url=None)


@app.get("/viz/health")
def viz_health() -> dict[str, str]:
    """Health probe for the viz server. Cheap, no external calls."""
    return {"status": "ok"}


# Register the /viz health route BEFORE mounting StaticFiles so the explicit
# route wins over the static mount at the same prefix.
app.mount(
    "/viz",
    StaticFiles(directory=str(WEB_VIZ_DIR), html=True),
    name="viz",
)
