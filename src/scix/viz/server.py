"""FastAPI app that serves the SciX visualization frontend.

The frontend is a plain static bundle under ``web/viz/`` — CDN-loaded libraries
(d3, deck.gl, ...) with no npm/build step. This module wires the directory to
``/viz/`` and exposes a minimal health endpoint. It is deliberately independent
of the MCP server so the two processes can evolve separately.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scix import mcp_server
from scix.viz.api import router as viz_api_router
from scix.viz.trace_stream import init_history as init_trace_history
from scix.viz.trace_stream import router as trace_stream_router

logger = logging.getLogger(__name__)

# Repo-anchored path to web/viz/.
# __file__ -> <repo>/src/scix/viz/server.py
#   parents[0]=viz, [1]=scix, [2]=src, [3]=repo_root
WEB_VIZ_DIR: Path = Path(__file__).resolve().parents[3] / "web" / "viz"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Pre-load INDUS so the first semantic/hybrid search isn't a 30 s cold
    # start. _init_model_impl already swallows ImportError/Exception, so a
    # missing torch/GPU degrades to lexical-only without blocking startup.
    mcp_server._init_model_impl()
    # Install the on-disk trace ring buffer and replay recent events into
    # the in-memory deque so agent_trace.html can preload on first paint.
    init_trace_history()
    yield


app = FastAPI(title="SciX Viz", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/viz/health")
def viz_health() -> dict[str, str]:
    """Health probe for the viz server. Cheap, no external calls."""
    return {"status": "ok"}


# Register JSON API routes BEFORE the static mount so that paths like
# /viz/api/paper/{bibcode} resolve to the APIRouter rather than being
# interpreted as static-file lookups.
app.include_router(viz_api_router)

# Register the trace-stream SSE router (adds GET /viz/api/trace/stream) also
# before the static mount so /viz/api/* paths always hit the APIRouter.
app.include_router(trace_stream_router)


# Register the /viz health route BEFORE mounting StaticFiles so the explicit
# route wins over the static mount at the same prefix.
app.mount(
    "/viz",
    StaticFiles(directory=str(WEB_VIZ_DIR), html=True),
    name="viz",
)
