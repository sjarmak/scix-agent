"""Streamable HTTP transport for the SciX MCP server.

Exposes the same 13 tools as the stdio transport (``python -m scix.mcp_server``)
over HTTP so remote agents and Claude Desktop users can connect via URL.

Usage:
    # Start with bearer-token auth (recommended):
    MCP_AUTH_TOKEN=<secret> python -m scix.mcp_server_http

    # Start without auth (local dev only):
    MCP_NO_AUTH=1 python -m scix.mcp_server_http

    # Custom port:
    MCP_PORT=9000 python -m scix.mcp_server_http

Clients connect to:
    POST http://<host>:8000/mcp   (JSON-RPC calls)
    GET  http://<host>:8000/mcp   (SSE stream)

Claude Desktop / Claude Code config:
    {
      "mcpServers": {
        "scix": {
          "url": "http://<host>:8000/mcp",
          "headers": {"Authorization": "Bearer <token>"}
        }
      }
    }
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class _AuthWrap:
    """ASGI middleware that checks bearer token before forwarding to the MCP session manager."""

    def __init__(self, inner: ASGIApp, token: str | None) -> None:
        self._inner = inner
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._token is not None:
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if auth != f"Bearer {self._token}":
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await self._inner(scope, receive, send)


def _build_app() -> Starlette:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from scix.mcp_server import _shutdown, create_server

    server = create_server()

    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        stateless=False,
    )

    auth_token = os.environ.get("MCP_AUTH_TOKEN")
    no_auth = os.environ.get("MCP_NO_AUTH", "").lower() in ("1", "true", "yes")

    if not auth_token and not no_auth:
        raise RuntimeError(
            "Set MCP_AUTH_TOKEN to require bearer-token auth, "
            "or MCP_NO_AUTH=1 to disable (local dev only)."
        )

    mcp_app: ASGIApp = session_manager.handle_request
    if not no_auth:
        mcp_app = _AuthWrap(mcp_app, auth_token)

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "server": "scix-mcp"})

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            host = os.environ.get("MCP_HOST", "0.0.0.0")
            port = int(os.environ.get("MCP_PORT", "8000"))
            auth_mode = "none (MCP_NO_AUTH)" if no_auth else "bearer token"
            logger.info(
                "SciX MCP HTTP server ready at http://%s:%d/mcp (auth=%s)",
                host,
                port,
                auth_mode,
            )
            try:
                yield
            finally:
                _shutdown()

    return Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Mount("/mcp", app=mcp_app),
        ],
        lifespan=lifespan,
        redirect_slashes=False,
    )


app = _build_app()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))

    uvicorn.run(
        "scix.mcp_server_http:app",
        host=host,
        port=port,
        log_level="info",
    )
