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
import time
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


class _RateLimitWrap:
    """ASGI middleware — token-bucket rate limiter per bearer token (or per IP).

    Limits:
      - 60 requests / minute sustained rate
      - 10 requests / second burst capacity

    Buckets are stored in-memory with periodic TTL cleanup (no Redis needed).
    Returns HTTP 429 with a Retry-After header when a client exceeds the limit.
    """

    _RATE: float = 60.0 / 60.0  # 1 token per second (60/min)
    _BURST: float = 10.0  # max bucket capacity
    _CLEANUP_INTERVAL: float = 300.0  # purge stale entries every 5 min

    def __init__(self, inner: ASGIApp) -> None:
        self._inner = inner
        # {client_key: (tokens, last_refill_time)}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._last_cleanup: float = time.monotonic()

    @staticmethod
    def _client_key(scope: Scope) -> str:
        """Extract a rate-limit key: bearer token if present, else client IP."""
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            return f"tok:{auth[7:]}"
        client = scope.get("client")
        ip = client[0] if client else "unknown"
        return f"ip:{ip}"

    def _consume(self, key: str) -> float | None:
        """Try to consume one token.  Returns None on success, or seconds
        until a token is available on failure."""
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self._BURST, now))
        # refill
        elapsed = now - last
        tokens = min(self._BURST, tokens + elapsed * self._RATE)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return None
        # how long until 1 token is available?
        wait = (1.0 - tokens) / self._RATE
        self._buckets[key] = (tokens, now)
        return wait

    def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < self._CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        stale = [
            k for k, (_, ts) in self._buckets.items() if now - ts > 120.0
        ]
        for k in stale:
            del self._buckets[k]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._inner(scope, receive, send)
            return

        self._maybe_cleanup()
        key = self._client_key(scope)
        retry_after = self._consume(key)

        if retry_after is not None:
            response = JSONResponse(
                {"error": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
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
    mcp_app = _RateLimitWrap(mcp_app)

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
