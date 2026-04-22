"""JSON API routes for the SciX viz frontend.

The viz HTML pages are pure static bundles served from ``web/viz/``. A few of
them — the UMAP browser in particular — need to resolve a ``bibcode`` to a
human-readable title on hover. Exposing the MCP surface to the browser would
be overkill (and a security hazard), so we add one tiny read-only lookup
endpoint here and include it on the viz FastAPI app.

The DB call is expressed as a FastAPI dependency (`get_fetcher`) so tests can
override it via ``app.dependency_overrides`` without needing a live database.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from scix import search as scix_search
from scix.db import get_connection
from scix.viz.trace_stream import TraceEvent, publish as publish_trace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/viz/api", tags=["viz"])

# Bibcode regex kept pragmatic — real ADS bibcodes are 19 chars with a narrow
# alphabet, but relaxing to [A-Za-z0-9.:&'+%-]{4,32} keeps us correct for every
# bibcode observed in the corpus while still excluding path-traversal tricks.
_BIBCODE_PATTERN = r"^[A-Za-z0-9.:&'+%\-]{4,32}$"


PaperDict = dict[str, object]
FetcherCallable = Callable[[str], Optional[PaperDict]]


def _fetch_paper(bibcode: str) -> Optional[PaperDict]:
    """Single-paper lookup against the live ``scix`` database.

    Returns a dict shaped ``{bibcode, title, abstract, community_id}`` or
    ``None`` when the bibcode is unknown. Network/DB errors propagate — do
    not swallow them (see coding-style guidance).
    """
    sql = (
        "SELECT p.bibcode, p.title, p.abstract, pm.community_id_coarse "
        "FROM papers p "
        "LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode "
        "WHERE p.bibcode = %s LIMIT 1"
    )
    conn: psycopg.Connection = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (bibcode,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "bibcode": row[0],
        "title": row[1],
        "abstract": row[2],
        "community_id": row[3],
    }


def get_fetcher() -> FetcherCallable:
    """FastAPI dependency returning the concrete paper-fetcher callable.

    Exposing this as a dependency is what lets the test suite swap in a stub
    via ``app.dependency_overrides[get_fetcher] = lambda: <stub>``.
    """
    return _fetch_paper


@router.get("/paper/{bibcode}")
def read_paper(
    bibcode: str = Path(..., pattern=_BIBCODE_PATTERN),
    fetcher: FetcherCallable = Depends(get_fetcher),
) -> PaperDict:
    """Return ``{bibcode,title,abstract,community_id}`` or 404."""
    record = fetcher(bibcode)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {bibcode}")
    return record


# ---------------------------------------------------------------------------
# Live demo: natural-language query -> real hybrid_search -> trace events.
# ---------------------------------------------------------------------------


class DemoSearchRequest(BaseModel):
    """Request payload for POST /viz/api/demo/search."""

    query: str = Field(..., min_length=2, max_length=400)
    top_n: int = Field(default=5, ge=1, le=15)


@router.post("/demo/search")
def demo_search(payload: DemoSearchRequest) -> dict:
    """Run a real lexical hybrid search and narrate the steps as trace events.

    This is the "agentic MCP visualized in real time" hook: each stage of the
    search (query-embed stub, hybrid_search, per-result paper lookup) publishes
    a ``TraceEvent`` so the V4 frontend can animate it on the UMAP overlay.

    Not authenticated — intended for local demo driving over localhost.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")

    # Stage 1: search-dispatch event (no bibcodes yet; panel narrates the query).
    publish_trace(
        TraceEvent(
            tool_name="search",
            latency_ms=0.0,
            params={"query": query[:120]},
            result_summary=f"query={query!r} top_n={payload.top_n}",
            bibcodes=(),
        )
    )

    t0 = time.time()
    conn = get_connection()
    try:
        result = scix_search.hybrid_search(
            conn,
            query_text=query,
            query_embedding=None,  # lexical-only: fast, no INDUS encode
            top_n=payload.top_n,
            lexical_limit=80,
            include_body=False,
        )
    finally:
        try:
            conn.close()
        except Exception:
            logger.debug("conn close failed", exc_info=True)
    search_ms = (time.time() - t0) * 1000.0

    papers = getattr(result, "papers", []) or []
    bibcodes = tuple(str(p.get("bibcode")) for p in papers if p.get("bibcode"))

    # Stage 2: search-results event — bibcodes populate the line-flash on UMAP.
    publish_trace(
        TraceEvent(
            tool_name="search",
            latency_ms=search_ms,
            params={"query": query[:120]},
            result_summary=f"{len(bibcodes)} results in {search_ms:.0f} ms",
            bibcodes=bibcodes,
        )
    )

    # Stage 3: simulate agent drilling into the top-3 — one get_paper per
    # result, spaced so the user can see them flash sequentially.
    drill = []
    for bib in bibcodes[:3]:
        publish_trace(
            TraceEvent(
                tool_name="get_paper",
                latency_ms=2.5,
                params={"bibcode": bib},
                bibcodes=(bib,),
            )
        )
        drill.append(bib)

    return {
        "query": query,
        "latency_ms": round(search_ms, 1),
        "total": getattr(result, "total", len(papers)),
        "bibcodes": list(bibcodes),
        "drilled": drill,
        "papers": [
            {
                "bibcode": p.get("bibcode"),
                "title": p.get("title"),
                "score": p.get("rrf_score") or p.get("score"),
            }
            for p in papers
        ],
    }
