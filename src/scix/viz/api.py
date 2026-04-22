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
from fastapi import APIRouter, Depends, HTTPException, Path, Query
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
EgoFetcherCallable = Callable[[str, int, int, int], Optional[PaperDict]]


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


# ---------------------------------------------------------------------------
# Ego-network viewer: /viz/api/ego/{bibcode}
#
# Returns the 1-hop citation neighborhood of a paper plus a weighted sample
# of its 2-hop neighborhood, shaped for a force-directed graph view. Direct
# edges ("ref" = center cites neighbor, "cite" = neighbor cites center) come
# straight from ``citation_edges``. Second-hop nodes are ranked by the number
# of distinct 1-hop neighbors that link to them — that is the "weight".
#
# Edge set is intentionally flat: each edge carries {source, target, kind} so
# the client can color/thicken without re-deriving topology. No weight >1 is
# exposed on individual edges because ``citation_edges`` is a boolean relation
# (PK on (source, target)); edge thickness on the frontend is driven by the
# node weight of the 2-hop endpoint.
# ---------------------------------------------------------------------------


# Caps are exposed as per-request Query params (clamped) so the frontend can
# ask for a smaller view on mobile without a server redeploy.
MAX_DIRECT_REFS_DEFAULT = 100
MAX_DIRECT_CITES_DEFAULT = 100
MAX_SECOND_HOP_DEFAULT = 200

# Per-neighbor cap on edges pulled during the 2-hop expansion. Highly-cited
# papers (review articles, seminal works) can have 100k+ inbound edges; without
# this cap, the second-hop query scans tens of millions of rows. 60 was chosen
# empirically as the largest value that keeps a cold-cache fetch under 1s for
# typical papers (100 refs + ~20 cites) on the 299M-edge production graph.
_PER_NEIGHBOR_EDGE_CAP = 60


def _fetch_ego_network(
    center_bibcode: str,
    max_refs: int,
    max_cites: int,
    max_second_hop: int,
) -> Optional[PaperDict]:
    """Build a 1-hop + weighted 2-hop citation neighborhood around a bibcode.

    Returns ``None`` if the center bibcode is not present in ``papers``. All
    DB errors propagate — do not swallow them (see coding-style guidance).

    The structure returned:

    ``{
        "center": {bibcode, title, community_id},
        "direct_refs":  [{bibcode, title, community_id}, ...],   # center cites
        "direct_cites": [{bibcode, title, community_id}, ...],   # cites center
        "second_hop_sample": [
            {bibcode, title, community_id, weight: int}, ...
        ],
        "edges": [{source, target, kind: 'ref'|'cite'|'hop'}, ...]
    }``
    """
    conn: psycopg.Connection = get_connection()
    try:
        with conn.cursor() as cur:
            # --- 1. Center paper -------------------------------------------
            cur.execute(
                "SELECT p.bibcode, p.title, pm.community_id_coarse "
                "FROM papers p "
                "LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode "
                "WHERE p.bibcode = %s LIMIT 1",
                (center_bibcode,),
            )
            center_row = cur.fetchone()
            if center_row is None:
                return None
            center = {
                "bibcode": center_row[0],
                "title": center_row[1],
                "community_id": center_row[2],
            }

            # --- 2. Direct refs + cites (one indexed lookup each) ----------
            cur.execute(
                "SELECT target_bibcode FROM citation_edges "
                "WHERE source_bibcode = %s LIMIT %s",
                (center_bibcode, max_refs),
            )
            ref_bibs = [r[0] for r in cur.fetchall()]

            cur.execute(
                "SELECT source_bibcode FROM citation_edges "
                "WHERE target_bibcode = %s LIMIT %s",
                (center_bibcode, max_cites),
            )
            cite_bibs = [r[0] for r in cur.fetchall()]

            # --- 3. Second hop, weighted by 1-hop-path count ---------------
            neighbor_bibs = list(dict.fromkeys(ref_bibs + cite_bibs))
            second_hop_rows: list[tuple[str, int]] = []
            # Map 2-hop bib -> list of 1-hop neighbors that reached it, for
            # edge emission below.
            second_hop_edges: dict[str, list[str]] = {}

            if neighbor_bibs and max_second_hop > 0:
                # LATERAL per-neighbor LIMIT avoids blowups on highly-cited
                # hubs: each neighbor contributes at most
                # ``_PER_NEIGHBOR_EDGE_CAP`` rows per direction.
                cur.execute(
                    """
                    WITH n AS (
                        SELECT unnest(%s::text[]) AS bib
                    ),
                    hop_raw AS (
                        SELECT n.bib AS via, h.bib AS hop
                        FROM n,
                        LATERAL (
                            (SELECT target_bibcode AS bib
                               FROM citation_edges
                              WHERE source_bibcode = n.bib
                              LIMIT %s)
                            UNION ALL
                            (SELECT source_bibcode AS bib
                               FROM citation_edges
                              WHERE target_bibcode = n.bib
                              LIMIT %s)
                        ) h
                    )
                    SELECT hop,
                           count(DISTINCT via) AS weight,
                           array_agg(DISTINCT via) AS via_list
                    FROM hop_raw
                    WHERE hop <> %s
                      AND hop <> ALL(%s::text[])
                    GROUP BY hop
                    ORDER BY weight DESC, hop ASC
                    LIMIT %s
                    """,
                    (
                        neighbor_bibs,
                        _PER_NEIGHBOR_EDGE_CAP,
                        _PER_NEIGHBOR_EDGE_CAP,
                        center_bibcode,
                        neighbor_bibs,
                        max_second_hop,
                    ),
                )
                for hop, weight, via_list in cur.fetchall():
                    second_hop_rows.append((hop, int(weight)))
                    second_hop_edges[hop] = list(via_list or [])

            # --- 4. Hydrate titles + community for every node in one shot -
            all_bibs: list[str] = list(
                dict.fromkeys(
                    [center_bibcode]
                    + ref_bibs
                    + cite_bibs
                    + [hop for hop, _ in second_hop_rows]
                )
            )
            cur.execute(
                "SELECT p.bibcode, p.title, pm.community_id_coarse "
                "FROM papers p "
                "LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode "
                "WHERE p.bibcode = ANY(%s::text[])",
                (all_bibs,),
            )
            meta: dict[str, dict[str, object]] = {}
            for row in cur.fetchall():
                meta[row[0]] = {"title": row[1], "community_id": row[2]}
    finally:
        conn.close()

    def _node(bib: str) -> dict[str, object]:
        m = meta.get(bib, {})
        return {
            "bibcode": bib,
            "title": m.get("title"),
            "community_id": m.get("community_id"),
        }

    direct_refs = [_node(b) for b in ref_bibs]
    direct_cites = [_node(b) for b in cite_bibs]
    second_hop_sample = [{**_node(b), "weight": w} for b, w in second_hop_rows]

    # Edges: center->ref for each ref, cite->center for each cite, and for
    # each 2-hop node, one edge per 1-hop neighbor that reached it.
    edges: list[dict[str, object]] = []
    for b in ref_bibs:
        edges.append({"source": center_bibcode, "target": b, "kind": "ref"})
    for b in cite_bibs:
        edges.append({"source": b, "target": center_bibcode, "kind": "cite"})
    for hop, _ in second_hop_rows:
        for via in second_hop_edges.get(hop, []):
            edges.append({"source": via, "target": hop, "kind": "hop"})

    # Prefer the just-hydrated metadata for the center — it's the same query
    # we used for neighbors, so the fields stay consistent.
    if center_bibcode in meta:
        center["community_id"] = meta[center_bibcode].get(
            "community_id", center["community_id"]
        )
        if meta[center_bibcode].get("title"):
            center["title"] = meta[center_bibcode]["title"]

    return {
        "center": center,
        "direct_refs": direct_refs,
        "direct_cites": direct_cites,
        "second_hop_sample": second_hop_sample,
        "edges": edges,
        "counts": {
            "direct_refs": len(direct_refs),
            "direct_cites": len(direct_cites),
            "second_hop": len(second_hop_sample),
            "edges": len(edges),
        },
    }


def get_ego_fetcher() -> EgoFetcherCallable:
    """FastAPI dependency returning the concrete ego-network fetcher.

    Like ``get_fetcher`` above, this lets tests swap in a stub via
    ``app.dependency_overrides[get_ego_fetcher] = lambda: <stub>``.
    """
    return _fetch_ego_network


@router.get("/ego/{bibcode}")
def read_ego_network(
    bibcode: str = Path(..., pattern=_BIBCODE_PATTERN),
    max_refs: int = Query(MAX_DIRECT_REFS_DEFAULT, ge=1, le=500),
    max_cites: int = Query(MAX_DIRECT_CITES_DEFAULT, ge=1, le=500),
    max_second_hop: int = Query(MAX_SECOND_HOP_DEFAULT, ge=0, le=1000),
    fetcher: EgoFetcherCallable = Depends(get_ego_fetcher),
) -> PaperDict:
    """Return the citation ego network for ``bibcode`` or 404 if unknown."""
    t0 = time.time()
    record = fetcher(bibcode, max_refs, max_cites, max_second_hop)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {bibcode}")
    record["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
    return record
