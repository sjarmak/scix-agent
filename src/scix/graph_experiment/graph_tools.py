"""Pure-igraph operations exposed by the experimental MCP server.

Each function takes the loaded graph plus tool args and returns a JSON-safe
dict. No DB access — the graph object holds everything needed (bibcode is
the vertex name; title/year/citation_count are vertex attributes).

Conventions:
- All bibcode inputs are looked up via the ``name`` vertex index. Unknown
  bibcodes return a structured ``{"error": "unknown_bibcode", ...}`` payload
  rather than raising — agents see clean error responses.
- All tools cap result size to keep responses manageable.
- Vertex enrichment is uniform via ``_vertex_payload``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)


_DEFAULT_MAX_NODES = 200
_DEFAULT_MAX_NEIGHBORS = 100
_DEFAULT_PPR_TOP_K = 50


def _vertex_payload(graph, vertex_id: int) -> dict[str, Any]:
    v = graph.vs[vertex_id]
    return {
        "bibcode": v["name"],
        "title": v["title"],
        "year": v["year"],
        "citation_count": v["citation_count"],
    }


def _resolve_bibcodes(graph, bibcodes: Iterable[str]) -> tuple[list[int], list[str]]:
    """Return (vertex_ids, unknown_bibcodes)."""
    name_to_idx = {name: i for i, name in enumerate(graph.vs["name"])}
    found: list[int] = []
    missing: list[str] = []
    for b in bibcodes:
        idx = name_to_idx.get(b)
        if idx is None:
            missing.append(b)
        else:
            found.append(idx)
    return found, missing


def shortest_path(
    graph,
    source_bibcode: str,
    target_bibcode: str,
    mode: Literal["out", "in", "all"] = "all",
) -> dict[str, Any]:
    """Shortest path between two bibcodes via citation edges.

    ``mode='out'`` follows citations forward, ``'in'`` follows them
    backward, ``'all'`` ignores edge direction.
    """
    found, missing = _resolve_bibcodes(graph, [source_bibcode, target_bibcode])
    if missing:
        return {"error": "unknown_bibcode", "missing": missing}
    src_id, tgt_id = found
    paths = graph.get_shortest_paths(src_id, to=tgt_id, mode=mode)
    if not paths or not paths[0]:
        return {"path": [], "length": None, "found": False}
    path_ids = paths[0]
    return {
        "path": [_vertex_payload(graph, vid) for vid in path_ids],
        "length": len(path_ids) - 1,
        "found": True,
        "mode": mode,
    }


def subgraph_around(
    graph,
    seed_bibcodes: list[str],
    hops: int = 1,
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> dict[str, Any]:
    """Induced subgraph around seed bibcodes within ``hops`` of any seed.

    Returns the subgraph's nodes (capped at ``max_nodes`` by descending
    citation_count) plus the in-subgraph edges as ``[source_idx, target_idx]``
    pairs into the returned ``nodes`` list.
    """
    seed_ids, missing = _resolve_bibcodes(graph, seed_bibcodes)
    if not seed_ids:
        return {"error": "no_resolvable_seeds", "missing": missing, "nodes": [], "edges": []}

    neighborhood: set[int] = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(max(0, hops)):
        next_frontier: set[int] = set()
        for vid in frontier:
            for nb in graph.neighbors(vid, mode="all"):
                if nb not in neighborhood:
                    neighborhood.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break

    if len(neighborhood) > max_nodes:
        ranked = sorted(
            neighborhood,
            key=lambda i: (graph.vs[i]["citation_count"] or 0),
            reverse=True,
        )
        # Always retain seeds even if they're below citation_count cutoff.
        keep_ranked = ranked[:max_nodes]
        retained = set(keep_ranked) | set(seed_ids)
        neighborhood = retained

    ordered = sorted(neighborhood)
    idx_remap = {global_idx: local_idx for local_idx, global_idx in enumerate(ordered)}

    edges: list[tuple[int, int]] = []
    for global_src in ordered:
        for global_tgt in graph.successors(global_src):
            if global_tgt in idx_remap:
                edges.append((idx_remap[global_src], idx_remap[global_tgt]))

    return {
        "nodes": [_vertex_payload(graph, vid) for vid in ordered],
        "edges": edges,
        "node_count": len(ordered),
        "edge_count": len(edges),
        "missing_seeds": missing,
        "truncated": len(ordered) >= max_nodes,
    }


def personalized_pagerank(
    graph,
    seed_bibcodes: list[str],
    top_k: int = _DEFAULT_PPR_TOP_K,
    damping: float = 0.85,
) -> dict[str, Any]:
    """Personalized PageRank seeded on the given bibcodes.

    Returns the top-K vertices by PPR score (excluding the seeds themselves).
    """
    seed_ids, missing = _resolve_bibcodes(graph, seed_bibcodes)
    if not seed_ids:
        return {"error": "no_resolvable_seeds", "missing": missing, "results": []}

    reset = [0.0] * graph.vcount()
    for sid in seed_ids:
        reset[sid] = 1.0 / len(seed_ids)

    scores = graph.personalized_pagerank(
        reset=reset, damping=damping, directed=True
    )

    seed_set = set(seed_ids)
    ranked = sorted(
        (i for i in range(graph.vcount()) if i not in seed_set),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return {
        "results": [
            {**_vertex_payload(graph, vid), "score": float(scores[vid])}
            for vid in ranked
        ],
        "seed_count": len(seed_ids),
        "missing_seeds": missing,
        "damping": damping,
    }


def multi_hop_neighbors(
    graph,
    bibcode: str,
    depth: int = 2,
    mode: Literal["out", "in", "all"] = "out",
    max_results: int = _DEFAULT_MAX_NEIGHBORS,
) -> dict[str, Any]:
    """Vertices reachable from ``bibcode`` within ``depth`` hops.

    Each result carries the discovered hop distance (1..depth). Sorted by
    hop ascending then citation_count descending. Caps at ``max_results``.
    """
    found, missing = _resolve_bibcodes(graph, [bibcode])
    if missing:
        return {"error": "unknown_bibcode", "missing": missing, "results": []}

    start_id = found[0]
    visited: dict[int, int] = {start_id: 0}
    frontier = {start_id}
    for hop in range(1, max(1, depth) + 1):
        next_frontier: set[int] = set()
        for vid in frontier:
            for nb in graph.neighbors(vid, mode=mode):
                if nb not in visited:
                    visited[nb] = hop
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break

    discovered = [(vid, h) for vid, h in visited.items() if vid != start_id]
    discovered.sort(
        key=lambda pair: (pair[1], -(graph.vs[pair[0]]["citation_count"] or 0))
    )

    return {
        "results": [
            {**_vertex_payload(graph, vid), "hop": hop} for vid, hop in discovered[:max_results]
        ],
        "total_discovered": len(discovered),
        "depth_used": min(depth, max(visited.values()) if visited else 0),
        "mode": mode,
        "truncated": len(discovered) > max_results,
    }


def pattern_match(
    graph,
    head_bibcode: str,
    edge_sequence: list[Literal["out", "in"]],
    max_results: int = _DEFAULT_MAX_NEIGHBORS,
) -> dict[str, Any]:
    """Walk a fixed-length directed pattern from ``head_bibcode``.

    ``edge_sequence`` is a list of edge directions (``'out'`` = follows a
    citation forward, ``'in'`` = follows it backward). Useful for
    expressing patterns like 'papers cited by papers cited by X' (``['out',
    'out']``) or 'co-cited papers' (``['out', 'in']``).
    """
    found, missing = _resolve_bibcodes(graph, [head_bibcode])
    if missing:
        return {"error": "unknown_bibcode", "missing": missing, "results": []}
    if not edge_sequence:
        return {"error": "empty_edge_sequence", "results": []}

    frontier = {found[0]}
    for direction in edge_sequence:
        if direction not in ("out", "in"):
            return {"error": "bad_edge_direction", "direction": direction}
        next_frontier: set[int] = set()
        for vid in frontier:
            next_frontier.update(graph.neighbors(vid, mode=direction))
        frontier = next_frontier
        if not frontier:
            break

    head_id = found[0]
    discovered = sorted(
        (vid for vid in frontier if vid != head_id),
        key=lambda i: -(graph.vs[i]["citation_count"] or 0),
    )

    return {
        "results": [_vertex_payload(graph, vid) for vid in discovered[:max_results]],
        "total_matches": len(discovered),
        "edge_sequence": list(edge_sequence),
        "truncated": len(discovered) > max_results,
    }


def graph_query_log(
    cypher_or_intent: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Logs an open-ended graph query the agent wanted to issue.

    The query is NOT executed — this tool exists to capture evidence of
    multi-hop intent. The trace logger preserves ``cypher_or_intent`` so
    day-4 analysis can categorise the query patterns agents reach for when
    given a freeform escape hatch.
    """
    return {
        "executed": False,
        "logged": True,
        "message": (
            "Freeform graph queries are not executed in this spike — your "
            "query has been logged for analysis. Use the structured "
            "primitives (shortest_path, subgraph_around, "
            "personalized_pagerank, multi_hop_neighbors, pattern_match) "
            "for actual results."
        ),
        "echo": cypher_or_intent,
        "notes": notes,
    }
