"""Experimental MCP server for the graph-experiment spike (bead vdtd).

Runs as a separate process from the production scix MCP server. Loads a
pre-built igraph snapshot at startup and exposes graph-traversal tools.

This server is NOT registered in the public/production tool surface — it's
attached locally to the agent harness used to run the day-4 benchmark.

Day-1 deliverable: server skeleton that loads a snapshot and exposes a
single ``_graph_stats`` introspection tool. Real tools land on day 2.

Usage:
    SCIX_GRAPH_EXP_SNAPSHOT=data/graph_experiment/astronomy_1hop.pkl.gz \\
        .venv/bin/python -m scix.mcp_graph_experiment
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from scix.graph_experiment import graph_tools
from scix.graph_experiment.loader import load_snapshot
from scix.graph_experiment.trace import TraceLogger

logger = logging.getLogger(__name__)


_DEFAULT_SNAPSHOT = Path(
    os.environ.get(
        "SCIX_GRAPH_EXP_SNAPSHOT", "data/graph_experiment/astronomy_1hop.pkl.gz"
    )
)
_DEFAULT_TRACE_DIR = Path(
    os.environ.get("SCIX_GRAPH_EXP_TRACE_DIR", "results/graph_experiment_traces")
)


class _ServerState:
    """Process-singleton holding the loaded graph + trace logger."""

    _graph: Any = None
    _trace: TraceLogger | None = None
    _loaded_at: float | None = None
    _snapshot_path: Path | None = None

    @classmethod
    def load(cls, snapshot_path: Path, trace_dir: Path) -> None:
        if cls._graph is not None:
            return
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Snapshot not found: {snapshot_path}. Run "
                f"scripts/build_graph_experiment_slice.py first."
            )
        t0 = time.time()
        cls._graph = load_snapshot(snapshot_path)
        cls._loaded_at = time.time()
        cls._snapshot_path = snapshot_path
        cls._trace = TraceLogger(trace_dir)
        logger.info(
            "loaded snapshot %s (%d nodes, %d edges) in %.1fs; trace -> %s",
            snapshot_path,
            cls._graph.vcount(),
            cls._graph.ecount(),
            cls._loaded_at - t0,
            cls._trace.path,
        )

    @classmethod
    def graph(cls):
        if cls._graph is None:
            raise RuntimeError("Graph not loaded; call _ServerState.load(...) first")
        return cls._graph

    @classmethod
    def trace(cls) -> TraceLogger:
        if cls._trace is None:
            raise RuntimeError("Trace logger not initialised")
        return cls._trace

    @classmethod
    def stats(cls) -> dict[str, Any]:
        g = cls.graph()
        return {
            "snapshot_path": str(cls._snapshot_path),
            "node_count": g.vcount(),
            "edge_count": g.ecount(),
            "vertex_attributes": list(g.vs.attributes()),
            "edge_attributes": list(g.es.attributes()),
            "is_directed": g.is_directed(),
            "session_id": cls.trace().session_id,
            "trace_path": str(cls.trace().path),
        }


def _build_server():
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    _ServerState.load(_DEFAULT_SNAPSHOT, _DEFAULT_TRACE_DIR)

    server = Server("scix-graph-experiment")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="_graph_stats",
                description=(
                    "Return metadata about the loaded graph slice — node and "
                    "edge counts, vertex/edge attributes, snapshot path, and "
                    "trace session id."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="shortest_path",
                description=(
                    "Shortest path between two papers via citation edges. "
                    "mode='out' walks citations forward (cites), 'in' walks "
                    "them backward (cited-by), 'all' ignores direction. "
                    "Returns the path as a list of papers with title/year, "
                    "and the path length."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_bibcode": {"type": "string"},
                        "target_bibcode": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["out", "in", "all"],
                            "default": "all",
                        },
                    },
                    "required": ["source_bibcode", "target_bibcode"],
                },
            ),
            Tool(
                name="subgraph_around",
                description=(
                    "Return the induced citation subgraph around a set of "
                    "seed bibcodes within ``hops`` of any seed. Useful for "
                    "exploring the local citation neighborhood of a working "
                    "set. Caps at max_nodes by descending citation_count."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "seed_bibcodes": {"type": "array", "items": {"type": "string"}},
                        "hops": {"type": "integer", "default": 1},
                        "max_nodes": {"type": "integer", "default": 200},
                    },
                    "required": ["seed_bibcodes"],
                },
            ),
            Tool(
                name="personalized_pagerank",
                description=(
                    "Personalized PageRank seeded on the given bibcodes. "
                    "Returns top-K papers most relevant to the seed set "
                    "according to the citation graph. Useful for HippoRAG-"
                    "style retrieval — given a working set of papers, find "
                    "the most relevant other papers in the slice."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "seed_bibcodes": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "default": 50},
                        "damping": {"type": "number", "default": 0.85},
                    },
                    "required": ["seed_bibcodes"],
                },
            ),
            Tool(
                name="multi_hop_neighbors",
                description=(
                    "Papers reachable from a bibcode within ``depth`` hops. "
                    "mode='out' = papers this one cites (and their citees, "
                    "etc.), 'in' = papers that cite this one (and their "
                    "citers), 'all' = both. Each result carries its hop "
                    "distance. Sorted by hop ascending, citation_count "
                    "descending."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bibcode": {"type": "string"},
                        "depth": {"type": "integer", "default": 2},
                        "mode": {
                            "type": "string",
                            "enum": ["out", "in", "all"],
                            "default": "out",
                        },
                        "max_results": {"type": "integer", "default": 100},
                    },
                    "required": ["bibcode"],
                },
            ),
            Tool(
                name="pattern_match",
                description=(
                    "Walk a fixed-length directed citation pattern from a "
                    "head paper. edge_sequence is a list of 'out'/'in' steps. "
                    "Examples: ['out', 'out'] = papers cited by papers cited "
                    "by X (transitive citation); ['out', 'in'] = co-cited "
                    "papers (papers that share a target with X); ['in', "
                    "'out'] = bibliographic-coupling targets."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "head_bibcode": {"type": "string"},
                        "edge_sequence": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["out", "in"]},
                        },
                        "max_results": {"type": "integer", "default": 100},
                    },
                    "required": ["head_bibcode", "edge_sequence"],
                },
            ),
            Tool(
                name="graph_query_log",
                description=(
                    "Freeform-query escape hatch. Submit a Cypher-style query, "
                    "natural-language graph intent, or any traversal you wish "
                    "the structured primitives supported. The query is NOT "
                    "executed — it is logged for offline analysis of what "
                    "patterns agents would reach for if given an open query "
                    "language. Use the structured tools for actual results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cypher_or_intent": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["cypher_or_intent"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool_handler(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        trace = _ServerState.trace()
        with trace.record(name, arguments) as ev:
            payload = _dispatch(name, arguments)
            if isinstance(payload, dict):
                ev.summarize(
                    result_keys=list(payload.keys()),
                    result_count=_estimate_result_count(payload),
                )
                if "error" in payload:
                    ev.fail(str(payload.get("error")))
        return [TextContent(type="text", text=json.dumps(payload, default=str))]

    return server


def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "_graph_stats":
        return _ServerState.stats()
    g = _ServerState.graph()
    if name == "shortest_path":
        return graph_tools.shortest_path(g, **arguments)
    if name == "subgraph_around":
        return graph_tools.subgraph_around(g, **arguments)
    if name == "personalized_pagerank":
        return graph_tools.personalized_pagerank(g, **arguments)
    if name == "multi_hop_neighbors":
        return graph_tools.multi_hop_neighbors(g, **arguments)
    if name == "pattern_match":
        return graph_tools.pattern_match(g, **arguments)
    if name == "graph_query_log":
        return graph_tools.graph_query_log(**arguments)
    return {"error": "unknown_tool", "name": name}


def _estimate_result_count(payload: dict[str, Any]) -> int | None:
    for key in ("results", "nodes", "path"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return None


async def main() -> None:
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import ServerCapabilities

    server = _build_server()
    init_options = InitializationOptions(
        server_name="scix-graph-experiment",
        server_version="0.0.1",
        capabilities=ServerCapabilities(tools={}),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main())
