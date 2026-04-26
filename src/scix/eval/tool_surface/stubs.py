"""Stub MCP servers for tool-surface eval.

Three variants of the scix MCP tool surface, each backed by canned synthetic
responses so we measure agent *selection* behavior independently from
retrieval quality.

Run as:

    python -m scix.eval.tool_surface.stubs --variant {v0,v1,v2} --log-file PATH

Each tool call is appended to ``--log-file`` as one JSONL record:

    {"ts": "...", "variant": "v0", "tool": "search", "args": {...}}

The runner reads this log to score tool-selection decisions against an oracle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool

# ---------------------------------------------------------------------------
# Canned synthetic responses — identical shape across variants
# ---------------------------------------------------------------------------

# Synthetic bibcodes that look real (year + journal + page) but are obviously
# not in the corpus. Three SMD disciplines represented.
_SAMPLE_PAPERS = [
    {
        "bibcode": "2024ApJ...001A...1A",
        "title": "Constraints on early dark energy from CMB and BAO",
        "authors": ["Smith J.", "Doe A.", "Lee K."],
        "year": 2024,
        "abstract": "We constrain early dark energy models using Planck and DESI BAO ...",
        "arxiv_class": "astro-ph.CO",
        "discipline": "astrophysics",
    },
    {
        "bibcode": "2023JGRA..002B...2B",
        "title": "Solar wind structure in the Parker Solar Probe encounter 14",
        "authors": ["Wang H.", "Chen R."],
        "year": 2023,
        "abstract": "Magnetic switchbacks observed during PSP E14 perihelion ...",
        "arxiv_class": "astro-ph.SR",
        "discipline": "heliophysics",
    },
    {
        "bibcode": "2025Icar..003C...3C",
        "title": "Subsurface ice distribution on Mars from MARSIS",
        "authors": ["Garcia M.", "Nguyen P."],
        "year": 2025,
        "abstract": "Mid-latitude radar reflections consistent with ice deposits ...",
        "arxiv_class": "astro-ph.EP",
        "discipline": "planetary",
    },
]

_SAMPLE_SECTIONS = [
    {
        "bibcode": "2024ApJ...001A...1A",
        "section": "Methods",
        "snippet": "We use MCMC sampling with emcee, 32 walkers, 2000 steps after burn-in ...",
        "score": 0.81,
    },
    {
        "bibcode": "2023JGRA..002B...2B",
        "section": "Discussion",
        "snippet": "The observed switchback rate is consistent with interchange reconnection ...",
        "score": 0.74,
    },
]

_SAMPLE_CHUNKS = [
    {
        "bibcode": "2024ApJ...001A...1A",
        "chunk_id": 17,
        "snippet": "Our best-fit H0 = 67.3 ± 0.8 km/s/Mpc, in tension with SH0ES ...",
        "score": 0.88,
    },
]

_SAMPLE_CLAIMS = [
    {
        "claim_id": "claim-001",
        "bibcode": "2024ApJ...001A...1A",
        "subject": "early dark energy",
        "predicate": "improves fit to",
        "object": "ACT and Planck CMB data",
        "evidence_section": "Results",
    },
]

_SAMPLE_ENTITIES = [
    {
        "name": "Parker Solar Probe",
        "qid": "Q1396039",
        "type": "mission",
        "discipline": "heliophysics",
    },
    {
        "name": "MARSIS",
        "qid": "Q1894108",
        "type": "instrument",
        "discipline": "planetary",
    },
]


def _stub_payload(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return a canned response keyed by tool name. Tools that consume
    ``grain`` / ``action`` / ``mode`` enums switch their return shape so the
    agent sees realistic-looking data."""
    grain = args.get("grain", "paper")
    action = args.get("action")
    mode = args.get("mode")

    # Unified search (v1/v2) — switch return shape on grain
    if name == "search":
        if grain == "concept":
            return {
                "concepts": ["dark energy", "Hubble tension", "BAO"],
                "papers": _SAMPLE_PAPERS[:2],
            }
        if grain == "section":
            return {"sections": _SAMPLE_SECTIONS}
        if grain == "chunk":
            return {"chunks": _SAMPLE_CHUNKS}
        if grain == "claim":
            return {"claims": _SAMPLE_CLAIMS}
        # default paper
        return {"papers": _SAMPLE_PAPERS, "mode": mode or "hybrid"}

    # Per-grain v0 tools
    if name == "concept_search":
        return {"concepts": ["dark energy", "Hubble tension", "BAO"], "papers": _SAMPLE_PAPERS[:2]}
    if name == "section_retrieval":
        return {"sections": _SAMPLE_SECTIONS}
    if name == "chunk_search":
        return {"chunks": _SAMPLE_CHUNKS}
    if name == "find_claims":
        return {"claims": _SAMPLE_CLAIMS}

    # Paper-ops (v1/v2 unified, v0 split)
    if name == "paper":
        if action == "read":
            return {"bibcode": args.get("bibcode"), "text": "Synthetic full-text section..."}
        if action == "claims":
            return {"bibcode": args.get("bibcode"), "claims": _SAMPLE_CLAIMS}
        if action == "blame":
            return {"claim_id": args.get("claim_id"), "earliest": _SAMPLE_PAPERS[0]}
        if action == "replications":
            return {"bibcode": args.get("bibcode"), "replications": _SAMPLE_PAPERS[1:]}
        return {"bibcode": args.get("bibcode"), "metadata": _SAMPLE_PAPERS[0]}
    if name == "get_paper":
        return {"bibcode": args.get("bibcode"), "metadata": _SAMPLE_PAPERS[0]}
    if name == "read_paper":
        return {"bibcode": args.get("bibcode"), "text": "Synthetic full-text section..."}
    if name == "read_paper_claims":
        return {"bibcode": args.get("bibcode"), "claims": _SAMPLE_CLAIMS}
    if name == "claim_blame":
        return {"claim_id": args.get("claim_id"), "earliest": _SAMPLE_PAPERS[0]}
    if name == "find_replications":
        return {"bibcode": args.get("bibcode"), "replications": _SAMPLE_PAPERS[1:]}

    # Citation
    if name == "citation":
        if mode == "similarity":
            return {"bibcode": args.get("bibcode"), "similar": _SAMPLE_PAPERS[1:]}
        return {"bibcode": args.get("bibcode"), "neighbors": _SAMPLE_PAPERS}
    if name == "citation_traverse":
        return {"bibcode": args.get("bibcode"), "neighbors": _SAMPLE_PAPERS}
    if name == "citation_similarity":
        return {"bibcode": args.get("bibcode"), "similar": _SAMPLE_PAPERS[1:]}

    # Entity
    if name == "entity":
        if action == "context":
            return {"entity": args.get("name"), "co_occurring": _SAMPLE_ENTITIES, "papers": _SAMPLE_PAPERS[:1]}
        if action == "papers":
            return {"entity": args.get("name"), "papers": _SAMPLE_PAPERS}
        return {"entity": args.get("name") or args.get("query"), "matches": _SAMPLE_ENTITIES}
    if name == "entity_context":
        return {"entity": args.get("name"), "co_occurring": _SAMPLE_ENTITIES, "papers": _SAMPLE_PAPERS[:1]}

    # Discovery (kept separate in all variants)
    if name == "graph_context":
        return {
            "query": args.get("query"),
            "entities": _SAMPLE_ENTITIES,
            "communities": [{"id": 12, "label": "early-universe cosmology", "size": 1456}],
        }
    if name == "find_gaps":
        return {
            "query": args.get("query"),
            "gaps": [{"region": "early dark energy + JWST", "paper_count": 3}],
        }
    if name == "temporal_evolution":
        return {
            "query": args.get("query"),
            "series": [{"year": 2020, "n": 12}, {"year": 2024, "n": 47}],
        }
    if name == "facet_counts":
        return {
            "query": args.get("query"),
            "facets": {
                "discipline": {"astrophysics": 1023, "planetary": 218},
                "year": {"2020": 156, "2024": 401},
            },
        }

    return {"error": "stub_no_handler", "tool": name, "args": args}


# ---------------------------------------------------------------------------
# Tool schemas per variant
# ---------------------------------------------------------------------------

# Common reusable schema fragments
_FILTER_PROPS = {
    "year_min": {"type": "integer", "description": "Minimum publication year"},
    "year_max": {"type": "integer", "description": "Maximum publication year"},
    "arxiv_class": {"type": "string", "description": "arXiv classification e.g. 'astro-ph.CO'"},
    "disciplines": {
        "type": "array",
        "items": {"type": "string"},
        "description": "SMD discipline tags",
    },
}


def _v0_tools() -> list[Tool]:
    """Current 18-tool surface (1 optional). Schemas mirror src/scix/mcp_server.py."""
    return [
        Tool(
            name="search",
            description=(
                "Hybrid retrieval over the paper corpus. Returns a ranked list of papers with "
                "title, authors, year, and citation counts. Defaults to hybrid mode (semantic + "
                "BM25 fused via RRF), the best general-purpose default for natural-language queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string", "enum": ["hybrid", "semantic", "keyword"], "default": "hybrid"},
                    "limit": {"type": "integer", "default": 10},
                    **_FILTER_PROPS,
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="concept_search",
            description=(
                "Concept-anchored search: expand the query through the entity ontology before "
                "retrieving papers. Use when the query mentions a specific concept, entity, "
                "method, or instrument and you want papers about that concept (not just "
                "containing the words)."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["query"],
            },
        ),
        Tool(
            name="get_paper",
            description="Fetch metadata for a single paper by bibcode (title, authors, abstract, citations).",
            inputSchema={
                "type": "object",
                "properties": {"bibcode": {"type": "string"}},
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="read_paper",
            description="Read the full text (or a specific section) of a paper by bibcode.",
            inputSchema={
                "type": "object",
                "properties": {"bibcode": {"type": "string"}, "section": {"type": "string"}},
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="citation_traverse",
            description=(
                "Walk the citation graph from a seed bibcode in either direction. "
                "direction=forward returns papers that cite the seed; backward returns "
                "papers cited by the seed; both returns the union."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bibcode": {"type": "string"},
                    "direction": {"type": "string", "enum": ["forward", "backward", "both"], "default": "both"},
                    "depth": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="citation_similarity",
            description=(
                "Find papers similar to a seed by citation patterns. method=co_citation finds "
                "papers frequently cited together with the seed; bibliographic_coupling finds "
                "papers that share many references with the seed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bibcode": {"type": "string"},
                    "method": {"type": "string", "enum": ["co_citation", "bibliographic_coupling"], "default": "co_citation"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="entity",
            description=(
                "Look up a scientific entity (instrument, mission, dataset, method, object) by "
                "name or QID. Returns canonical name, type, aliases, and Wikidata identifier."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "qid": {"type": "string"},
                    "type": {"type": "string"},
                },
            },
        ),
        Tool(
            name="entity_context",
            description=(
                "Return the co-occurrence neighborhood for an entity: other entities that "
                "frequently appear with it in the same papers, plus a sample of representative papers."
            ),
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["name"],
            },
        ),
        Tool(
            name="graph_context",
            description=(
                "For a free-text query, return the entity and community neighborhood that the "
                "query lands in. Useful for understanding where a topic sits in the corpus."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        Tool(
            name="find_gaps",
            description=(
                "Identify under-explored regions in the literature for a given query. Returns "
                "topic intersections with low paper counts (potential research gaps)."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["query"],
            },
        ),
        Tool(
            name="temporal_evolution",
            description="Time series of paper counts on a topic across years. Returns a year→count distribution.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "year_min": {"type": "integer"}, "year_max": {"type": "integer"}},
                "required": ["query"],
            },
        ),
        Tool(
            name="facet_counts",
            description="For a query, return facet distributions (discipline, year, arxiv_class). Use for aggregate / summary views.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 50}},
                "required": ["query"],
            },
        ),
        Tool(
            name="claim_blame",
            description=(
                "Trace a scientific claim back to the earliest non-retracted paper that asserts it. "
                "Use when given a specific claim and asked who said it first."
            ),
            inputSchema={
                "type": "object",
                "properties": {"claim_id": {"type": "string"}, "claim_text": {"type": "string"}},
            },
        ),
        Tool(
            name="find_replications",
            description="Find papers that replicate, extend, or contradict a seed paper's main claim.",
            inputSchema={
                "type": "object",
                "properties": {"bibcode": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="section_retrieval",
            description=(
                "Section-grain hybrid retrieval over paper bodies. Returns ranked paragraph-level "
                "sections (Methods/Results/Discussion). Use when looking for evidence inside paper bodies, "
                "not just titles/abstracts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    **_FILTER_PROPS,
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_paper_claims",
            description="Return the structured (subject, predicate, object) claims extracted from a paper.",
            inputSchema={
                "type": "object",
                "properties": {"bibcode": {"type": "string"}},
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="find_claims",
            description=(
                "Search across the corpus's structured claim store. Returns matching (claim, paper) "
                "pairs. Use for queries like 'who claims X' or 'find papers asserting Y'."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["query"],
            },
        ),
        Tool(
            name="chunk_search",
            description=(
                "Chunk-grain retrieval over paper full-text via Qdrant. Returns ranked text chunks "
                "(~512 tokens) with bibcode + section + snippet. Use when looking for fine-grained "
                "evidence inside paper bodies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    **_FILTER_PROPS,
                },
                "required": ["query"],
            },
        ),
    ]


def _v1_tools() -> list[Tool]:
    """Proposed 8-tool consolidated surface."""
    return [
        Tool(
            name="search",
            description=(
                "Unified retrieval over the corpus. Pick the grain that matches the query: "
                "'paper' for whole-document retrieval (default), 'concept' for ontology-expanded "
                "concept search, 'section' for paragraph-level evidence inside paper bodies, "
                "'chunk' for fine-grained ~512-token text chunks, 'claim' for structured "
                "(subject, predicate, object) claims. Mode controls the retrieval algorithm; "
                "'hybrid' (semantic + BM25 via RRF) is the best default."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "grain": {"type": "string", "enum": ["paper", "concept", "section", "chunk", "claim"], "default": "paper"},
                    "mode": {"type": "string", "enum": ["hybrid", "semantic", "keyword"], "default": "hybrid"},
                    "limit": {"type": "integer", "default": 10},
                    **_FILTER_PROPS,
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="paper",
            description=(
                "Operations on a single paper, keyed by bibcode. action='metadata' returns "
                "title/authors/abstract/citations (default); 'read' returns full text or a "
                "specific section; 'claims' returns the structured claims extracted from that "
                "paper; 'blame' traces a claim_id back to the earliest non-retracted paper that "
                "asserted it; 'replications' finds papers that replicate/extend/contradict the "
                "seed paper's main claim."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bibcode": {"type": "string"},
                    "action": {"type": "string", "enum": ["metadata", "read", "claims", "blame", "replications"], "default": "metadata"},
                    "section": {"type": "string", "description": "For action=read: restrict to one section"},
                    "claim_id": {"type": "string", "description": "For action=blame: which claim to trace"},
                },
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="citation",
            description=(
                "Citation-graph operations on a seed paper. mode='traverse' walks the citation "
                "graph (direction=forward/backward/both, depth=N); mode='similarity' returns "
                "papers similar by citation patterns (method=co_citation or bibliographic_coupling)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bibcode": {"type": "string"},
                    "mode": {"type": "string", "enum": ["traverse", "similarity"], "default": "traverse"},
                    "direction": {"type": "string", "enum": ["forward", "backward", "both"], "default": "both"},
                    "depth": {"type": "integer", "default": 1},
                    "method": {"type": "string", "enum": ["co_citation", "bibliographic_coupling"], "default": "co_citation"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["bibcode"],
            },
        ),
        Tool(
            name="entity",
            description=(
                "Operations on a scientific entity (instrument, mission, dataset, method, object). "
                "action='lookup' resolves a name/QID to canonical entity (default); 'context' "
                "returns the co-occurrence neighborhood (other entities + sample papers); "
                "'papers' returns papers tagged with that entity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "qid": {"type": "string"},
                    "action": {"type": "string", "enum": ["lookup", "context", "papers"], "default": "lookup"},
                    "type": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="graph_context",
            description=(
                "For a free-text query, return the entity and community neighborhood that the "
                "query lands in. Useful for understanding where a topic sits in the corpus."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        Tool(
            name="find_gaps",
            description=(
                "Identify under-explored regions in the literature for a given query. Returns "
                "topic intersections with low paper counts (potential research gaps)."
            ),
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["query"],
            },
        ),
        Tool(
            name="temporal_evolution",
            description="Time series of paper counts on a topic across years. Returns a year→count distribution.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "year_min": {"type": "integer"}, "year_max": {"type": "integer"}},
                "required": ["query"],
            },
        ),
        Tool(
            name="facet_counts",
            description="For a query, return facet distributions (discipline, year, arxiv_class). Use for aggregate / summary views.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 50}},
                "required": ["query"],
            },
        ),
    ]


def _v2_tools() -> list[Tool]:
    """v1 with descriptions stripped to bare one-liners (description-quality ablation)."""
    terse = {
        "search": "Search the corpus.",
        "paper": "Operations on a single paper by bibcode.",
        "citation": "Citation-graph operations on a paper.",
        "entity": "Look up a scientific entity.",
        "graph_context": "Entity/community context for a query.",
        "find_gaps": "Find under-explored topics.",
        "temporal_evolution": "Topic counts over time.",
        "facet_counts": "Facet distributions for a query.",
    }
    out = []
    for tool in _v1_tools():
        out.append(Tool(name=tool.name, description=terse[tool.name], inputSchema=tool.inputSchema))
    return out


_VARIANTS = {"v0": _v0_tools, "v1": _v1_tools, "v2": _v2_tools}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _make_server(variant: str, log_path: Path) -> Server:
    server = Server(f"scix-stub-{variant}")
    tools = _VARIANTS[variant]()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", buffering=1)  # line-buffered

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        args = arguments or {}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "variant": variant,
            "tool": name,
            "args": args,
        }
        log_fh.write(json.dumps(record) + "\n")
        payload = _stub_payload(name, args)
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server


async def _run(variant: str, log_path: Path) -> None:
    server = _make_server(variant, log_path)
    init_options = InitializationOptions(
        server_name=f"scix-stub-{variant}",
        server_version="0.1.0",
        capabilities=ServerCapabilities(tools={}),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub MCP server for tool-surface eval")
    parser.add_argument("--variant", required=True, choices=list(_VARIANTS))
    parser.add_argument("--log-file", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_run(args.variant, args.log_file))


if __name__ == "__main__":
    main()
