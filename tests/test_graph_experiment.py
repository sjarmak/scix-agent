"""Tests for the graph-experiment spike (bead vdtd).

Loader is integration-tested separately (it needs a live DB); here we cover
the pure-Python pieces against a tiny synthetic igraph.
"""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import pytest

from scix.graph_experiment import graph_tools
from scix.graph_experiment.slice_config import SliceConfig
from scix.graph_experiment.trace import TraceLogger


def _toy_graph() -> ig.Graph:
    """5-node DAG: A -> B -> C, A -> D, D -> C, E orphan."""
    g = ig.Graph(directed=True)
    g.add_vertices(5)
    g.vs["name"] = ["A", "B", "C", "D", "E"]
    g.vs["title"] = ["alpha", "beta", "gamma", "delta", "epsilon"]
    g.vs["year"] = [2020, 2021, 2022, 2021, 2019]
    g.vs["citation_count"] = [10, 5, 100, 3, 0]
    g.add_edges([(0, 1), (1, 2), (0, 3), (3, 2)])
    return g


# --------------------------------------------------------------------------
# slice_config
# --------------------------------------------------------------------------


def test_slice_config_astronomy_1hop_defaults() -> None:
    c = SliceConfig.astronomy_1hop()
    assert c.name == "astronomy_1hop"
    assert c.hop_depth == 1
    assert "astronomy" in c.seed_filter_sql
    assert c.snapshot_path == Path("data/graph_experiment/astronomy_1hop.pkl.gz")


def test_slice_config_is_frozen() -> None:
    c = SliceConfig.astronomy_1hop()
    with pytest.raises(Exception):
        c.name = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------


def test_trace_logger_appends_jsonl(tmp_path: Path) -> None:
    tl = TraceLogger(tmp_path, session_id="t1")
    with tl.record("test_tool", {"q": "hi"}) as ev:
        ev.summarize(rows=2)
    line = tl.path.read_text().strip()
    payload = json.loads(line)
    assert payload["tool_name"] == "test_tool"
    assert payload["args"] == {"q": "hi"}
    assert payload["ok"] is True
    assert payload["result_summary"] == {"rows": 2}
    assert payload["session_id"] == "t1"


def test_trace_logger_records_exceptions(tmp_path: Path) -> None:
    tl = TraceLogger(tmp_path, session_id="t2")
    with pytest.raises(ValueError):
        with tl.record("boom", {}) as _:
            raise ValueError("kaboom")
    payload = json.loads(tl.path.read_text().strip())
    assert payload["ok"] is False
    assert "kaboom" in payload["error"]


# --------------------------------------------------------------------------
# graph_tools
# --------------------------------------------------------------------------


def test_shortest_path_finds_two_hop_path() -> None:
    g = _toy_graph()
    result = graph_tools.shortest_path(g, "A", "C", mode="out")
    assert result["found"] is True
    assert result["length"] == 2
    assert [v["bibcode"] for v in result["path"]] == ["A", "B", "C"]


def test_shortest_path_unreachable_returns_not_found() -> None:
    g = _toy_graph()
    result = graph_tools.shortest_path(g, "C", "A", mode="out")
    assert result["found"] is False


def test_shortest_path_unknown_bibcode() -> None:
    g = _toy_graph()
    result = graph_tools.shortest_path(g, "A", "Z", mode="out")
    assert result["error"] == "unknown_bibcode"
    assert result["missing"] == ["Z"]


def test_subgraph_around_includes_seeds_and_neighbors() -> None:
    g = _toy_graph()
    result = graph_tools.subgraph_around(g, ["A"], hops=1, max_nodes=10)
    bibcodes = {n["bibcode"] for n in result["nodes"]}
    assert {"A", "B", "D"}.issubset(bibcodes)
    assert result["edge_count"] >= 2


def test_subgraph_around_respects_max_nodes_but_keeps_seeds() -> None:
    g = _toy_graph()
    result = graph_tools.subgraph_around(g, ["A", "E"], hops=2, max_nodes=2)
    bibcodes = {n["bibcode"] for n in result["nodes"]}
    # Seeds are always retained even when max_nodes would otherwise drop them.
    assert {"A", "E"}.issubset(bibcodes)


def test_personalized_pagerank_excludes_seeds_and_returns_topk() -> None:
    g = _toy_graph()
    result = graph_tools.personalized_pagerank(g, ["A"], top_k=3)
    bibs = [r["bibcode"] for r in result["results"]]
    assert "A" not in bibs
    assert len(result["results"]) <= 3
    assert all("score" in r for r in result["results"])


def test_multi_hop_neighbors_annotates_hop_distance() -> None:
    g = _toy_graph()
    result = graph_tools.multi_hop_neighbors(g, "A", depth=2, mode="out", max_results=10)
    by_bibcode = {r["bibcode"]: r["hop"] for r in result["results"]}
    assert by_bibcode["B"] == 1
    assert by_bibcode["D"] == 1
    assert by_bibcode["C"] == 2


def test_pattern_match_two_hop_forward() -> None:
    g = _toy_graph()
    result = graph_tools.pattern_match(g, "A", ["out", "out"])
    bibs = [r["bibcode"] for r in result["results"]]
    assert "C" in bibs


def test_pattern_match_co_citation_pattern() -> None:
    """A out -> B,D ; B in -> A ; D in -> A. So A out->in finds A itself,
    excluded. Add another paper E that also cites B to test co-citation."""
    g = _toy_graph()
    g.add_vertex(name="F", title="phi", year=2023, citation_count=2)
    g.add_edges([(g.vs.find(name="F").index, g.vs.find(name="B").index)])
    result = graph_tools.pattern_match(g, "A", ["out", "in"])
    bibs = [r["bibcode"] for r in result["results"]]
    assert "F" in bibs
    assert "A" not in bibs


def test_pattern_match_rejects_bad_direction() -> None:
    g = _toy_graph()
    result = graph_tools.pattern_match(g, "A", ["sideways"])  # type: ignore[list-item]
    assert result["error"] == "bad_edge_direction"


def test_graph_query_log_does_not_execute() -> None:
    result = graph_tools.graph_query_log(
        "MATCH (p:Paper)-[:CITES*1..3]->(q) RETURN p, q",
        notes="testing",
    )
    assert result["executed"] is False
    assert result["logged"] is True
    assert "MATCH" in result["echo"]
    assert result["notes"] == "testing"
