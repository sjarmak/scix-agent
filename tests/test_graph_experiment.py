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


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


def test_benchmark_jsonl_roundtrip(tmp_path: Path) -> None:
    from scix.graph_experiment.benchmark import Question, read_jsonl, write_jsonl

    questions = [
        Question(
            id="t1_q1",
            tier="one_hop_control",
            prompt="cite this",
            expected_hop_depth=1,
            rubric="rubric body",
            seed_bibcodes=("2024ApJ...1..1A",),
            metadata={"source": "test"},
        )
    ]
    path = tmp_path / "bench.jsonl"
    n = write_jsonl(path, questions)
    assert n == 1
    loaded = read_jsonl(path)
    assert loaded == questions


def test_materialize_templates_skips_when_picker_returns_wrong_count() -> None:
    from scix.graph_experiment.benchmark import (
        BENCHMARK_TEMPLATES,
        materialize_templates,
    )

    def picker(template):
        # always return one bibcode regardless of template seed_count
        return ("2024ApJ...x..1A",)

    out = materialize_templates(BENCHMARK_TEMPLATES, picker)
    # seed_count==0 templates always materialize. seed_count==1 templates
    # also resolve. seed_count >= 2 templates are skipped because the
    # picker returns only 1.
    template_ids = {q.id for q in out}
    one_or_zero = {
        t.template_id for t in BENCHMARK_TEMPLATES if t.seed_count in (0, 1)
    }
    multi = {t.template_id for t in BENCHMARK_TEMPLATES if t.seed_count >= 2}
    assert template_ids == one_or_zero
    assert template_ids.isdisjoint(multi)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def _write_trace(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def test_summarize_traces_aggregates_tool_calls_and_depth(tmp_path: Path) -> None:
    from scix.graph_experiment.analysis import summarize_traces

    path = tmp_path / "trace_a.jsonl"
    _write_trace(
        path,
        [
            {
                "event_id": "1",
                "session_id": "s1",
                "tool_name": "shortest_path",
                "args": {"source_bibcode": "X", "target_bibcode": "Y"},
                "duration_ms": 10.0,
                "ok": True,
            },
            {
                "event_id": "2",
                "session_id": "s1",
                "tool_name": "multi_hop_neighbors",
                "args": {"bibcode": "X", "depth": 3},
                "duration_ms": 20.0,
                "ok": True,
            },
            {
                "event_id": "3",
                "session_id": "s1",
                "tool_name": "pattern_match",
                "args": {"head_bibcode": "X", "edge_sequence": ["out", "in"]},
                "duration_ms": 30.0,
                "ok": False,
                "error": "test",
            },
            {
                "event_id": "4",
                "session_id": "s1",
                "tool_name": "graph_query_log",
                "args": {"cypher_or_intent": "MATCH ..."},
                "duration_ms": 1.0,
                "ok": True,
            },
        ],
    )

    summary = summarize_traces([path])
    assert summary.event_count == 4
    assert summary.tool_call_counts["shortest_path"] == 1
    assert summary.error_counts["pattern_match"] == 1
    assert summary.max_depth == 3
    assert summary.depth_distribution[3] == 1  # multi_hop_neighbors depth=3
    assert summary.depth_distribution[2] == 1  # pattern_match len(edge_sequence)=2
    assert summary.depth_distribution[1] == 1  # shortest_path => 1
    assert summary.depth_distribution[0] == 1  # graph_query_log => 0
    assert len(summary.freeform_queries) == 1


def test_compare_summaries_reports_depth_shift(tmp_path: Path) -> None:
    from scix.graph_experiment.analysis import compare_summaries, summarize_traces

    control_path = tmp_path / "ctrl.jsonl"
    treatment_path = tmp_path / "trt.jsonl"
    _write_trace(
        control_path,
        [
            {
                "event_id": "c1",
                "session_id": "ctrl",
                "tool_name": "search",
                "args": {"query": "x"},
                "duration_ms": 5.0,
                "ok": True,
            }
        ],
    )
    _write_trace(
        treatment_path,
        [
            {
                "event_id": "t1",
                "session_id": "trt",
                "tool_name": "multi_hop_neighbors",
                "args": {"bibcode": "X", "depth": 4},
                "duration_ms": 12.0,
                "ok": True,
            },
            {
                "event_id": "t2",
                "session_id": "trt",
                "tool_name": "graph_query_log",
                "args": {"cypher_or_intent": "..."},
                "duration_ms": 1.0,
                "ok": True,
            },
        ],
    )
    cmp = compare_summaries(
        summarize_traces([control_path]), summarize_traces([treatment_path])
    )
    assert cmp["depth_shift_max"] == 4
    assert cmp["freeform_query_emergence"] == 1
    assert "multi_hop_neighbors" in cmp["new_tool_usage"]


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def test_harness_config_control_omits_experimental_server(tmp_path: Path) -> None:
    from scix.graph_experiment.harness import HarnessConfig

    cfg = HarnessConfig(
        snapshot_path=tmp_path / "snap.pkl.gz",
        trace_dir=tmp_path / "traces",
        production_mcp_url="https://mcp.example.com/mcp/",
        production_mcp_token="fake-token-123",
    )
    config = cfg.mcp_config_for("control", "sess-1")
    servers = config["mcpServers"]
    assert "scix" in servers
    assert "scix-graph-experiment" not in servers
    assert servers["scix"]["headers"]["Authorization"] == "Bearer fake-token-123"


def test_harness_config_treatment_includes_experimental_server(tmp_path: Path) -> None:
    from scix.graph_experiment.harness import HarnessConfig

    cfg = HarnessConfig(
        snapshot_path=tmp_path / "snap.pkl.gz",
        trace_dir=tmp_path / "traces",
    )
    config = cfg.mcp_config_for("treatment", "sess-2")
    servers = config["mcpServers"]
    assert "scix-graph-experiment" in servers
    exp = servers["scix-graph-experiment"]
    assert exp["type"] == "stdio"
    assert exp["env"]["SCIX_GRAPH_EXP_SESSION"] == "sess-2"
    assert exp["env"]["SCIX_GRAPH_EXP_SNAPSHOT"] == str(cfg.snapshot_path)


def test_parse_stream_json_extracts_answer_and_cost() -> None:
    from scix.graph_experiment.harness import _parse_stream_json

    stream = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello, "},
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "world."},
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "total_cost_usd": 0.0125,
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                }
            ),
        ]
    )
    parsed = _parse_stream_json(stream)
    assert parsed["final_answer"] == "Hello, \nworld."
    assert parsed["cost_usd"] == 0.0125
    assert parsed["total_tokens"] == 1200
    assert parsed["error"] is None


def test_parse_stream_json_handles_malformed_lines() -> None:
    from scix.graph_experiment.harness import _parse_stream_json

    stream = (
        "not json\n"
        + json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}})
        + "\n"
    )
    parsed = _parse_stream_json(stream)
    assert parsed["final_answer"] == "ok"
