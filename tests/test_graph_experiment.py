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


def test_harness_config_stdio_production_runs_mcp_server_via_python(
    tmp_path: Path,
) -> None:
    from scix.graph_experiment.harness import HarnessConfig

    cfg = HarnessConfig(
        snapshot_path=tmp_path / "snap.pkl.gz",
        trace_dir=tmp_path / "traces",
        production_mcp_stdio=True,
    )
    config = cfg.mcp_config_for("control", "sess-stdio")
    servers = config["mcpServers"]
    assert servers["scix"]["type"] == "stdio"
    assert servers["scix"]["args"] == ["-m", "scix.mcp_server"]
    # treatment-only server is not present in control variant
    assert "scix-graph-experiment" not in servers


def test_harness_config_url_takes_precedence_over_stdio(tmp_path: Path) -> None:
    from scix.graph_experiment.harness import HarnessConfig

    cfg = HarnessConfig(
        snapshot_path=tmp_path / "snap.pkl.gz",
        trace_dir=tmp_path / "traces",
        production_mcp_url="https://example.com/mcp/",
        production_mcp_stdio=True,  # ignored when url is set
    )
    servers = cfg.mcp_config_for("control", "sess-x")["mcpServers"]
    assert servers["scix"]["type"] == "http"


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


# --------------------------------------------------------------------------
# bench_runner: pickers, materialization, verdict policy
# --------------------------------------------------------------------------


def _bench_graph() -> ig.Graph:
    """Larger toy graph for bench_runner tests.

    8 nodes with varied citation counts/years to exercise picker filters.
    Triangle A-B-C used by t2_ppr_relevance.
    """
    g = ig.Graph(directed=True)
    g.add_vertices(8)
    g.vs["name"] = ["A", "B", "C", "D", "E", "F", "G", "H"]
    g.vs["title"] = [
        "core method paper",
        "applied algorithm",
        "exoplanet survey technique",
        "review of dark matter",
        "old result",
        "recent finding",
        "obscure note",
        "mid-cited paper",
    ]
    g.vs["year"] = [2015, 2018, 2021, 2010, 2012, 2024, 2019, 2022]
    g.vs["citation_count"] = [500, 250, 150, 120, 80, 30, 5, 60]
    # triangle A-B-C, plus assorted edges
    g.add_edges(
        [(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (5, 0), (7, 5), (6, 7)]
    )
    return g


def test_pick_top_cited_orders_and_filters() -> None:
    from scix.graph_experiment.bench_runner import pick_top_cited

    g = _bench_graph()
    ids = pick_top_cited(g, min_count=100)
    counts = [g.vs[i]["citation_count"] for i in ids]
    assert counts == sorted(counts, reverse=True)
    assert all(c >= 100 for c in counts)


def test_pick_top_cited_year_window() -> None:
    from scix.graph_experiment.bench_runner import pick_top_cited

    g = _bench_graph()
    ids = pick_top_cited(g, min_count=50, max_year=2018)
    names = {g.vs[i]["name"] for i in ids}
    # A(2015,500), B(2018,250), D(2010,120), E(2012,80) all qualify
    assert names == {"A", "B", "D", "E"}


def test_pick_top_cited_title_contains() -> None:
    from scix.graph_experiment.bench_runner import pick_top_cited

    g = _bench_graph()
    ids = pick_top_cited(
        g, min_count=100, title_contains=("method", "algorithm", "technique")
    )
    names = {g.vs[i]["name"] for i in ids}
    # A(method), B(algorithm), C(technique) — D's title has no match
    assert names == {"A", "B", "C"}


def test_pick_top_cited_handles_missing_attrs() -> None:
    from scix.graph_experiment.bench_runner import pick_top_cited

    g = _bench_graph()
    g.vs[3]["year"] = None
    g.vs[2]["citation_count"] = None
    ids = pick_top_cited(g, min_year=2000, min_count=10)
    names = {g.vs[i]["name"] for i in ids}
    assert "C" not in names  # citation_count is None
    assert "D" not in names  # year is None when min_year is set


def test_materialize_questions_resolves_against_real_graph() -> None:
    from scix.graph_experiment.bench_runner import materialize_questions

    g = _bench_graph()
    questions = materialize_questions(g)
    ids = {q.id for q in questions}
    # Tier-1 templates that need any cited paper should always resolve.
    assert "t1_direct_citations" in ids
    assert "t1_topical_search" in ids  # seed_count == 0
    # All resolved seeds must be names actually present in the graph.
    valid_names = set(g.vs["name"])
    for q in questions:
        for bib in q.seed_bibcodes:
            assert bib in valid_names


def test_materialize_questions_finds_triangle_for_ppr() -> None:
    from scix.graph_experiment.bench_runner import materialize_questions

    g = _bench_graph()
    questions = {q.id: q for q in materialize_questions(g)}
    ppr = questions.get("t2_ppr_relevance")
    assert ppr is not None
    assert len(ppr.seed_bibcodes) == 3
    # Graph has triangle A-B-C; picker should locate it.
    assert set(ppr.seed_bibcodes) == {"A", "B", "C"}


def test_go_no_go_recommends_go_when_depth_lifts_and_freeform_fires() -> None:
    from scix.graph_experiment.bench_runner import go_no_go

    verdict, rationale = go_no_go(
        {
            "depth_shift_max": 3,
            "depth_shift_median": 1.5,
            "freeform_query_emergence": 4,
            "new_tool_usage": {"shortest_path": 5, "multi_hop_neighbors": 2},
        }
    )
    assert verdict == "GO"
    assert "Apache AGE" in rationale


def test_go_no_go_inconclusive_when_freeform_is_zero() -> None:
    from scix.graph_experiment.bench_runner import go_no_go

    verdict, _ = go_no_go(
        {
            "depth_shift_max": 1,
            "depth_shift_median": 0.5,
            "freeform_query_emergence": 0,
            "new_tool_usage": {},
        }
    )
    assert verdict == "INCONCLUSIVE"


def test_go_no_go_no_go_when_depth_does_not_lift() -> None:
    from scix.graph_experiment.bench_runner import go_no_go

    verdict, rationale = go_no_go(
        {
            "depth_shift_max": 0,
            "depth_shift_median": 0.0,
            "freeform_query_emergence": 0,
            "new_tool_usage": {},
        }
    )
    assert verdict == "NO-GO"
    assert "not justified" in rationale


def test_render_markdown_includes_verdict_and_per_question_rows() -> None:
    from scix.graph_experiment.bench_runner import render_markdown
    from scix.graph_experiment.benchmark import Question
    from scix.graph_experiment.harness import RunResult

    questions = [
        Question(
            id="q1",
            tier="one_hop_control",
            prompt="?",
            expected_hop_depth=1,
            rubric="r",
            seed_bibcodes=("A",),
        )
    ]
    results = [
        RunResult(
            question_id="q1",
            variant="control",
            session_id="s1",
            final_answer="hi",
            duration_seconds=1.0,
            cost_usd=0.01,
            total_tokens=100,
            trace_path=None,
            raw_event_count=1,
        ),
        RunResult(
            question_id="q1",
            variant="treatment",
            session_id="s2",
            final_answer="hi",
            duration_seconds=1.0,
            cost_usd=0.02,
            total_tokens=120,
            trace_path=None,
            raw_event_count=2,
        ),
    ]
    comparison = {
        "depth_shift_max": 2,
        "depth_shift_median": 1.0,
        "freeform_query_emergence": 1,
        "new_tool_usage": {"shortest_path": 1},
        "control_summary": {
            "event_count": 1,
            "max_depth": 0,
            "median_depth": 0.0,
        },
        "treatment_summary": {
            "event_count": 2,
            "max_depth": 2,
            "median_depth": 1.0,
        },
    }
    md = render_markdown(
        verdict="GO",
        rationale="ok",
        comparison=comparison,
        questions=questions,
        results=results,
    )
    assert "**Verdict:** GO" in md
    assert "`q1`" in md
    assert "$0.0100" in md
    assert "$0.0200" in md
