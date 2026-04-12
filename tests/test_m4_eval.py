"""Tests for PRD §M4 three-way eval (u12)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scix import resolve_entities as re_mod
from scix.eval.metrics import (
    INHOUSE_DISCLAIMER,
    GraphWalkTask,
    QueryFixture,
    ThreeWayConfig,
    ThreeWayResults,
    format_m4_report,
    run_three_way_eval,
)


@pytest.fixture(autouse=True)
def _reset_resolver_mocks():
    re_mod._reset_mocks()
    yield
    re_mod._reset_mocks()


def _fixture_queries() -> list[QueryFixture]:
    return [
        QueryFixture(
            query_id="q_a",
            query_text="test query a",
            relevant_bibcodes=frozenset({"A1", "A2"}),
        ),
        QueryFixture(
            query_id="q_b",
            query_text="test query b",
            relevant_bibcodes=frozenset({"B1"}),
        ),
    ]


def _fixture_tasks() -> list[GraphWalkTask]:
    return [
        GraphWalkTask(
            task_id="gw_a",
            seed_bibcode="A1",
            expected_bibcodes=frozenset({"A2"}),
        ),
    ]


def _make_configs() -> list[ThreeWayConfig]:
    def baseline(fixture):
        orderings = {
            "q_a": ["A1", "NOISE", "A2"],
            "q_b": ["NOISE", "B1"],
            "gw_a": ["A1", "NOISE", "A2"],
        }
        return orderings[fixture.query_id], 1.0

    def with_static(fixture):
        orderings = {
            "q_a": ["A1", "A2", "NOISE"],
            "q_b": ["B1", "NOISE"],
            "gw_a": ["A1", "A2", "NOISE"],
        }
        return orderings[fixture.query_id], 1.5

    def with_jit(fixture):
        orderings = {
            "q_a": ["A2", "A1", "NOISE"],
            "q_b": ["B1", "NOISE"],
            "gw_a": ["A2", "A1", "NOISE"],
        }
        return orderings[fixture.query_id], 2.0

    return [
        ThreeWayConfig(name="hybrid_baseline", description="baseline", retrieve=baseline),
        ThreeWayConfig(name="hybrid_plus_static", description="static", retrieve=with_static),
        ThreeWayConfig(name="hybrid_plus_jit", description="jit", retrieve=with_jit),
    ]


def test_run_three_way_eval_returns_three_configs():
    results = run_three_way_eval(_fixture_queries(), _fixture_tasks(), _make_configs())
    assert isinstance(results, ThreeWayResults)
    assert set(results.query_reports) == {
        "hybrid_baseline",
        "hybrid_plus_static",
        "hybrid_plus_jit",
    }
    assert set(results.graph_walk_reports) == {
        "hybrid_baseline",
        "hybrid_plus_static",
        "hybrid_plus_jit",
    }


def test_three_way_eval_rejects_non_three_configs():
    with pytest.raises(ValueError):
        run_three_way_eval(_fixture_queries(), _fixture_tasks(), _make_configs()[:2])


def test_static_and_jit_outperform_baseline_on_fixture():
    results = run_three_way_eval(_fixture_queries(), _fixture_tasks(), _make_configs())
    baseline = results.query_reports["hybrid_baseline"].mean_ndcg_at_10
    static = results.query_reports["hybrid_plus_static"].mean_ndcg_at_10
    jit = results.query_reports["hybrid_plus_jit"].mean_ndcg_at_10
    assert static >= baseline
    assert jit >= baseline


def test_format_m4_report_includes_disclaimer_at_top():
    results = run_three_way_eval(_fixture_queries(), _fixture_tasks(), _make_configs())
    report = format_m4_report(
        results=results,
        configs=_make_configs(),
        n_queries=len(_fixture_queries()),
        n_graph_walk=len(_fixture_tasks()),
        timestamp="2026-04-12 00:00:00",
    )
    assert report.startswith(INHOUSE_DISCLAIMER)
    # All three configs must appear in the report body.
    assert "hybrid_baseline" in report
    assert "hybrid_plus_static" in report
    assert "hybrid_plus_jit" in report
    # nDCG@10 column header present.
    assert "nDCG@10" in report
    assert "Recall@20" in report
    assert "MRR" in report


def test_eval_three_way_script_writes_report(tmp_path):
    import eval_three_way

    output = tmp_path / "m4_inhouse_eval.md"
    results = eval_three_way.run(output_path=output)
    assert output.exists()
    content = output.read_text()
    assert INHOUSE_DISCLAIMER.splitlines()[0] in content
    assert "hybrid_baseline" in content
    assert "hybrid_plus_static" in content
    assert "hybrid_plus_jit" in content
    # Three configs, so three reports each side.
    assert len(results.query_reports) == 3
    assert len(results.graph_walk_reports) == 3
    # Entity-enrichment configs should not hurt the fixture.
    baseline = results.query_reports["hybrid_baseline"].mean_ndcg_at_10
    jit = results.query_reports["hybrid_plus_jit"].mean_ndcg_at_10
    assert jit >= baseline
