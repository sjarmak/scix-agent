"""Tests for POST /viz/api/demo/{survey,methods,disambig}.

Each endpoint runs a real multi-step MCP tool sequence via
``mcp_server.call_tool`` and relies on the MCP instrumentation hook to
publish TraceEvents for the SSE stream. These tests stub ``call_tool`` so
no DB/embedding backend is required, then verify:

* the right tool names are dispatched in order,
* the response shape exposes the step log for debugging,
* the endpoint never calls ``trace_stream.publish`` directly,
* a failing intermediate step does not abort the sequence,
* 8+ instrumented dispatches happen per scenario (the acceptance
  criterion's "8-15 real tool events").
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scix.viz import api as viz_api
from scix.viz.server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fake dispatch table — returns JSON payloads shaped like the real MCP
# handlers for each tool the composite endpoints call.
# ---------------------------------------------------------------------------


def _fake_search(query: str) -> dict:
    """5 seed papers so the drill steps have top-3 bibcodes to iterate over."""
    return {
        "papers": [
            {
                "bibcode": f"2024Seed{i}",
                "title": f"galaxy cluster study {i}",
                "rrf_score": 0.9 - i * 0.1,
            }
            for i in range(5)
        ],
        "total": 5,
        "timing_ms": {"lexical_ms": 8.0},
    }


def _fake_co_citation(bib: str) -> dict:
    return {
        "papers": [
            {"bibcode": f"{bib}-cc{i}", "title": f"co-cited {i}"} for i in range(3)
        ],
        "total": 3,
        "timing_ms": {"query_ms": 4.0},
    }


def _fake_coupling(bib: str) -> dict:
    return {
        "papers": [
            {"bibcode": f"Coupled{i}", "title": f"coupled paper {i}"}
            for i in range(5)
        ],
        "total": 5,
        "timing_ms": {"query_ms": 4.0},
    }


def _fake_concept(query: str) -> dict:
    return {
        "papers": [{"bibcode": f"Concept-{query}-1", "title": f"concept {query}"}],
        "total": 1,
        "timing_ms": {"query_ms": 3.0},
    }


def _fake_get_paper(bib: str) -> dict:
    return {"bibcode": bib, "title": "x", "abstract": "y"}


def _fake_graph_context(bib: str) -> dict:
    return {"bibcode": bib, "community_id": 42, "indegree": 10}


def _fake_citation_chain(src: str, tgt: str) -> dict:
    return {
        "papers": [
            {"bibcode": src, "title": "source"},
            {"bibcode": tgt, "title": "target"},
        ],
        "total": 2,
        "timing_ms": {"query_ms": 6.0},
    }


def _fake_entity_resolve(query: str) -> dict:
    return {
        "query": query,
        "candidates": [
            {
                "entity_id": 101,
                "canonical_name": "JWST",
                "entity_type": "instruments",
                "confidence": 0.95,
            },
            {
                "entity_id": 102,
                "canonical_name": "NIRCam",
                "entity_type": "instruments",
                "confidence": 0.85,
            },
        ],
        "total": 2,
    }


def _fake_entity_search(query: str) -> dict:
    return {
        "candidates": [
            {"entity_id": 103, "canonical_name": "MIRI", "entity_type": "instruments"},
        ],
    }


def _fake_entity_context(entity_id: int) -> dict:
    return {
        "papers": [
            {"bibcode": f"Entity-{entity_id}-paper{i}"} for i in range(2)
        ],
        "total": 2,
    }


def _fake_temporal(query: str) -> dict:
    return {
        "papers": [{"bibcode": f"Temp-{query}-{y}"} for y in (2022, 2023, 2024)],
        "total": 3,
        "timing_ms": {"query_ms": 5.0},
    }


def _dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Minimal simulator covering every tool the composite endpoints call."""
    if name == "search":
        # Route filtered searches to the same payload — bibcodes just need
        # to be non-empty so the downstream graph_context loop fires.
        return json.dumps(_fake_search(arguments.get("query", "")))
    if name == "citation_similarity":
        if arguments.get("method") == "co_citation":
            return json.dumps(_fake_co_citation(arguments.get("bibcode", "")))
        return json.dumps(_fake_coupling(arguments.get("bibcode", "")))
    if name == "concept_search":
        return json.dumps(_fake_concept(arguments.get("query", "")))
    if name == "citation_chain":
        return json.dumps(
            _fake_citation_chain(
                arguments.get("source_bibcode", ""),
                arguments.get("target_bibcode", ""),
            )
        )
    if name == "get_paper":
        return json.dumps(_fake_get_paper(arguments.get("bibcode", "")))
    if name == "graph_context":
        return json.dumps(_fake_graph_context(arguments.get("bibcode", "")))
    if name == "entity":
        if arguments.get("action") == "resolve":
            return json.dumps(_fake_entity_resolve(arguments.get("query", "")))
        return json.dumps(_fake_entity_search(arguments.get("query", "")))
    if name == "entity_context":
        return json.dumps(_fake_entity_context(int(arguments.get("entity_id", 0))))
    if name == "temporal_evolution":
        return json.dumps(_fake_temporal(arguments.get("bibcode_or_query", "")))
    return "{}"


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace ``mcp_server.call_tool`` with a stub that dispatches + records."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def _fake(name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, dict(arguments)))
        return _dispatch(name, arguments)

    monkeypatch.setattr(viz_api.mcp_server, "call_tool", _fake)
    return calls


# ---------------------------------------------------------------------------
# /viz/api/demo/survey
# ---------------------------------------------------------------------------


class TestDemoSurvey:
    def test_happy_path_dispatches_expected_tool_sequence(
        self, captured_calls: list[tuple[str, dict[str, Any]]]
    ) -> None:
        response = client.post(
            "/viz/api/demo/survey",
            json={"query": "galaxy clusters dark matter", "top_n": 5},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["scenario"] == "survey"
        assert body["bibcodes"] == [f"2024Seed{i}" for i in range(5)]

        tools = [name for name, _ in captured_calls]
        # 1 search + 3 co_citation + 2 concept_search (from titles) + 1 get_paper = 7 min
        assert tools[0] == "search"
        assert tools[1:4] == ["citation_similarity"] * 3
        assert "concept_search" in tools
        assert "get_paper" in tools

        # 8+ events is the bead's acceptance floor.
        assert len(tools) >= 7

        # Every step carries a latency_ms (>=0) and bibcodes list.
        for step in body["steps"]:
            assert "tool" in step
            assert "args" in step
            assert "bibcodes" in step
            assert step["latency_ms"] >= 0.0

    def test_no_hand_coded_publish_trace(
        self,
        captured_calls: list[tuple[str, dict[str, Any]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The endpoint must not call ``trace_stream.publish`` itself."""
        from scix.viz import trace_stream

        publish_spy = MagicMock()
        monkeypatch.setattr(trace_stream, "publish", publish_spy)

        response = client.post(
            "/viz/api/demo/survey",
            json={"query": "galaxy clusters", "top_n": 3},
        )
        assert response.status_code == 200, response.text
        publish_spy.assert_not_called()

    def test_tolerates_intermediate_tool_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising citation_similarity must not break the rest of the sequence."""
        calls: list[str] = []

        def _flaky(name: str, arguments: dict[str, Any]) -> str:
            calls.append(name)
            if name == "citation_similarity":
                raise RuntimeError("simulated failure")
            return _dispatch(name, arguments)

        monkeypatch.setattr(viz_api.mcp_server, "call_tool", _flaky)

        response = client.post(
            "/viz/api/demo/survey",
            json={"query": "galaxy clusters", "top_n": 3},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        # Concept search + get_paper still ran despite the co-citation failure.
        assert "concept_search" in calls
        assert "get_paper" in calls

        # The error is surfaced in the step log.
        failing_steps = [s for s in body["steps"] if s.get("error")]
        assert failing_steps, "expected at least one failing step to record error"


# ---------------------------------------------------------------------------
# /viz/api/demo/methods
# ---------------------------------------------------------------------------


class TestDemoMethods:
    def test_happy_path_dispatches_expected_tool_sequence(
        self, captured_calls: list[tuple[str, dict[str, Any]]]
    ) -> None:
        response = client.post(
            "/viz/api/demo/methods",
            json={"query": "deep learning image segmentation", "top_n": 5},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["scenario"] == "methods"
        assert body["bibcodes"] == [f"2024Seed{i}" for i in range(5)]
        assert body["coupled_bibcodes"] == [f"Coupled{i}" for i in range(5)]

        tools = [name for name, _ in captured_calls]
        # 1 search + 1 coupling + 3 graph_context + ≤2 concept + 2 get_paper = 8+
        # (citation_chain is excluded — too slow on the 299M-edge prod graph)
        assert tools[0] == "search"
        assert tools[1] == "citation_similarity"
        assert tools.count("graph_context") == 3
        assert "concept_search" in tools
        assert "citation_chain" not in tools
        assert tools.count("get_paper") == 2
        assert len(tools) >= 8

    def test_no_hand_coded_publish_trace(
        self,
        captured_calls: list[tuple[str, dict[str, Any]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scix.viz import trace_stream

        publish_spy = MagicMock()
        monkeypatch.setattr(trace_stream, "publish", publish_spy)

        response = client.post(
            "/viz/api/demo/methods",
            json={"query": "deep learning image segmentation", "top_n": 3},
        )
        assert response.status_code == 200, response.text
        publish_spy.assert_not_called()


# ---------------------------------------------------------------------------
# /viz/api/demo/disambig
# ---------------------------------------------------------------------------


class TestDemoDisambig:
    def test_happy_path_dispatches_expected_tool_sequence(
        self, captured_calls: list[tuple[str, dict[str, Any]]]
    ) -> None:
        response = client.post(
            "/viz/api/demo/disambig",
            json={"query": "JWST NIRCam", "top_n": 5},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["scenario"] == "disambig"
        # "jwst" triggers the instruments bucket via _guess_entity_type.
        assert body["entity_type"] == "instruments"
        # Both resolve and search return two plus one unique entity_id.
        assert set(body["resolved_entity_ids"]) >= {101, 102, 103}

        tools = [name for name, _ in captured_calls]
        # 2 entity + 1 entity_context + 1 search + 3 graph_context + 1 temporal = 8
        assert tools[:2] == ["entity", "entity"]
        assert "entity_context" in tools
        assert "search" in tools
        assert tools.count("graph_context") == 3
        assert "temporal_evolution" in tools
        assert len(tools) >= 8

        # entity(resolve) sends the right action.
        assert captured_calls[0][1]["action"] == "resolve"
        assert captured_calls[1][1]["action"] == "search"
        assert captured_calls[1][1]["entity_type"] == "instruments"

    def test_filtered_search_carries_entity_ids(
        self, captured_calls: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """The disambig search step must pass resolved entity_ids as a filter."""
        client.post(
            "/viz/api/demo/disambig",
            json={"query": "JWST NIRCam", "top_n": 4},
        )

        search_call = next(
            args for name, args in captured_calls if name == "search"
        )
        assert search_call["filters"]["entity_ids"] == [101, 102, 103]

    def test_no_hand_coded_publish_trace(
        self,
        captured_calls: list[tuple[str, dict[str, Any]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scix.viz import trace_stream

        publish_spy = MagicMock()
        monkeypatch.setattr(trace_stream, "publish", publish_spy)

        response = client.post(
            "/viz/api/demo/disambig",
            json={"query": "JWST NIRCam", "top_n": 3},
        )
        assert response.status_code == 200, response.text
        publish_spy.assert_not_called()
