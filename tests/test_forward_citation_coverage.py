"""Exact edge-coverage reporting for ``forward_citations`` intent results."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from scix.citation_contexts_coverage import compute_forward_coverage
from scix.mcp_handlers.citation import _handle_cited_by_intent
from scix.mcp_tool_specs import _TOOL_SPECS


class _Cursor:
    def __init__(self, *, contexts_available: int, total_edges: int) -> None:
        self._coverage = (contexts_available, total_edges)
        self._rows: list[tuple[Any, ...]] = []
        self.description: list[SimpleNamespace] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        if "FROM citation_edges" in sql:
            self._rows = [self._coverage]
            self.description = [
                SimpleNamespace(name="contexts_available"),
                SimpleNamespace(name="total_edges"),
            ]
            return

        self._rows = []
        self.description = [
            SimpleNamespace(name=name)
            for name in (
                "source_bibcode",
                "intent",
                "context_excerpt",
                "title",
                "year",
                "first_author",
                "citation_count",
                "n_contexts",
            )
        ]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor


@pytest.mark.parametrize(
    ("contexts_available", "total_edges", "expected_pct"),
    [(0, 340, 0.0), (1, 340, 1 / 340), (4, 0, 0.0)],
)
def test_compute_forward_coverage_reports_exact_edge_denominator(
    contexts_available: int,
    total_edges: int,
    expected_pct: float,
) -> None:
    conn = _Connection(_Cursor(contexts_available=contexts_available, total_edges=total_edges))

    coverage = compute_forward_coverage(conn, "1996RvMP...68.1259J")

    assert coverage["contexts_available"] == contexts_available
    assert coverage["total_edges"] == total_edges
    assert coverage["coverage_pct"] == pytest.approx(expected_pct)


def test_intent_handler_always_surfaces_exact_forward_coverage() -> None:
    conn = _Connection(_Cursor(contexts_available=0, total_edges=340))

    result = json.loads(
        _handle_cited_by_intent(
            conn,
            {"target_bibcode": "1996RvMP...68.1259J", "intent": "method"},
        )
    )

    assert result["papers"] == []
    assert result["coverage"]["contexts_available"] == 0
    assert result["coverage"]["total_edges"] == 340
    assert result["coverage"]["coverage_pct"] == 0.0


def test_tool_description_documents_sparse_intent_coverage_and_response_counts() -> None:
    spec = next(spec for spec in _TOOL_SPECS if spec["name"] == "forward_citations")
    description = spec["description"].lower()

    assert "~0.27%" in description
    assert "contexts_available" in description
    assert "total_edges" in description
