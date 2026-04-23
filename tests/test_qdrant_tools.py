"""Unit tests for scix.qdrant_tools feature-flagging and MCP integration.

These tests do NOT require a running Qdrant instance. They validate:
  1. is_enabled() correctly reads the QDRANT_URL env var.
  2. bibcode_to_point_id is deterministic and within the int64 range Qdrant accepts.
  3. The MCP server registers/omits the find_similar_by_examples tool
     based on the feature flag, and the self-test passes in both modes.
  4. The dispatch handler returns a structured error when Qdrant is disabled.

An integration test that hits a live Qdrant server is provided separately
(see ``test_qdrant_tools_integration.py``, skipped unless QDRANT_URL is set).
"""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from scix import qdrant_tools
from scix import mcp_server


@pytest.fixture
def no_qdrant(monkeypatch: pytest.MonkeyPatch):
    """Ensure QDRANT_URL is unset for disabled-path tests."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield


@pytest.fixture
def fake_qdrant(monkeypatch: pytest.MonkeyPatch):
    """Set QDRANT_URL to a non-routable value — enabled flag on, but no real server."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:59999")
    yield


class TestIsEnabled:
    def test_disabled_when_env_unset(self, no_qdrant):
        assert qdrant_tools.is_enabled() is False
        assert mcp_server._qdrant_enabled() is False

    def test_enabled_when_env_set(self, fake_qdrant):
        assert qdrant_tools.is_enabled() is True
        assert mcp_server._qdrant_enabled() is True


class TestBibcodeToPointId:
    def test_deterministic(self):
        a = qdrant_tools.bibcode_to_point_id("2019A&A...622A...2P")
        b = qdrant_tools.bibcode_to_point_id("2019A&A...622A...2P")
        assert a == b

    def test_positive_int64(self):
        # Qdrant numeric point IDs must fit in unsigned int64.
        pid = qdrant_tools.bibcode_to_point_id("2019A&A...622A...2P")
        assert 0 <= pid < 2**63

    def test_distinct_bibcodes_distinct_ids(self):
        ids = {qdrant_tools.bibcode_to_point_id(f"fake{i}") for i in range(100)}
        assert len(ids) == 100  # no collisions in small sample


class TestExpectedToolSet:
    def test_base_13_when_disabled(self, no_qdrant):
        assert mcp_server._expected_tool_set() == set(mcp_server.EXPECTED_TOOLS)
        assert len(mcp_server._expected_tool_set()) == 13

    def test_14_when_enabled(self, fake_qdrant):
        tools = mcp_server._expected_tool_set()
        assert "find_similar_by_examples" in tools
        assert len(tools) == 14


class TestHandlerDisabled:
    def test_returns_structured_error_without_backend(self, no_qdrant):
        out = mcp_server._handle_find_similar_by_examples(
            {"positive_bibcodes": ["2019A&A...622A...2P"]}
        )
        payload = json.loads(out)
        assert payload["error"] == "qdrant_not_configured"

    def test_rejects_empty_positives(self, fake_qdrant):
        out = mcp_server._handle_find_similar_by_examples(
            {"positive_bibcodes": []}
        )
        payload = json.loads(out)
        assert "error" in payload
        assert "positive_bibcodes" in payload["error"]


class TestMCPSelfTest:
    """The server self-test must pass in both enabled and disabled modes."""

    def test_self_test_passes_without_qdrant(self, no_qdrant):
        status = mcp_server.startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == 13
        assert "find_similar_by_examples" not in status["tool_names"]

    def test_self_test_passes_with_qdrant(self, fake_qdrant):
        status = mcp_server.startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == 14
        assert "find_similar_by_examples" in status["tool_names"]


class TestHandlerWithMockedClient:
    """Exercise the happy path via a mocked Qdrant client."""

    def test_dispatch_returns_structured_results(self, fake_qdrant):
        fake_hits = [
            qdrant_tools.SimilarPaper(
                bibcode="2020ApJ...900..001X",
                title="Test paper",
                year=2020,
                first_author="Doe, J.",
                score=0.91,
                arxiv_class=["astro-ph.EP"],
                community_semantic=14,
                doctype="article",
            ),
        ]
        with mock.patch.object(
            qdrant_tools, "find_similar_by_examples", return_value=fake_hits
        ):
            out = mcp_server._handle_find_similar_by_examples({
                "positive_bibcodes": ["2019A&A...622A...2P"],
                "limit": 3,
            })
        payload = json.loads(out)
        assert payload["backend"] == "qdrant"
        assert payload["collection"] == qdrant_tools.COLLECTION
        assert len(payload["results"]) == 1
        assert payload["results"][0]["bibcode"] == "2020ApJ...900..001X"
        assert payload["results"][0]["score"] == pytest.approx(0.91)
