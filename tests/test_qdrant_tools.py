"""Unit tests for scix.qdrant_tools feature-flagging and MCP integration.

The find_similar_by_examples MCP tool was retired on 2026-04-25 (Qdrant
backend not in active use). This test suite was rewritten on the same
date to validate the retirement contract:

  1. is_enabled() correctly reads the QDRANT_URL env var (the helper is
     still used by other code paths).
  2. bibcode_to_point_id is deterministic and within the int64 range Qdrant
     accepts (kept in case the tool returns; fits the eventual NAS-Qdrant
     migration).
  3. The MCP server does NOT register find_similar_by_examples regardless
     of QDRANT_URL, and the self-test reports exactly 15 tools either way.
  4. The dispatch path for the retired name returns a clear
     "tool_removed" error.
"""
from __future__ import annotations

import json
import os

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
    """Post-2026-04-25: the tool set is exactly 15 regardless of Qdrant state."""

    def test_15_tools_when_qdrant_disabled(self, no_qdrant):
        assert mcp_server._expected_tool_set() == set(mcp_server.EXPECTED_TOOLS)
        assert len(mcp_server._expected_tool_set()) == 15
        assert "find_similar_by_examples" not in mcp_server._expected_tool_set()

    def test_15_tools_when_qdrant_enabled(self, fake_qdrant):
        # The retired tool is gone from the active surface even when the
        # Qdrant feature flag is on; the gating is now hardcoded to no-op.
        tools = mcp_server._expected_tool_set()
        assert "find_similar_by_examples" not in tools
        assert len(tools) == 15


class TestRetiredToolDispatch:
    """The retired tool name must return a structured tool_removed error."""

    def test_dispatch_returns_tool_removed(self, no_qdrant):
        from scix.mcp_server import _dispatch_consolidated

        # Use _dispatch_consolidated directly — _dispatch_tool runs the
        # alias layer, but find_similar_by_examples is NOT in
        # _DEPRECATED_ALIASES (it was hard-removed, not renamed).
        out = _dispatch_consolidated(
            None,  # conn unused for the retired-tool branch
            "find_similar_by_examples",
            {"positive_bibcodes": ["2019A&A...622A...2P"]},
        )
        payload = json.loads(out)
        assert payload["error"] == "tool_removed"
        assert payload["removed_in"] == "2026-04-25"
        assert "find_similar_by_examples" in payload["message"]


class TestMCPSelfTest:
    """The server self-test must pass at exactly 15 tools regardless of QDRANT_URL."""

    def test_self_test_passes_without_qdrant(self, no_qdrant):
        status = mcp_server.startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == 15
        assert "find_similar_by_examples" not in status["tool_names"]

    def test_self_test_passes_with_qdrant(self, fake_qdrant):
        status = mcp_server.startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == 15
        assert "find_similar_by_examples" not in status["tool_names"]
