"""Tests for the tool-surface eval scorer.

Covers the args-subset matcher, the alt_ok candidate generator, and the
score_run dispatch over (tool_correct, params_correct) for typical and edge
cases."""

from __future__ import annotations

import pytest

from scix.eval.tool_surface.scorer import (
    _args_subset_match,
    _candidate_oracles,
    _mcp_calls_only,
    _strip_mcp_prefix,
    score_run,
)


class TestArgsSubsetMatch:
    def test_empty_required_matches_anything(self):
        assert _args_subset_match({"foo": 1}, {}) is True
        assert _args_subset_match({}, {}) is True

    def test_subset_present_with_equal_values(self):
        assert _args_subset_match({"grain": "section", "limit": 10}, {"grain": "section"}) is True

    def test_extra_keys_in_actual_are_fine(self):
        assert _args_subset_match({"grain": "paper", "mode": "hybrid", "limit": 20}, {"grain": "paper"}) is True

    def test_missing_key_fails(self):
        assert _args_subset_match({"limit": 10}, {"grain": "section"}) is False

    def test_value_mismatch_fails(self):
        assert _args_subset_match({"grain": "paper"}, {"grain": "section"}) is False


class TestCandidateOracles:
    def test_primary_only(self):
        cs = _candidate_oracles({"tool": "search", "args_subset": {"grain": "paper"}})
        assert cs == [{"tool": "search", "args_subset": {"grain": "paper"}}]

    def test_alt_ok_as_strings(self):
        cs = _candidate_oracles({"tool": "section_retrieval", "args_subset": {}, "alt_ok": ["read_paper", "chunk_search"]})
        assert {c["tool"] for c in cs} == {"section_retrieval", "read_paper", "chunk_search"}
        # Bare strings produce empty args_subset
        assert all(c["args_subset"] == {} for c in cs if c["tool"] in ("read_paper", "chunk_search"))

    def test_alt_ok_as_dicts(self):
        cs = _candidate_oracles({
            "tool": "search",
            "args_subset": {"grain": "section"},
            "alt_ok": [{"tool": "paper", "args_subset": {"action": "read"}}],
        })
        assert len(cs) == 2
        assert cs[1] == {"tool": "paper", "args_subset": {"action": "read"}}


class TestStripMcpPrefix:
    def test_strips_double_underscore_prefix(self):
        assert _strip_mcp_prefix("mcp__scixstub_v1__search") == "search"

    def test_handles_underscored_tool_name(self):
        assert _strip_mcp_prefix("mcp__scixstub_v0__citation_traverse") == "citation_traverse"

    def test_passthrough_for_non_mcp(self):
        assert _strip_mcp_prefix("ToolSearch") == "ToolSearch"


class TestMcpCallsOnly:
    def test_filters_toolsearch_and_strips_prefix(self):
        calls = [
            {"name": "ToolSearch", "input": {"query": "select:..."}},
            {"name": "mcp__scixstub_v1__search", "input": {"query": "x", "grain": "paper"}},
            {"name": "Bash", "input": {"command": "ls"}},
            {"name": "mcp__scixstub_v1__paper", "input": {"bibcode": "B"}},
        ]
        out = _mcp_calls_only(calls)
        assert [c["name"] for c in out] == ["search", "paper"]
        assert out[0]["input"] == {"query": "x", "grain": "paper"}


class TestScoreRun:
    def _make_run(self, variant: str, tool_calls: list[dict]) -> dict:
        return {
            "session_id": "s",
            "variant": variant,
            "query_id": "q01",
            "run_idx": 0,
            "tool_calls": tool_calls,
        }

    def test_correct_tool_and_params(self):
        run = self._make_run("v1", [
            {"name": "mcp__scixstub_v1__search", "input": {"query": "x", "grain": "paper", "limit": 10}}
        ])
        oracle = {"v1": {"tool": "search", "args_subset": {"grain": "paper"}}}["v1"]
        s = score_run(run, {"v1": oracle})
        assert s["tool_correct"] is True
        assert s["params_correct"] is True
        assert s["first_tool"] == "search"
        assert s["mcp_call_count"] == 1

    def test_correct_tool_wrong_params(self):
        run = self._make_run("v1", [
            {"name": "mcp__scixstub_v1__search", "input": {"query": "x", "grain": "paper"}}
        ])
        oracle = {"v1": {"tool": "search", "args_subset": {"grain": "section"}}}["v1"]
        s = score_run(run, {"v1": oracle})
        assert s["tool_correct"] is True
        assert s["params_correct"] is False

    def test_wrong_tool(self):
        run = self._make_run("v1", [
            {"name": "mcp__scixstub_v1__paper", "input": {"bibcode": "B"}}
        ])
        oracle = {"v1": {"tool": "search", "args_subset": {}}}["v1"]
        s = score_run(run, {"v1": oracle})
        assert s["tool_correct"] is False
        assert s["params_correct"] is False
        assert s["first_tool"] == "paper"

    def test_no_mcp_calls_scores_zero(self):
        run = self._make_run("v1", [{"name": "ToolSearch", "input": {}}])
        oracle = {"v1": {"tool": "search", "args_subset": {}}}["v1"]
        s = score_run(run, {"v1": oracle})
        assert s["tool_correct"] is False
        assert s["params_correct"] is False
        assert s["first_tool"] is None
        assert s["mcp_call_count"] == 0

    def test_alt_ok_accepts_alternative_tool(self):
        # Oracle says section_retrieval but read_paper is alt_ok
        run = self._make_run("v0", [
            {"name": "mcp__scixstub_v0__read_paper", "input": {"bibcode": "B"}}
        ])
        oracle = {
            "v0": {
                "tool": "section_retrieval",
                "args_subset": {},
                "alt_ok": ["read_paper"],
            }
        }["v0"]
        s = score_run(run, {"v0": oracle})
        assert s["tool_correct"] is True
        # alt_ok with bare string has empty required args, so any args pass
        assert s["params_correct"] is True

    def test_alt_ok_dict_accepts_with_required_args(self):
        # alt_ok requires action=read; the call has action=read → both correct
        run = self._make_run("v1", [
            {"name": "mcp__scixstub_v1__paper", "input": {"bibcode": "B", "action": "read"}}
        ])
        oracle = {
            "v1": {
                "tool": "search",
                "args_subset": {"grain": "section"},
                "alt_ok": [{"tool": "paper", "args_subset": {"action": "read"}}],
            }
        }["v1"]
        s = score_run(run, {"v1": oracle})
        assert s["tool_correct"] is True
        assert s["params_correct"] is True

    def test_first_call_is_what_counts(self):
        # If the agent makes multiple calls, only the FIRST is scored
        run = self._make_run("v1", [
            {"name": "mcp__scixstub_v1__paper", "input": {"bibcode": "B"}},
            {"name": "mcp__scixstub_v1__search", "input": {"grain": "paper", "query": "x"}},
        ])
        oracle = {"v1": {"tool": "search", "args_subset": {"grain": "paper"}}}["v1"]
        s = score_run(run, {"v1": oracle})
        assert s["tool_correct"] is False  # first was paper, not search
        assert s["mcp_call_count"] == 2  # but we count both

    def test_no_oracle_returns_no_oracle_flag(self):
        run = self._make_run("v9", [{"name": "mcp__scixstub_v9__search", "input": {}}])
        s = score_run(run, {})
        assert s.get("no_oracle") is True
        assert s["mcp_call_count"] == 1
