"""Direct unit tests for the deprecated-alias dispatch/transform logic.

Bead: ``scix_experiments-hqjv`` (DEEP_AUDIT 2026-06-18 coverage gap).

``_dispatch_tool`` / ``_dispatch_consolidated`` and their supporting
``_transform_deprecated_args`` rewriter were previously exercised only
*indirectly* — via ~40 dispatch/integration tests that each happen to route
through the alias layer. This module pins the alias-transform contract
*directly*, with no database, so a regression in the rewrite table surfaces
here instead of as a confusing failure three layers down an integration test.

The contract under test (see ``mcp_server`` near ``_ALIAS_TRANSFORMS``):

* Each deprecated alias rewrites to the correct consolidated **target** plus
  any forced ``set_args`` (mode / action / method / direction / annotate flag),
  and the three key-renaming aliases (keyword_search, entity_search,
  search_within_paper) move their old arg into the new key.
* ``_dispatch_tool`` injects ``use_instead`` guidance from
  ``_DEPRECATED_ALIASES`` — which for seven aliases deliberately differs from
  the dispatch target (e.g. ``get_citation_context`` dispatches to its own
  handler but advertises ``citation_traverse`` as the modern equivalent).
* An unknown tool name returns the structured ``unknown_tool`` error envelope.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix import mcp_server
from scix.mcp_errors import ErrorCode
from scix.mcp_server import (
    _ALIAS_TRANSFORMS,
    _DEPRECATED_ALIASES,
    _dispatch_consolidated,
    _dispatch_tool,
    _transform_deprecated_args,
)

# ---------------------------------------------------------------------------
# Expected (alias -> target + transformed args) table.
#
# Written out by hand rather than derived from _ALIAS_TRANSFORMS so the test is
# an independent pin, not a tautology that merely re-reads the source dict.
# Each row: (alias, input_args, expected_target, expected_args).
# ---------------------------------------------------------------------------

_TRANSFORM_CASES: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = [
    # search merges (mode flag forced).
    (
        "semantic_search",
        {"query": "dark matter"},
        "search",
        {"query": "dark matter", "mode": "semantic"},
    ),
    # keyword_search additionally renames terms -> query.
    (
        "keyword_search",
        {"terms": "black holes", "limit": 5},
        "search",
        {"query": "black holes", "limit": 5, "mode": "keyword"},
    ),
    # citation_traverse family (mode / direction forced).
    ("citation_graph", {"bibcode": "B"}, "citation_traverse", {"bibcode": "B", "mode": "graph"}),
    ("citation_chain", {"bibcode": "B"}, "citation_traverse", {"bibcode": "B", "mode": "chain"}),
    (
        "get_citations",
        {"bibcode": "B"},
        "citation_traverse",
        {"bibcode": "B", "mode": "graph", "direction": "forward"},
    ),
    (
        "get_references",
        {"bibcode": "B"},
        "citation_traverse",
        {"bibcode": "B", "mode": "graph", "direction": "backward"},
    ),
    # citation_similarity family (method forced).
    (
        "co_citation_analysis",
        {"bibcode": "B"},
        "citation_similarity",
        {"bibcode": "B", "method": "co_citation"},
    ),
    (
        "bibliographic_coupling",
        {"bibcode": "B"},
        "citation_similarity",
        {"bibcode": "B", "method": "coupling"},
    ),
    # entity family (action forced); entity_search renames entity_name -> query.
    ("entity_search", {"entity_name": "JWST"}, "entity", {"query": "JWST", "action": "search"}),
    ("resolve_entity", {"query": "JWST"}, "entity", {"query": "JWST", "action": "resolve"}),
    ("entity_context", {"query": "JWST"}, "entity", {"query": "JWST", "action": "profile"}),
    # forward_citations annotate modes.
    (
        "cited_by_intent",
        {"bibcode": "B"},
        "forward_citations",
        {"bibcode": "B", "annotate": "intent"},
    ),
    (
        "find_replications",
        {"bibcode": "B"},
        "forward_citations",
        {"bibcode": "B", "annotate": "relation"},
    ),
    # get_paper with entities forced on.
    ("document_context", {"bibcode": "B"}, "get_paper", {"bibcode": "B", "include_entities": True}),
    (
        "get_openalex_topics",
        {"bibcode": "B"},
        "get_paper",
        {"bibcode": "B", "include_entities": True},
    ),
    # graph_context community flag.
    (
        "get_paper_metrics",
        {"bibcode": "B"},
        "graph_context",
        {"bibcode": "B", "include_community": False},
    ),
    (
        "explore_community",
        {"bibcode": "B"},
        "graph_context",
        {"bibcode": "B", "include_community": True},
    ),
    # read_paper family; search_within_paper renames query -> search_query.
    ("read_paper_section", {"bibcode": "B"}, "read_paper", {"bibcode": "B"}),
    (
        "search_within_paper",
        {"query": "method", "bibcode": "B"},
        "read_paper",
        {"search_query": "method", "bibcode": "B"},
    ),
    # Pure legacy passthroughs — target == alias, args untouched. These dispatch
    # to a dedicated handler even though use_instead advertises a modern tool.
    ("get_author_papers", {"author": "Smith"}, "get_author_papers", {"author": "Smith"}),
    ("get_citation_context", {"bibcode": "B"}, "get_citation_context", {"bibcode": "B"}),
    ("add_to_working_set", {"bibcode": "B"}, "add_to_working_set", {"bibcode": "B"}),
    ("get_working_set", {}, "get_working_set", {}),
    ("get_session_summary", {}, "get_session_summary", {}),
    ("clear_working_set", {}, "clear_working_set", {}),
    ("entity_profile", {"bibcode": "B"}, "entity_profile", {"bibcode": "B"}),
]


# ---------------------------------------------------------------------------
# Structural invariants — the two tables must stay in lockstep.
# ---------------------------------------------------------------------------


def test_alias_tables_have_identical_keys() -> None:
    """Every deprecated alias must have exactly one transform entry.

    A name in ``_DEPRECATED_ALIASES`` but not ``_ALIAS_TRANSFORMS`` would be
    dispatched straight to its ``use_instead`` target with un-rewritten args
    (``_transform_deprecated_args`` returns the input unchanged when ``spec is
    None``); a name in ``_ALIAS_TRANSFORMS`` but not ``_DEPRECATED_ALIASES``
    would never reach the alias path at all.
    """
    assert set(_DEPRECATED_ALIASES) == set(_ALIAS_TRANSFORMS)


def test_every_alias_is_covered_by_this_module() -> None:
    """Guard against silent drift: a new alias must add a transform case here."""
    assert {alias for alias, *_ in _TRANSFORM_CASES} == set(_DEPRECATED_ALIASES)


def test_passthrough_aliases_advertise_a_different_modern_tool() -> None:
    """The seven aliases that dispatch to their own dedicated handler
    (``target == alias``) must still advertise a *different* modern tool in
    ``use_instead`` — that divergence is the whole reason both tables exist.
    A refactor collapsing ``use_instead`` onto the dispatch target would be a
    real behaviour change, so pin it explicitly rather than only via the
    per-alias dispatch assertions below."""
    passthrough = {a for a, spec in _ALIAS_TRANSFORMS.items() if spec.target == a}
    assert passthrough, "expected some self-targeting passthrough aliases"
    assert all(_DEPRECATED_ALIASES[a] != a for a in passthrough)


# ---------------------------------------------------------------------------
# _transform_deprecated_args — pure arg rewriting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "input_args", "expected_target", "expected_args"),
    _TRANSFORM_CASES,
    ids=[c[0] for c in _TRANSFORM_CASES],
)
def test_transform_rewrites_to_target_and_args(
    alias: str, input_args: dict[str, Any], expected_target: str, expected_args: dict[str, Any]
) -> None:
    use_instead = _DEPRECATED_ALIASES[alias]
    target, new_args = _transform_deprecated_args(alias, use_instead, dict(input_args))
    assert target == expected_target
    assert new_args == expected_args


def test_transform_does_not_mutate_input_args() -> None:
    """The rewriter must copy: callers reuse the original args dict for logging."""
    original = {"terms": "black holes", "limit": 5}
    snapshot = dict(original)
    _transform_deprecated_args("keyword_search", "search", original)
    assert original == snapshot


def test_transform_unknown_alias_passes_through_unchanged() -> None:
    """No transform entry -> (new_name, copy-of-args), no rewriting applied."""
    target, new_args = _transform_deprecated_args("not_an_alias", "search", {"query": "x"})
    assert target == "search"
    assert new_args == {"query": "x"}


# ---------------------------------------------------------------------------
# arg_fn edge cases — the three key-renaming helpers have fallback behaviour
# that the representative cases above don't fully cover.
# ---------------------------------------------------------------------------


def test_kw_terms_to_query_keeps_existing_query_when_terms_absent() -> None:
    _, args = _transform_deprecated_args("keyword_search", "search", {"query": "existing"})
    assert args["query"] == "existing"


def test_kw_terms_to_query_defaults_to_empty_when_both_absent() -> None:
    _, args = _transform_deprecated_args("keyword_search", "search", {})
    assert args["query"] == ""


def test_kw_terms_wins_over_query_when_both_present() -> None:
    _, args = _transform_deprecated_args("keyword_search", "search", {"terms": "t", "query": "q"})
    assert args["query"] == "t"
    assert "terms" not in args


def test_entity_name_does_not_clobber_explicit_query() -> None:
    """entity_search keeps an explicit ``query``; ``entity_name`` is left as-is."""
    _, args = _transform_deprecated_args(
        "entity_search", "entity", {"entity_name": "e", "query": "q"}
    )
    assert args["query"] == "q"


def test_query_to_search_query_noop_when_query_absent() -> None:
    _, args = _transform_deprecated_args("search_within_paper", "read_paper", {"bibcode": "B"})
    assert "search_query" not in args


# ---------------------------------------------------------------------------
# _dispatch_tool — deprecation wrapping + use_instead injection.
#
# _dispatch_consolidated is stubbed so no database is touched and we can capture
# exactly what name/args the alias layer dispatched.
# ---------------------------------------------------------------------------


@pytest.fixture
def capture_dispatch(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace _dispatch_consolidated with a recorder returning fixed JSON."""
    seen: list[tuple[str, dict[str, Any]]] = []

    def _fake(conn: Any, name: str, args: dict[str, Any]) -> str:
        seen.append((name, dict(args)))
        return json.dumps({"ok": True})

    monkeypatch.setattr(mcp_server, "_dispatch_consolidated", _fake)
    return seen


@pytest.mark.parametrize(
    ("alias", "input_args", "expected_target", "expected_args"),
    _TRANSFORM_CASES,
    ids=[c[0] for c in _TRANSFORM_CASES],
)
def test_dispatch_tool_routes_alias_to_target_with_transformed_args(
    capture_dispatch: list[tuple[str, dict[str, Any]]],
    alias: str,
    input_args: dict[str, Any],
    expected_target: str,
    expected_args: dict[str, Any],
) -> None:
    """End-to-end through _dispatch_tool: the consolidated handler is invoked
    with the rewritten target name and args."""
    out = _dispatch_tool(MagicMock(), alias, dict(input_args))
    assert capture_dispatch == [(expected_target, expected_args)]

    # And the response carries the deprecation metadata the agent reads.
    data = json.loads(out)
    assert data["deprecated"] is True
    assert data["original_tool"] == alias
    assert data["use_instead"] == _DEPRECATED_ALIASES[alias]


def test_dispatch_tool_does_not_mutate_caller_args(
    capture_dispatch: list[tuple[str, dict[str, Any]]],
) -> None:
    """The caller's args dict is logged before rewriting (``args=%s``), so the
    alias layer must not rewrite it in place."""
    original = {"terms": "black holes", "limit": 5}
    snapshot = dict(original)
    _dispatch_tool(MagicMock(), "keyword_search", original)
    assert original == snapshot


def test_dispatch_tool_set_args_override_conflicting_caller_value(
    capture_dispatch: list[tuple[str, dict[str, Any]]],
) -> None:
    """A forced ``set_args`` flag wins over a conflicting caller-supplied value:
    a stale client cannot escape the consolidated tool's forced mode. Here
    ``get_paper_metrics`` forces ``include_community=False`` even though the
    caller passed ``True``."""
    _dispatch_tool(MagicMock(), "get_paper_metrics", {"bibcode": "B", "include_community": True})
    target, args = capture_dispatch[0]
    assert target == "graph_context"
    assert args["include_community"] is False


def test_dispatch_tool_passes_non_deprecated_tool_through_unwrapped(
    capture_dispatch: list[tuple[str, dict[str, Any]]],
) -> None:
    """A current (non-alias) tool name dispatches verbatim, with no
    deprecation metadata grafted onto the result."""
    out = _dispatch_tool(MagicMock(), "search", {"query": "x", "mode": "hybrid"})
    assert capture_dispatch == [("search", {"query": "x", "mode": "hybrid"})]
    data = json.loads(out)
    assert "deprecated" not in data
    assert "use_instead" not in data
    assert "original_tool" not in data


def test_dispatch_tool_leaves_non_object_result_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_wrap_deprecated`` only grafts metadata onto a JSON *object*. When a
    handler returns a top-level array the result passes through unchanged — pin
    that documented behaviour so the deprecation graft can't start raising on
    list-returning handlers."""
    monkeypatch.setattr(mcp_server, "_dispatch_consolidated", lambda *a: "[]")
    out = _dispatch_tool(MagicMock(), "keyword_search", {"terms": "x"})
    assert json.loads(out) == []


# ---------------------------------------------------------------------------
# _dispatch_consolidated — unknown-tool error envelope.
# ---------------------------------------------------------------------------


def test_dispatch_consolidated_unknown_tool_returns_error_envelope() -> None:
    """An unrecognised name returns the structured unknown_tool envelope
    without touching the connection."""
    conn = MagicMock()
    out = _dispatch_consolidated(conn, "totally_not_a_real_tool", {})
    data = json.loads(out)
    assert data["error_code"] == ErrorCode.UNKNOWN_TOOL
    assert "totally_not_a_real_tool" in data["error"]
    conn.cursor.assert_not_called()


def test_dispatch_tool_unknown_non_alias_tool_surfaces_unknown_tool() -> None:
    """An unknown name that is also not a deprecated alias flows straight to
    the unknown_tool envelope — not wrapped as deprecated."""
    out = _dispatch_tool(MagicMock(), "totally_not_a_real_tool", {})
    data = json.loads(out)
    assert data["error_code"] == ErrorCode.UNKNOWN_TOOL
    assert "deprecated" not in data
