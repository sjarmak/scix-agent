"""Unit tests for the thin mechanical CLI (scix.cli).

DB-free: tool specs, the handler registry, and the connection are all injected.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from scix import cli

FAKE_SPECS = [
    dict(
        name="get_paper",
        description="Get a paper by bibcode.\n(second line ignored)",
        inputSchema={
            "type": "object",
            "properties": {"bibcode": {"type": "string", "description": "ADS bibcode"}},
            "required": ["bibcode"],
        },
    ),
    dict(
        name="search",
        description="Search.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
                "filters": {"type": "object"},
                "hybrid": {"type": "boolean"},
                "mode": {"type": "string", "enum": ["hybrid", "dense", "bm25"]},
            },
            "required": ["query"],
        },
    ),
    # Reasoning tool — MUST be excluded from the CLI.
    dict(
        name="lit_review",
        description="Reasoning.",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
]


def _registry():
    calls: list[tuple] = []

    def handler(conn, args):
        calls.append((conn, args))
        return json.dumps(args)

    return {"get_paper": handler, "search": handler}, calls


@contextmanager
def _conn_factory():
    yield "FAKECONN"


def _run(argv, registry):
    return cli.run(argv, specs=FAKE_SPECS, registry=registry, conn_factory=_conn_factory)


def test_allowlist_excludes_reasoning_and_session_tools():
    assert "lit_review" not in cli.MECHANICAL_TOOLS
    assert "synthesize_findings" not in cli.MECHANICAL_TOOLS
    assert "get_working_set" not in cli.MECHANICAL_TOOLS
    assert {"search", "get_paper", "read_paper", "facet_counts"} <= cli.MECHANICAL_TOOLS


def test_parser_only_builds_allowlisted_subcommands():
    parser = cli.build_parser(FAKE_SPECS)
    # lit_review is in FAKE_SPECS but not in MECHANICAL_TOOLS -> not a subcommand.
    with pytest.raises(SystemExit):
        parser.parse_args(["lit_review", "--query", "x"])
    ns = parser.parse_args(["get_paper", "2010PhRvC..81c4911A"])
    assert ns._tool == "get_paper"


def test_positional_shorthand_maps_to_primary_key(capsys):
    registry, calls = _registry()
    rc = _run(["get_paper", "2010PhRvC..81c4911A"], registry)
    assert rc == 0
    assert calls[0][1] == {"bibcode": "2010PhRvC..81c4911A"}
    assert json.loads(capsys.readouterr().out) == {"bibcode": "2010PhRvC..81c4911A"}


def test_flags_typed_and_json_parsed(capsys):
    registry, calls = _registry()
    _run(["search", "--query", "halos", "--k", "5", "--filters", '{"year_min":2020}'], registry)
    assert calls[0][1] == {"query": "halos", "k": 5, "filters": {"year_min": 2020}}


def test_omitted_optionals_not_in_args():
    registry, calls = _registry()
    _run(["search", "--query", "halos"], registry)
    assert calls[0][1] == {"query": "halos"}  # k/filters/hybrid/mode absent -> handler defaults


def test_boolean_optional_action():
    registry, calls = _registry()
    _run(["search", "--query", "q", "--hybrid"], registry)
    assert calls[0][1]["hybrid"] is True
    registry, calls = _registry()
    _run(["search", "--query", "q", "--no-hybrid"], registry)
    assert calls[0][1]["hybrid"] is False


def test_enum_choice_enforced():
    registry, _ = _registry()
    with pytest.raises(SystemExit):
        _run(["search", "--query", "q", "--mode", "bogus"], registry)


def test_invalid_json_arg_errors():
    registry, _ = _registry()
    with pytest.raises(SystemExit, match="valid JSON"):
        _run(["search", "--query", "q", "--filters", "{not json"], registry)


def test_missing_required_errors():
    registry, _ = _registry()
    with pytest.raises(SystemExit):
        _run(["search"], registry)  # --query / positional required


def test_handler_receives_conn(capsys):
    registry, calls = _registry()
    _run(["get_paper", "X"], registry)
    assert calls[0][0] == "FAKECONN"


def test_pretty_indents(capsys):
    registry, _ = _registry()
    _run(["--pretty", "search", "--query", "q"], registry)
    out = capsys.readouterr().out
    assert "\n" in out.strip()  # indented multi-line
    assert json.loads(out) == {"query": "q"}


def test_explicit_flag_overrides_positional_absence(capsys):
    # primary given via --flag instead of positional
    registry, calls = _registry()
    _run(["get_paper", "--bibcode", "2010X"], registry)
    assert calls[0][1] == {"bibcode": "2010X"}
