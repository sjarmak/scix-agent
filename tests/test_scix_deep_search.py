"""Tests for ``scripts/scix_deep_search.py`` (MH-7 acceptance).

These tests mock the dispatcher — no real ``claude -p`` invocation, no
OAuth round-trip. The dispatcher seam is :class:`Dispatcher` (Protocol)
in the script; tests inject a deterministic async iterator via
:class:`FakeDispatcher`.

Acceptance criteria covered (from the work-unit task spec):

- AC #5 — flag parsing (``--max-turns``, ``--rigor``, ``--skeptic``).
- AC #3 — transcript directory contents (4 files).
- AC #6 — ``time_to_first_useful_output`` field present and correct.
- AC #7 — fixture dispatcher returning 30 turns is truncated to 25.
- AC #8 — persona file frontmatter is valid YAML; required body sections
  are present (including the verbatim "When you assert a claim…" line).
- The wrapper does NOT import any paid SDK.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module loader — script lives outside the ``src/scix`` package so we
# import it by path rather than via package import. This isolates the
# test from the repo's package layout and matches how the script is
# invoked in production (``python scripts/scix_deep_search.py``).
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "scix_deep_search.py"
)
_SPEC = importlib.util.spec_from_file_location("scix_deep_search", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
sds = importlib.util.module_from_spec(_SPEC)
sys.modules["scix_deep_search"] = sds
_SPEC.loader.exec_module(sds)


# ---------------------------------------------------------------------------
# Fixtures — fake dispatcher
# ---------------------------------------------------------------------------


class FakeDispatcher:
    """Async-iterator dispatcher that yields a canned event sequence.

    Each ``__call__`` invocation replays the configured event list with
    a small per-event ``asyncio.sleep`` so timestamps differ enough for
    the time-to-first-output assertion to be meaningful.
    """

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        per_event_delay_s: float = 0.0,
    ) -> None:
        self._events = events
        self._delay = per_event_delay_s
        self.last_prompt: str | None = None
        self.last_max_turns: int | None = None

    async def __call__(
        self, prompt: str, max_turns: int
    ) -> AsyncIterator[dict[str, Any]]:
        self.last_prompt = prompt
        self.last_max_turns = max_turns
        import asyncio

        for ev in self._events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield ev


@pytest.fixture
def text_events() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "Investigating H0 tension lineage. "},
        {
            "type": "tool_use",
            "tool_name": "concept_search",
            "text": "(searched for 'local H0 SH0ES')",
        },
        {
            "type": "tool_result",
            "text": "Top hit: 2011ApJ...730..119R — Riess+ 2011.",
        },
        {
            "type": "text",
            "text": (
                "The earliest assertion is `bibcode:2011ApJ...730..119R §1` "
                "\"Our value of H0 is 2.4σ higher\"."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    ns = sds.parse_args(["What is H0?"])
    assert ns.question == "What is H0?"
    assert ns.max_turns == 25
    assert ns.rigor is False
    assert ns.skeptic is False
    assert ns.runs_dir == sds.DEFAULT_RUNS_DIR


def test_parse_args_max_turns() -> None:
    ns = sds.parse_args(["q", "--max-turns", "30"])
    assert ns.max_turns == 30


def test_parse_args_rigor() -> None:
    ns = sds.parse_args(["q", "--rigor"])
    assert ns.rigor is True


def test_parse_args_skeptic_flag_present() -> None:
    ns = sds.parse_args(["q", "--skeptic"])
    assert ns.skeptic is True


def test_main_skeptic_raises_notimplemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError) as exc:
        sds.main(["q", "--skeptic", "--runs-dir", str(tmp_path)])
    assert "SH-2" in str(exc.value)
    assert "MH-0" in str(exc.value)


def test_run_deep_search_skeptic_raises_notimplemented(
    tmp_path: Path, text_events: list[dict[str, Any]]
) -> None:
    dispatcher = FakeDispatcher(text_events)
    with pytest.raises(NotImplementedError):
        sds.run_deep_search(
            "q", dispatcher, runs_dir=tmp_path, skeptic=True
        )


# ---------------------------------------------------------------------------
# Run-id format
# ---------------------------------------------------------------------------


def test_make_run_id_format() -> None:
    rid = sds.make_run_id()
    # Shape: YYYY-MM-DD_HHMMSS_<6-hex>
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}$", rid), rid


def test_make_run_id_unique() -> None:
    ids = {sds.make_run_id() for _ in range(50)}
    # Random suffix collision over 50 draws of 24-bit hex is ~10^-8.
    assert len(ids) >= 49


# ---------------------------------------------------------------------------
# Transcript directory contents
# ---------------------------------------------------------------------------


def test_run_deep_search_writes_all_files(
    tmp_path: Path, text_events: list[dict[str, Any]]
) -> None:
    dispatcher = FakeDispatcher(text_events)
    result = sds.run_deep_search(
        "What is H0?", dispatcher, runs_dir=tmp_path
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "question.txt").is_file()
    assert (result.run_dir / "answer.md").is_file()
    assert (result.run_dir / "tool_calls.jsonl").is_file()
    assert (result.run_dir / "metadata.json").is_file()

    assert (result.run_dir / "question.txt").read_text().strip() == "What is H0?"
    answer_text = (result.run_dir / "answer.md").read_text()
    assert "earliest assertion" in answer_text  # text events concatenated
    assert "(searched for" not in answer_text  # tool_use events excluded


def test_tool_calls_jsonl_is_one_event_per_line(
    tmp_path: Path, text_events: list[dict[str, Any]]
) -> None:
    dispatcher = FakeDispatcher(text_events)
    result = sds.run_deep_search("q", dispatcher, runs_dir=tmp_path)

    lines = (result.run_dir / "tool_calls.jsonl").read_text().splitlines()
    assert len(lines) == len(text_events)
    parsed = [json.loads(line) for line in lines]
    assert [ev["type"] for ev in parsed] == [ev["type"] for ev in text_events]


# ---------------------------------------------------------------------------
# metadata.json schema
# ---------------------------------------------------------------------------


_REQUIRED_METADATA_KEYS = {
    "run_id",
    "question",
    "start_time",
    "end_time",
    "duration_s",
    "n_turns",
    "max_turns",
    "truncated",
    "model",
    "persona_path",
    "rigor",
    "skeptic",
    "time_to_first_useful_output",
}


def test_metadata_json_schema_required_keys(
    tmp_path: Path, text_events: list[dict[str, Any]]
) -> None:
    dispatcher = FakeDispatcher(text_events)
    result = sds.run_deep_search(
        "q", dispatcher, runs_dir=tmp_path, max_turns=25
    )
    md = json.loads((result.run_dir / "metadata.json").read_text())
    assert set(md.keys()) == _REQUIRED_METADATA_KEYS, (
        f"missing keys: {_REQUIRED_METADATA_KEYS - set(md.keys())}, "
        f"extra keys: {set(md.keys()) - _REQUIRED_METADATA_KEYS}"
    )

    # Type checks
    assert isinstance(md["run_id"], str)
    assert isinstance(md["question"], str)
    assert isinstance(md["start_time"], str)
    assert isinstance(md["end_time"], str)
    assert isinstance(md["duration_s"], (int, float))
    assert isinstance(md["n_turns"], int)
    assert isinstance(md["max_turns"], int)
    assert isinstance(md["truncated"], bool)
    assert isinstance(md["model"], str)
    assert isinstance(md["persona_path"], str)
    assert isinstance(md["rigor"], bool)
    assert isinstance(md["skeptic"], bool)
    assert md["time_to_first_useful_output"] is None or isinstance(
        md["time_to_first_useful_output"], (int, float)
    )

    # Value checks
    assert md["max_turns"] == 25
    assert md["truncated"] is False
    assert md["n_turns"] == len(text_events)
    assert md["model"] == "sonnet"
    assert md["persona_path"].endswith("deep_search_investigator.md")


def test_metadata_records_rigor_flag(
    tmp_path: Path, text_events: list[dict[str, Any]]
) -> None:
    dispatcher = FakeDispatcher(text_events)
    result = sds.run_deep_search(
        "q", dispatcher, runs_dir=tmp_path, rigor=True
    )
    md = json.loads((result.run_dir / "metadata.json").read_text())
    assert md["rigor"] is True


# ---------------------------------------------------------------------------
# --max-turns enforcement
# ---------------------------------------------------------------------------


def test_max_turns_truncates_30_to_25(tmp_path: Path) -> None:
    events = [{"type": "text", "text": f"turn {i}. "} for i in range(30)]
    dispatcher = FakeDispatcher(events)
    result = sds.run_deep_search(
        "q", dispatcher, runs_dir=tmp_path, max_turns=25
    )

    # Only 25 events recorded
    lines = (result.run_dir / "tool_calls.jsonl").read_text().splitlines()
    assert len(lines) == 25

    md = json.loads((result.run_dir / "metadata.json").read_text())
    assert md["truncated"] is True
    assert md["n_turns"] == 25
    assert md["max_turns"] == 25


def test_max_turns_no_truncation_when_under_budget(
    tmp_path: Path, text_events: list[dict[str, Any]]
) -> None:
    dispatcher = FakeDispatcher(text_events)
    result = sds.run_deep_search(
        "q", dispatcher, runs_dir=tmp_path, max_turns=25
    )
    md = json.loads((result.run_dir / "metadata.json").read_text())
    assert md["truncated"] is False
    assert md["n_turns"] == len(text_events)


# ---------------------------------------------------------------------------
# time_to_first_useful_output
# ---------------------------------------------------------------------------


def test_time_to_first_useful_output_set_on_bibcode(tmp_path: Path) -> None:
    events = [
        {"type": "text", "text": "thinking..."},
        {"type": "text", "text": "still thinking..."},
        {
            "type": "tool_result",
            "text": "Found 2011ApJ...730..119R",
        },
        {"type": "text", "text": "done"},
    ]
    dispatcher = FakeDispatcher(events, per_event_delay_s=0.01)
    result = sds.run_deep_search("q", dispatcher, runs_dir=tmp_path)

    md = json.loads((result.run_dir / "metadata.json").read_text())
    assert md["time_to_first_useful_output"] is not None
    assert md["time_to_first_useful_output"] > 0
    # The bibcode-bearing event was the third — should have observed at
    # least 2 prior delays of 0.01s each (~0.02s minimum).
    assert md["time_to_first_useful_output"] >= 0.02


def test_time_to_first_useful_output_none_when_no_bibcode(tmp_path: Path) -> None:
    events = [
        {"type": "text", "text": "no bibcode here"},
        {"type": "text", "text": "still none"},
    ]
    dispatcher = FakeDispatcher(events)
    result = sds.run_deep_search("q", dispatcher, runs_dir=tmp_path)
    md = json.loads((result.run_dir / "metadata.json").read_text())
    assert md["time_to_first_useful_output"] is None


def test_bibcode_regex_matches_canonical_forms() -> None:
    # Real bibcodes from astrophysics
    cases = [
        "2011ApJ...730..119R",
        "2014PhRvL.112x1101B",
        "2022ApJ...934L...7R",
        "1998AJ....116.1009R",
    ]
    for b in cases:
        assert sds.BIBCODE_RE.search(b), f"failed to match bibcode: {b}"


# ---------------------------------------------------------------------------
# Persona file
# ---------------------------------------------------------------------------


_PERSONA_PATH = (
    Path(__file__).resolve().parent.parent
    / ".claude"
    / "agents"
    / "deep_search_investigator.md"
)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_yaml, body) — the file must start with ``---``."""
    assert text.startswith("---\n"), "persona file must start with YAML frontmatter"
    rest = text[4:]
    end = rest.find("\n---\n")
    assert end != -1, "persona frontmatter missing closing ``---``"
    return rest[:end], rest[end + 5 :]


def test_persona_file_exists() -> None:
    assert _PERSONA_PATH.is_file(), f"persona file not found: {_PERSONA_PATH}"


def test_persona_frontmatter_is_valid_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    text = _PERSONA_PATH.read_text()
    fm, _ = _split_frontmatter(text)
    parsed = yaml.safe_load(fm)
    assert isinstance(parsed, dict)
    assert parsed.get("name") == "deep_search_investigator"
    assert isinstance(parsed.get("description"), str) and parsed["description"]
    assert isinstance(parsed.get("tools"), list) and len(parsed["tools"]) >= 13


def test_persona_lists_15_tools() -> None:
    yaml = pytest.importorskip("yaml")
    fm, _ = _split_frontmatter(_PERSONA_PATH.read_text())
    parsed = yaml.safe_load(fm)
    tools = parsed["tools"]
    # 13 existing + claim_blame + find_replications = 15
    assert len(tools) == 15
    names = [t.split("__")[-1] for t in tools]
    assert "claim_blame" in names
    assert "find_replications" in names
    # Sample of existing tools must be present
    for expected in ("search", "concept_search", "citation_chain", "read_paper"):
        assert expected in names


def test_persona_body_contains_required_sections() -> None:
    text = _PERSONA_PATH.read_text()
    _, body = _split_frontmatter(text)
    # The required sections (per PRD MH-7 + Agent harness shape).
    assert "## Linking" in body, "missing Linking section"
    assert "# Investigation discipline" in body, "missing Investigation discipline section"
    assert "# Refusal of exhaustiveness" in body, "missing Refusal of exhaustiveness section"


def test_persona_contains_mandatory_citation_sentence() -> None:
    """AC #2 — the verbatim citation-discipline rule must appear."""
    text = _PERSONA_PATH.read_text()
    required = (
        "When you assert a claim, the next sentence must cite a "
        "bibcode + section + quoted span from a tool result."
    )
    assert required in text, (
        "persona must contain the verbatim citation-discipline sentence "
        "(AC #2)"
    )


def test_persona_documents_bibcode_format() -> None:
    """AC #1 — linking format is `bibcode:YYYYJ...VVV..PPPX §section`."""
    text = _PERSONA_PATH.read_text()
    assert "bibcode:" in text
    assert "§section" in text or "§" in text


def test_persona_refuses_exhaustiveness_with_search_url() -> None:
    text = _PERSONA_PATH.read_text()
    assert "search?q=" in text, "Refusal section must point at a search URL"


# ---------------------------------------------------------------------------
# No paid-API SDK in the wrapper
# ---------------------------------------------------------------------------


def test_wrapper_imports_no_paid_sdk() -> None:
    src = _SCRIPT_PATH.read_text()
    for forbidden in ("anthropic", "openai", "cohere"):
        # Allow the word in comments/docstrings — only flag actual import lines.
        forbidden_imports = [
            line
            for line in src.splitlines()
            if line.strip().startswith(("import ", "from "))
            and forbidden in line
        ]
        assert not forbidden_imports, (
            f"forbidden paid-SDK import found: {forbidden_imports!r}"
        )


def test_dispatcher_seam_is_dependency_injected() -> None:
    """``run_deep_search`` must accept any Dispatcher; tests rely on this."""
    import inspect

    sig = inspect.signature(sds.run_deep_search)
    assert "dispatcher" in sig.parameters
    # Dispatcher Protocol is the published seam
    assert hasattr(sds, "Dispatcher")
    assert hasattr(sds, "RealDispatcher")


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_names_persona() -> None:
    p = sds.build_prompt("hello", rigor=False)
    assert "deep_search_investigator" in p
    assert "hello" in p


def test_build_prompt_encodes_rigor() -> None:
    p_off = sds.build_prompt("q", rigor=False)
    p_on = sds.build_prompt("q", rigor=True)
    assert "[rigor=on]" in p_on
    assert "[rigor=on]" not in p_off
