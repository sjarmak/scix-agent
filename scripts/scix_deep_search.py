#!/usr/bin/env python3
"""SciX Deep Search v1 CLI wrapper (MH-7).

Thin wrapper that takes a question, dispatches the
``deep_search_investigator`` persona via Claude Code OAuth subagents
(``claude -p``), captures the structured event stream, and writes a
transcript directory to ``~/.scix/deep_search_runs/<run_id>/``.

This wrapper does **not** import any paid-API SDK (anthropic / openai /
cohere). Per the SciX Deep Search v1 PRD constraints, all agent calls go
through Claude Code OAuth subprocesses.

Run-id format
-------------
``YYYY-MM-DD_HHMMSS_<6-char-hex>`` (lexicographically sortable).

Transcript files
----------------
- ``question.txt`` — raw question text
- ``answer.md`` — final answer (concatenated text fragments)
- ``tool_calls.jsonl`` — one JSON event per line (preserves dispatcher stream)
- ``metadata.json`` — schema documented in :class:`RunMetadata`

CLI
---
::

    python scripts/scix_deep_search.py "<question>" \\
        [--max-turns N] [--rigor] [--skeptic] [--runs-dir DIR]

Flags
-----
- ``--max-turns N`` (default 25 per PRD amendment A2). Truncation is
  recorded in metadata as ``truncated: true`` when the dispatcher emits
  more turns than requested.
- ``--rigor`` toggles output detail (recorded in metadata; persona-side
  consumption is a future hook). Default OFF per PRD amendment A6.
- ``--skeptic`` raises :class:`NotImplementedError` — SH-2 is gated on
  MH-0 latency probe outcome (PRD amendment A10).

Notes
-----
The "dispatcher" is a callable seam — production uses
:class:`RealDispatcher` which wraps
:func:`scix.eval.persona_judge._run_claude_subprocess`; tests inject a
:class:`FakeDispatcher` that yields canned events without touching OAuth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import sys
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSONA_NAME: str = "deep_search_investigator"
"""Subagent name; matches the slug in ``.claude/agents/<name>.md``."""

PERSONA_PATH_REL: str = ".claude/agents/deep_search_investigator.md"
"""Persona file path relative to repo root (recorded in metadata)."""

DEFAULT_MAX_TURNS: int = 25
"""PRD amendment A2: ≤25 tool turns per investigation."""

DEFAULT_MODEL: str = "sonnet"
"""Recorded in metadata; matches the persona's `model:` frontmatter field."""

DEFAULT_RUNS_DIR: Path = Path.home() / ".scix" / "deep_search_runs"
"""Per task spec; one subdirectory per run."""

# Canonical 19-char ADS bibcode: YYYY (4) + journal-segment (5, padded
# with ``.``) + volume (4, right-justified, may contain letters for
# qualifier journals like Phys.Rev.Lett.) + page-position (4, leading
# letter or ``.`` then digits) + first-author initial (1). Real-world
# examples: ``2011ApJ...730..119R``, ``2014PhRvL.112x1101B``,
# ``2014A&A...568A.103S``, ``1998AJ....116.1009R``. Match by length:
# 4 digits + 14 chars of mixed alpha/digit/``.``/``&`` + uppercase
# initial. False positives on this regex are harmless for
# time-to-first-useful-output.
BIBCODE_RE: re.Pattern[str] = re.compile(
    r"\b\d{4}[A-Za-z&\.\d]{14}[A-Z]\b"
)

SKEPTIC_NOT_IMPLEMENTED_MSG: str = (
    "SH-2 skeptic_subagent_debate not yet implemented "
    "(gated on MH-0 latency probe outcome per amendment A10)"
)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallEvent:
    """One streamed event from the dispatcher.

    The dispatcher emits a flexible JSON shape (matches
    ``claude -p --output-format=stream-json`` events). We keep the shape
    open and pass through the dict unchanged in :class:`RunResult`. This
    DTO is used for type clarity in the test fixtures.
    """

    type: str
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text:
            d["text"] = self.text
        if self.extra:
            d.update(self.extra)
        return d


@dataclass(frozen=True)
class RunMetadata:
    """Schema for ``metadata.json``.

    All fields are required; ``time_to_first_useful_output`` is a float
    (seconds since start) or ``None`` if no bibcode was emitted.
    """

    run_id: str
    question: str
    start_time: str  # ISO-8601 with timezone
    end_time: str
    duration_s: float
    n_turns: int
    max_turns: int
    truncated: bool
    model: str
    persona_path: str
    rigor: bool
    skeptic: bool
    time_to_first_useful_output: float | None


@dataclass(frozen=True)
class RunResult:
    """Result of a single deep-search run.

    Returned by :func:`run_deep_search`. The on-disk transcript is the
    durable artifact; this struct is for in-process callers (tests, REPL).
    """

    run_dir: Path
    answer: str
    metadata: RunMetadata
    tool_calls: tuple[dict[str, Any], ...]


# ---------------------------------------------------------------------------
# Dispatcher protocol + implementations
# ---------------------------------------------------------------------------


class Dispatcher(Protocol):
    """Async-iterator dispatcher.

    A dispatcher is any async callable that yields a stream of event dicts
    given a prompt and a max-turns budget. Production:
    :class:`RealDispatcher`. Tests: pass a :class:`FakeDispatcher`.

    The yielded events are written to ``tool_calls.jsonl`` verbatim;
    events whose ``type`` field is ``"text"`` (or whose dict has a
    ``"text"`` key with no ``"tool"`` key) are concatenated into the
    final ``answer.md``.
    """

    def __call__(self, prompt: str, max_turns: int) -> AsyncIterator[dict[str, Any]]: ...


@dataclass
class RealDispatcher:
    """Production dispatcher — wraps the OAuth ``claude -p`` subprocess.

    Invokes ``claude -p --output-format=stream-json --max-turns N``,
    parses each line of stdout as a JSON event, and yields it. On
    non-zero exit, raises :class:`RuntimeError` with stderr context.

    No paid-API SDK is imported. The ``claude`` binary is the Claude
    Code OAuth client; if absent, raises :class:`FileNotFoundError`.
    """

    claude_binary: str = "claude"
    timeout_s: float = 600.0

    async def __call__(
        self, prompt: str, max_turns: int
    ) -> AsyncIterator[dict[str, Any]]:
        # Local import to keep the test path import-light and to localize
        # the dependency on the persona-judge helper.
        proc = await asyncio.create_subprocess_exec(
            self.claude_binary,
            "-p",
            "--output-format",
            "stream-json",
            "--max-turns",
            str(max_turns),
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None  # for type checkers

        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Stream-json should never emit non-JSON; log and pass
                    # through as a text event so the transcript stays
                    # complete. Don't swallow silently.
                    logger.warning("non-JSON line from claude -p: %s", line[:200])
                    event = {"type": "raw", "text": line}
                yield event
        finally:
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.timeout_s)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"claude -p timed out after {self.timeout_s}s"
                ) from exc

        if proc.returncode != 0:
            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"claude -p exited {proc.returncode}: stderr={stderr!r}"
            )


# ---------------------------------------------------------------------------
# Run-id, paths, prompt builder
# ---------------------------------------------------------------------------


def make_run_id(now: datetime | None = None) -> str:
    """Build a sortable, unique run identifier.

    Format: ``YYYY-MM-DD_HHMMSS_<6-char-hex>``. The hex suffix is from
    :func:`secrets.token_hex` — concurrent invocations don't collide.
    """
    ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{secrets.token_hex(3)}"


def build_prompt(question: str, *, rigor: bool) -> str:
    """Build the prompt sent to ``claude -p``.

    The prompt names the subagent and passes the question. The ``--rigor``
    flag is encoded as a prefix the persona can read; v1 records it but
    does not consume it (per PRD amendment A6 it shifts only the UX layer).
    """
    rigor_marker = "[rigor=on] " if rigor else ""
    return (
        f"Use the {PERSONA_NAME} subagent to investigate the following "
        f"question. Answer with bibcode-anchored evidence per the persona's "
        f"Linking discipline. {rigor_marker}\n\n"
        f"Question: {question}"
    )


def _resolve_persona_path() -> str:
    """Best-effort absolute path to the persona file (for metadata)."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    candidate = repo_root / PERSONA_PATH_REL
    return str(candidate)


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------


async def _stream_run(
    dispatcher: Dispatcher,
    prompt: str,
    *,
    max_turns: int,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    float | None,
    float,
    bool,
]:
    """Drive the dispatcher and capture the event stream.

    Returns:
        events: list of event dicts (truncated to ``max_turns`` entries).
        text_fragments: list of text strings to concatenate as the answer.
        time_to_first: seconds from start to first bibcode-bearing event,
            or None if no bibcode appeared.
        elapsed_s: total wall-clock elapsed.
        truncated: True if the dispatcher emitted >max_turns events.
    """
    events: list[dict[str, Any]] = []
    text_fragments: list[str] = []
    time_to_first: float | None = None
    truncated = False

    start = time.monotonic()
    async for event in dispatcher(prompt, max_turns):
        if len(events) >= max_turns:
            truncated = True
            break
        events.append(event)

        # Pull the textual payload for the answer accumulator.
        text = event.get("text", "") if isinstance(event, dict) else ""
        if isinstance(text, str) and text:
            # Treat events that have a "tool" or "tool_name" key as tool
            # events — exclude them from the human-readable answer.
            is_tool_event = any(k in event for k in ("tool", "tool_name", "tool_use"))
            if not is_tool_event:
                text_fragments.append(text)

        # First-bibcode detection — scan the whole event JSON, not just
        # the text field, so tool results count toward "first useful
        # output" (per spec: "first bibcode-anchored fragment is emitted").
        if time_to_first is None:
            haystack = json.dumps(event) if isinstance(event, dict) else str(event)
            if BIBCODE_RE.search(haystack):
                time_to_first = time.monotonic() - start

    elapsed = time.monotonic() - start
    return events, text_fragments, time_to_first, elapsed, truncated


def _write_transcript(
    run_dir: Path,
    *,
    question: str,
    answer: str,
    tool_calls: Iterable[dict[str, Any]],
    metadata: RunMetadata,
) -> None:
    """Persist the four transcript files.

    Creates ``run_dir`` (parents OK). Files written with mode 0644 by
    default; metadata.json + answer.md are utf-8 with trailing newline.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "question.txt").write_text(question + "\n", encoding="utf-8")
    answer_payload = answer + ("\n" if not answer.endswith("\n") else "")
    (run_dir / "answer.md").write_text(answer_payload, encoding="utf-8")
    with (run_dir / "tool_calls.jsonl").open("w", encoding="utf-8") as fh:
        for ev in tool_calls:
            fh.write(json.dumps(ev, ensure_ascii=False))
            fh.write("\n")
    (run_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_deep_search(
    question: str,
    dispatcher: Dispatcher,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    rigor: bool = False,
    skeptic: bool = False,
    runs_dir: Path | None = None,
    persona_path: str | None = None,
    model: str = DEFAULT_MODEL,
    run_id: str | None = None,
) -> RunResult:
    """Synchronous entry point.

    Builds the prompt, runs the dispatcher's async iterator, and writes
    the transcript directory. Returns :class:`RunResult` for in-process
    callers; the on-disk artifacts are the durable output.

    Raises:
        NotImplementedError: when ``skeptic=True`` (SH-2 deferred).
    """
    if skeptic:
        raise NotImplementedError(SKEPTIC_NOT_IMPLEMENTED_MSG)

    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    run_id = run_id or make_run_id()
    persona_path = persona_path or _resolve_persona_path()
    prompt = build_prompt(question, rigor=rigor)

    start_dt = datetime.now(timezone.utc)
    events, fragments, time_to_first, elapsed, truncated = asyncio.run(
        _stream_run(dispatcher, prompt, max_turns=max_turns)
    )
    end_dt = datetime.now(timezone.utc)

    answer = "".join(fragments).strip()
    metadata = RunMetadata(
        run_id=run_id,
        question=question,
        start_time=start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        duration_s=round(elapsed, 3),
        n_turns=len(events),
        max_turns=max_turns,
        truncated=truncated,
        model=model,
        persona_path=persona_path,
        rigor=rigor,
        skeptic=skeptic,
        time_to_first_useful_output=(
            None if time_to_first is None else round(time_to_first, 3)
        ),
    )

    run_dir = runs_dir / run_id
    _write_transcript(
        run_dir,
        question=question,
        answer=answer,
        tool_calls=events,
        metadata=metadata,
    )

    return RunResult(
        run_dir=run_dir,
        answer=answer,
        metadata=metadata,
        tool_calls=tuple(events),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Returns an :class:`argparse.Namespace` with attributes:
    ``question``, ``max_turns``, ``rigor``, ``skeptic``, ``runs_dir``.
    """
    parser = argparse.ArgumentParser(
        prog="scix_deep_search",
        description=(
            "SciX Deep Search v1 — investigation-shaped subagent that "
            "answers a question with bibcode-anchored evidence."
        ),
    )
    parser.add_argument("question", help="Question to investigate (single argv).")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max tool turns (default {DEFAULT_MAX_TURNS}; PRD A2).",
    )
    parser.add_argument(
        "--rigor",
        action="store_true",
        default=False,
        help="Toggle rigor (extra detail). Default OFF (PRD A6).",
    )
    parser.add_argument(
        "--skeptic",
        action="store_true",
        default=False,
        help=(
            "Toggle SH-2 skeptic debate. Currently raises NotImplementedError "
            "(gated on MH-0 latency probe; PRD A10)."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Output directory (default {DEFAULT_RUNS_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console entry point."""
    args = parse_args(argv)

    if args.skeptic:
        raise NotImplementedError(SKEPTIC_NOT_IMPLEMENTED_MSG)

    dispatcher = RealDispatcher()
    result = run_deep_search(
        args.question,
        dispatcher,
        max_turns=args.max_turns,
        rigor=args.rigor,
        skeptic=args.skeptic,
        runs_dir=args.runs_dir,
    )

    # Print the answer to stdout for callers who pipe; the transcript
    # directory location goes to stderr to keep stdout clean.
    print(result.answer)
    print(f"Transcript: {result.run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    sys.exit(main())
