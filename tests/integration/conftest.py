"""Shared fixtures for SciX Deep Search v1 flagship integration tests.

These tests exercise the four CI-blocking flagship questions
(MH-8a/b/c/d) from the SciX Deep Search v1 PRD
(``docs/prd/scix_deep_search_v1.md``).

CI runs in **mock mode by default** (env var ``SCIX_FLAGSHIP_MOCK=1`` is
the implicit default). Mock mode swaps the production
:class:`scix_deep_search.RealDispatcher` for a per-test
:class:`MockDispatcher` whose canned events satisfy the assertions —
this validates the assertion logic, not the agent itself.

Live runs (``SCIX_FLAGSHIP_MOCK=0``) are operator-triggered and burn
OAuth budget. Live runs are skipped in CI by default.

Why a fixture-level mock instead of monkeypatching ``RealDispatcher``?
Because the wrapper takes the dispatcher as a constructor argument (DI
seam), so tests can simply pass a :class:`MockDispatcher` without
patching anything. This matches the pattern in
``tests/test_scix_deep_search.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module loader — the wrapper script lives outside the ``src/scix``
# package so we import it by path. This matches how the script is
# invoked in production (``python scripts/scix_deep_search.py``).
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_SDS_PATH: Path = _REPO_ROOT / "scripts" / "scix_deep_search.py"

_spec = importlib.util.spec_from_file_location("scix_deep_search", _SDS_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"could not load scix_deep_search from {_SDS_PATH}"
)
_sds = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("scix_deep_search", _sds)
_spec.loader.exec_module(_sds)


# ---------------------------------------------------------------------------
# MockDispatcher — replays a canned event sequence
# ---------------------------------------------------------------------------


class MockDispatcher:
    """Async-iterator dispatcher that yields a canned event sequence.

    Drop-in replacement for :class:`scix_deep_search.RealDispatcher`
    that bypasses the OAuth subprocess. The events are the dispatcher's
    streamed dicts as documented in the wrapper module.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events: list[dict[str, Any]] = list(events)
        self.last_prompt: str | None = None
        self.last_max_turns: int | None = None

    async def __call__(
        self, prompt: str, max_turns: int
    ) -> AsyncIterator[dict[str, Any]]:
        self.last_prompt = prompt
        self.last_max_turns = max_turns
        for ev in self._events:
            yield ev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sds_module() -> Any:
    """The loaded ``scix_deep_search`` module."""
    return _sds


@pytest.fixture(scope="session")
def mock_mode_enabled() -> bool:
    """Whether mock mode is on.

    Default: True (CI mode). Set ``SCIX_FLAGSHIP_MOCK=0`` to enable
    live runs (operator-only).
    """
    return os.environ.get("SCIX_FLAGSHIP_MOCK", "1") != "0"


@pytest.fixture
def run_flagship(
    tmp_path: Path,
    mock_mode_enabled: bool,
    sds_module: Any,
):
    """Run a flagship question with a canned-event mock dispatcher.

    Skips when ``SCIX_FLAGSHIP_MOCK=0`` is set — live runs are
    operator-triggered and not part of CI.

    Returns a callable ``run(question, fixture)`` that returns a
    :class:`scix_deep_search.RunResult`.
    """

    def _run(question: str, fixture: dict[str, Any]):
        if not mock_mode_enabled:
            pytest.skip(
                "SCIX_FLAGSHIP_MOCK=0; live runs are operator-triggered"
            )
        events = fixture["events"]
        dispatcher = MockDispatcher(events)
        return sds_module.run_deep_search(
            question,
            dispatcher,
            runs_dir=tmp_path,
            max_turns=fixture.get("max_turns", 25),
        )

    return _run


# ---------------------------------------------------------------------------
# Substring-grounded check helper (no INDUS load required)
# ---------------------------------------------------------------------------


def collect_bibcodes(text: str) -> set[str]:
    """Return the set of canonical 19-char bibcodes in ``text``.

    Reuses the wrapper's :data:`scix_deep_search.BIBCODE_RE` so the
    flagship tests and the wrapper share one definition.
    """
    return set(_sds.BIBCODE_RE.findall(text))


def collect_bibcodes_from_events(
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> set[str]:
    """Walk every string value in every event and harvest bibcodes.

    Used by MH-8d to confirm ``>= 3`` distinct prior-art bibcodes were
    surfaced via tool calls.
    """
    import json

    bibs: set[str] = set()
    for ev in events:
        try:
            blob = json.dumps(ev, ensure_ascii=False)
        except (TypeError, ValueError):
            continue
        bibs.update(_sds.BIBCODE_RE.findall(blob))
    return bibs


def assert_all_claims_grounded(
    answer: str,
    tool_results_for_grounding: list[dict[str, Any]],
    threshold: float = 0.82,
) -> None:
    """Assert every assertion in ``answer`` is substring-grounded in
    ``tool_results_for_grounding``.

    Uses :func:`scix.citation_grounded.grounded_check` with the
    substring-only path (an embedder that returns zero vectors so any
    non-substring assertion is forced through the substring path —
    which short-circuits to ``SUBSTRING_SCORE=0.95``). This avoids the
    INDUS load in tests.

    The caller is responsible for ensuring the canned tool_results
    contain the substrings the canned answer asserts.
    """
    from scix import citation_grounded

    # Force the substring path: an embedder that returns zero-magnitude
    # vectors yields cosine 0.0 for the embedding path. Combined with
    # the threshold, only substring-matched assertions can be grounded
    # — which is exactly the invariant the mock fixtures satisfy.
    def _zero_embedder(texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    citation_grounded.set_embedder(_zero_embedder)
    try:
        report = citation_grounded.grounded_check(
            answer,
            tool_results_for_grounding,
            threshold=threshold,
        )
    finally:
        citation_grounded.set_embedder(None)

    assert report.grounded, (
        f"ungrounded sentences found in answer: {list(report.unmatched)}"
    )


__all__ = [
    "MockDispatcher",
    "assert_all_claims_grounded",
    "collect_bibcodes",
    "collect_bibcodes_from_events",
]
