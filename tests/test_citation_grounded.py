"""Unit tests for ``scix.citation_grounded`` (PRD MH-6 + amendment A6).

These tests use a deterministic count-vector embedder (vocab built from the
inputs of each call) so cosine similarity is reproducible without loading
INDUS. All tests are pure — no DB, no network, no model load.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pytest

from scix.citation_grounded import (
    DEFAULT_THRESHOLD,
    INLINE_MARKER,
    GroundingReport,
    RevisedDraft,
    grounded_check,
    revise_with_gate,
    set_embedder,
)


# ---------------------------------------------------------------------------
# Test embedder: count-vector over the call's own vocab
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "we",
        "they",
        "it",
        "from",
        "as",
        "than",
        "no",
    }
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def count_vector_embedder(texts: list[str]) -> list[list[float]]:
    """Build a count-vector over the call's own vocabulary.

    For each call, we collect the union of content tokens, then encode
    each input text as a vector of token frequencies. Cosine similarity
    on these vectors is high when texts share content words and low
    otherwise — a useful proxy for paraphrase similarity in unit tests.
    """
    vocab: list[str] = []
    seen: set[str] = set()
    tokenized: list[list[str]] = []
    for text in texts:
        toks = _tokens(text)
        tokenized.append(toks)
        for t in toks:
            if t not in seen:
                seen.add(t)
                vocab.append(t)

    vectors: list[list[float]] = []
    for toks in tokenized:
        counts = Counter(toks)
        vec = [float(counts.get(v, 0)) for v in vocab]
        vectors.append(vec)
    return vectors


def zero_embedder(texts: list[str]) -> list[list[float]]:
    """Embedder that returns all-zero vectors (cosine always 0).

    Used to confirm the substring short-circuit beats the embedding path.
    """
    return [[0.0, 0.0, 0.0] for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _install_count_embedder():
    """Default embedder for every test in this module."""
    set_embedder(count_vector_embedder)
    yield
    set_embedder(None)


# ---------------------------------------------------------------------------
# RevisionDispatcher fakes
# ---------------------------------------------------------------------------


class FakeRevisionDispatcher:
    """Records calls and returns canned drafts.

    If ``drafts`` is exhausted, returns the last draft unchanged
    (simulates a persona that has nothing more to add).
    """

    def __init__(self, drafts: list[str]) -> None:
        self.drafts = list(drafts)
        self.calls: list[tuple[str, list[str]]] = []
        self._last: str = ""

    def revise(self, draft: str, unmatched: list[str]) -> str:
        self.calls.append((draft, list(unmatched)))
        if self.drafts:
            self._last = self.drafts.pop(0)
        return self._last


class IdentityRevisionDispatcher:
    """Returns the draft verbatim — never adds support."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def revise(self, draft: str, unmatched: list[str]) -> str:
        self.calls.append((draft, list(unmatched)))
        return draft


# ---------------------------------------------------------------------------
# (a) grounded=True when every assertion matches a tool result quote
# ---------------------------------------------------------------------------


def test_grounded_true_when_substring_matches() -> None:
    """An assertion that is a substring of a tool quote is grounded."""
    # Assertion text appears verbatim inside the quote (after the
    # surrounding clause is added in the tool result).
    draft = "We measure H0 = 73.0 km/s/Mpc."
    tool_results = [
        {
            "quote": (
                "Our SH0ES analysis: We measure H0 = 73.0 km/s/Mpc. "
                "This uses Cepheid distances."
            )
        },
    ]

    report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

    assert report.grounded is True
    assert report.unmatched == ()
    assert len(report.assertions) >= 1
    assert report.threshold_used == DEFAULT_THRESHOLD


def test_grounded_true_via_paraphrase() -> None:
    """A paraphrase with high content-word overlap is grounded."""
    draft = "Riess and collaborators measured the Hubble constant to be 73 km/s/Mpc."
    tool_results = [
        {
            "text": (
                "Riess and collaborators measured the Hubble constant to be 73 km/s/Mpc "
                "with calibrated Cepheid distances."
            )
        },
    ]

    report = grounded_check(draft, tool_results, threshold=0.5)

    assert report.grounded is True


# ---------------------------------------------------------------------------
# (b) grounded=False with unmatched listed when an assertion has no support
# ---------------------------------------------------------------------------


def test_grounded_false_when_unsupported() -> None:
    """An assertion absent from the tool results is unmatched."""
    draft = (
        "The Planck collaboration reports H0 = 67.4 km/s/Mpc. "
        "Aliens built the pyramids in 2024."
    )
    tool_results = [
        {"quote": "Planck collaboration reports H0 = 67.4 km/s/Mpc from CMB."},
    ]

    report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

    assert report.grounded is False
    assert len(report.unmatched) == 1
    assert "Aliens" in report.unmatched[0]


# ---------------------------------------------------------------------------
# (c) revise_with_gate halts at max_revisions=2 and strips residuals
# ---------------------------------------------------------------------------


def test_revise_with_gate_halts_at_max_revisions() -> None:
    """When revisions never substantiate, the loop halts and strips residuals."""
    draft = "Wombats demonstrate quantum gravity at 5 sigma."
    tool_results = [{"text": "Einstein published the special theory of relativity in 1905."}]
    dispatcher = IdentityRevisionDispatcher()

    result = revise_with_gate(
        draft,
        tool_results,
        dispatcher,
        max_revisions=2,
        rigor_mode=False,
    )

    assert result.revision_count == 2
    assert len(dispatcher.calls) == 2
    assert result.grounded is False
    assert "Wombats demonstrate quantum gravity" in result.stripped[0]


def test_revise_with_gate_succeeds_within_budget() -> None:
    """If a revision substantiates the draft, the loop exits early."""
    initial_draft = "The wombat invented relativity in 1905."
    fixed_draft = (
        "Einstein published the special theory of relativity in 1905."
    )
    tool_results = [
        {"text": "Einstein published the special theory of relativity in 1905."}
    ]
    dispatcher = FakeRevisionDispatcher([fixed_draft])

    result = revise_with_gate(
        initial_draft,
        tool_results,
        dispatcher,
        max_revisions=2,
        threshold=0.5,
    )

    assert result.grounded is True
    assert result.revision_count == 1
    assert result.stripped == ()
    assert result.answer == fixed_draft


# ---------------------------------------------------------------------------
# (d) rigor_mode toggles inline-vs-footnote
# ---------------------------------------------------------------------------


def test_rigor_mode_inline_marker() -> None:
    """rigor_mode=True replaces stripped sentences with the inline literal."""
    draft = "Wombats demonstrate quantum gravity at 5 sigma."
    tool_results = [{"text": "Einstein established gravity as spacetime curvature."}]
    dispatcher = IdentityRevisionDispatcher()

    result = revise_with_gate(
        draft,
        tool_results,
        dispatcher,
        max_revisions=2,
        rigor_mode=True,
    )

    assert INLINE_MARKER in result.answer
    assert "Footnotes" not in result.answer
    # Default mode markers should not appear in rigor mode.
    assert "¹" not in result.answer
    assert result.grounded is False


def test_default_mode_footnoted() -> None:
    """rigor_mode=False replaces stripped sentences with superscript markers + footnotes."""
    draft = "Wombats demonstrate quantum gravity at 5 sigma."
    tool_results = [{"text": "Einstein established gravity as spacetime curvature."}]
    dispatcher = IdentityRevisionDispatcher()

    result = revise_with_gate(
        draft,
        tool_results,
        dispatcher,
        max_revisions=2,
        rigor_mode=False,
    )

    assert "¹" in result.answer
    assert "Footnotes" in result.answer
    assert INLINE_MARKER not in result.answer
    assert "Wombats demonstrate quantum gravity" in result.answer  # in footnote


def test_default_mode_multiple_footnotes() -> None:
    """Multiple stripped sentences get distinct superscript markers."""
    draft = (
        "Wombats demonstrate dark matter in 1924. "
        "Penguins detected the Higgs at 9 sigma. "
        "We measure H0 = 73 from SH0ES."
    )
    tool_results = [{"text": "We measure H0 = 73 from SH0ES Cepheid distances."}]
    dispatcher = IdentityRevisionDispatcher()

    result = revise_with_gate(
        draft,
        tool_results,
        dispatcher,
        max_revisions=1,
        rigor_mode=False,
        threshold=0.5,
    )

    assert "¹" in result.answer
    assert "²" in result.answer
    assert result.answer.count("Footnotes") == 1


# ---------------------------------------------------------------------------
# (e) substring short-circuit returns higher score than embedding-only path
# ---------------------------------------------------------------------------


def test_substring_shortcircuit_beats_zero_embedder() -> None:
    """With a zero-vector embedder, a pure-embedding match would fail.

    But because the assertion is a substring of the tool quote, the
    short-circuit returns 0.95 and the assertion is grounded.
    """
    set_embedder(zero_embedder)
    try:
        # Assertion is a verbatim sentence inside the tool quote.
        draft = "The detection was at 5 sigma significance."
        tool_results = [
            {
                "quote": (
                    "Background: The detection was at 5 sigma significance. "
                    "We attribute this to the new instrument."
                )
            },
        ]

        report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

        assert report.grounded is True, (
            "Substring short-circuit should have returned 0.95, "
            "exceeding the 0.82 threshold despite the zero embedder."
        )
    finally:
        set_embedder(count_vector_embedder)


def test_no_match_with_zero_embedder_and_no_substring() -> None:
    """Without substring overlap and a zero embedder, scores are 0 and grounding fails."""
    set_embedder(zero_embedder)
    try:
        draft = "The wombat orbits Pluto."
        tool_results = [{"text": "Penguins eat fish."}]

        report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

        assert report.grounded is False
    finally:
        set_embedder(count_vector_embedder)


# ---------------------------------------------------------------------------
# Assertion parser sanity
# ---------------------------------------------------------------------------


def test_assertion_parser_filters_non_claims() -> None:
    """Sentences without claim content are not subject to the gate."""
    # No verbs, no numbers, no entities, no citations — just a header-like phrase.
    draft = "thoughts here."
    tool_results: list[dict] = []

    report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

    # No assertions => grounded vacuously True.
    assert report.assertions == ()
    assert report.grounded is True


def test_assertion_parser_detects_numeric_claim() -> None:
    """A numeric claim is flagged as an assertion even without content verbs."""
    draft = "H0 = 73.0 km/s/Mpc."
    tool_results: list[dict] = []  # no support

    report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

    assert len(report.assertions) == 1
    assert report.grounded is False  # no tool support


def test_assertion_parser_detects_named_entity() -> None:
    """A mid-sentence uppercase token (named entity) flags the sentence."""
    draft = "We confirm Riess findings."
    tool_results: list[dict] = []

    report = grounded_check(draft, tool_results, threshold=DEFAULT_THRESHOLD)

    assert len(report.assertions) == 1


# ---------------------------------------------------------------------------
# Tool-result extraction
# ---------------------------------------------------------------------------


def test_extract_handles_nested_results() -> None:
    """Nested ``results`` lists with dict items are flattened."""
    draft = "The H0 measurement was 73 km/s/Mpc."
    tool_results = [
        {
            "results": [
                {"snippet": "H0 measurement was 73 km/s/Mpc reported by SH0ES."},
                {"snippet": "Other unrelated text."},
            ]
        }
    ]

    report = grounded_check(draft, tool_results, threshold=0.5)

    assert report.grounded is True


# ---------------------------------------------------------------------------
# DTO basics
# ---------------------------------------------------------------------------


def test_grounding_report_is_frozen() -> None:
    """GroundingReport is a frozen dataclass (immutability)."""
    report = grounded_check("Trivial.", [], threshold=DEFAULT_THRESHOLD)
    assert isinstance(report, GroundingReport)
    with pytest.raises(Exception):
        report.grounded = True  # type: ignore[misc]


def test_revised_draft_is_frozen() -> None:
    """RevisedDraft is a frozen dataclass."""
    dispatcher = IdentityRevisionDispatcher()
    result = revise_with_gate("Trivial.", [], dispatcher)
    assert isinstance(result, RevisedDraft)
    with pytest.raises(Exception):
        result.answer = "x"  # type: ignore[misc]
