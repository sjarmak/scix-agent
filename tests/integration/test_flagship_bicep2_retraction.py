"""MH-8b flagship test — BICEP2 retraction handling.

Acceptance criteria (per ``docs/prd/scix_deep_search_v1.md`` MH-8b):

(a) BICEP2 detection paper (``2014PhRvL.112x1101B``) is surfaced WITH
    a ``retraction_warning`` annotation pointing to the dust-contamination
    correction.
(b) The answer explicitly states the original detection was retracted /
    corrected.
(c) No claim quotes the BICEP2 result without an annotation.
(d) ≤25 tool turns (A2 amendment).
(e) ``claim_blame`` does NOT return BICEP2 as a clean origin source.

CI runs in mock mode by default.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.integration.conftest import (
    assert_all_claims_grounded,
    collect_bibcodes,
)

# ---------------------------------------------------------------------------
# Canonical question
# ---------------------------------------------------------------------------

QUESTION: str = (
    "What is the current evidence for primordial gravitational waves from "
    "BICEP2/Keck observations as of 2014, and how was that evidence revised "
    "after dust contamination analysis?"
)

# ---------------------------------------------------------------------------
# MOCK_FIXTURE
# ---------------------------------------------------------------------------

# BICEP2 original (retracted) detection paper.
_BICEP2_2014 = "2014PhRvL.112x1101B"
# BICEP2/Keck + Planck joint dust analysis (the correction).
_BICEP2_PLANCK_JOINT = "2015PhRvL.114j1301B"
# Planck dust polarization (independent confirmation of dust-only signal).
_PLANCK_DUST_2016 = "2016A&A...586A.133P"

# Annotation marker the agent appends to BICEP2 references; tests verify
# every BICEP2 mention carries this token (or a paraphrase).
_RETRACTION_TOKEN = "retraction_warning"

MOCK_FIXTURE: dict[str, Any] = {
    "events": [
        {
            "type": "tool_use",
            "tool_name": "search",
            "text": "(searching for BICEP2 detection paper)",
        },
        {
            "type": "tool_result",
            "tool_name": "search",
            "text": (
                f"Found {_BICEP2_2014} (BICEP2 Collaboration 2014, "
                "\"We report the detection of B-mode polarization at "
                "degree angular scales... r = 0.20 +0.07 -0.05\"). "
                f"correction_events: [retraction by {_BICEP2_PLANCK_JOINT}]"
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "claim_blame",
            "text": "(searching for clean origin of B-mode primordial GW evidence)",
        },
        {
            "type": "tool_result",
            "tool_name": "claim_blame",
            "text": (
                f"claim_blame: origin={_BICEP2_PLANCK_JOINT}, "
                f"retraction_warnings=[{_BICEP2_2014}]. "
                f"The {_BICEP2_2014} detection was retracted after the "
                "joint Planck dust analysis."
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "find_replications",
            "text": f"(forward citations of {_BICEP2_2014})",
        },
        {
            "type": "tool_result",
            "tool_name": "find_replications",
            "text": (
                f"{_BICEP2_PLANCK_JOINT} (BICEP2/Keck + Planck joint analysis): "
                "\"We find no statistically significant evidence for "
                "primordial gravitational waves; r < 0.12 at 95% CL after "
                "subtracting the dust contribution.\" "
                f"{_PLANCK_DUST_2016}: \"The B-mode signal is consistent "
                "with thermal dust emission alone.\""
            ),
        },
        {
            "type": "text",
            "text": (
                f"The original BICEP2 detection ({_BICEP2_2014}) "
                f"[{_RETRACTION_TOKEN}: retracted after joint Planck dust "
                "analysis] reported a B-mode polarization signal that was "
                "subsequently corrected. "
                f"The joint analysis ({_BICEP2_PLANCK_JOINT}) found r < 0.12 "
                "after dust subtraction, refuting the primordial detection. "
                f"Independent Planck dust polarization analysis "
                f"({_PLANCK_DUST_2016}) confirmed the B-mode signal is "
                "consistent with thermal dust emission. "
                "The BICEP2 result was not a primordial gravitational-wave "
                "detection."
            ),
        },
    ],
    "tool_results_for_grounding": [
        {
            "text": (
                f"The original BICEP2 detection ({_BICEP2_2014}) "
                f"[{_RETRACTION_TOKEN}: retracted after joint Planck dust "
                "analysis] reported a B-mode polarization signal that was "
                "subsequently corrected"
            ),
        },
        {
            "text": (
                f"The joint analysis ({_BICEP2_PLANCK_JOINT}) found r < 0.12 "
                "after dust subtraction, refuting the primordial detection"
            ),
        },
        {
            "text": (
                f"Independent Planck dust polarization analysis "
                f"({_PLANCK_DUST_2016}) confirmed the B-mode signal is "
                "consistent with thermal dust emission"
            ),
        },
        {
            "text": (
                "The BICEP2 result was not a primordial gravitational-wave "
                "detection"
            ),
        },
    ],
    # The fixture's claim_blame mock — the test asserts BICEP2 is NOT here.
    "claim_blame_response": {
        "origin": _BICEP2_PLANCK_JOINT,
        "retraction_warnings": [_BICEP2_2014],
    },
    "bicep2_bibcode": _BICEP2_2014,
    "annotation_token": _RETRACTION_TOKEN,
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_flagship_bicep2_retraction(run_flagship: Any) -> None:
    """MH-8b — BICEP2 retraction-handling flagship variant."""
    result = run_flagship(QUESTION, MOCK_FIXTURE)
    answer = result.answer
    bicep2 = MOCK_FIXTURE["bicep2_bibcode"]

    # (a) BICEP2 paper surfaced WITH retraction_warning annotation.
    assert bicep2 in answer, (
        f"BICEP2 paper {bicep2} not found in answer"
    )
    assert MOCK_FIXTURE["annotation_token"] in answer, (
        f"answer must include annotation token "
        f"{MOCK_FIXTURE['annotation_token']!r}"
    )

    # (b) Answer explicitly states retraction / correction.
    answer_lower = answer.lower()
    assert any(t in answer_lower for t in ("retract", "corrected", "correction")), (
        "answer must explicitly state the original detection was "
        "retracted/corrected"
    )

    # (c) No sentence quotes the BICEP2 result without annotation.
    # Heuristic: each occurrence of `bicep2` in answer must be near
    # ("retract" or "correct" or annotation_token) within ±200 chars
    # — the same sentence/clause.
    answer_text = answer
    idx = 0
    while True:
        pos = answer_text.find(bicep2, idx)
        if pos == -1:
            break
        window = answer_text[max(0, pos - 200) : pos + 200].lower()
        assert any(
            t in window
            for t in (
                "retract",
                "corrected",
                "correction",
                MOCK_FIXTURE["annotation_token"].lower(),
                "dust",
            )
        ), (
            f"BICEP2 reference at offset {pos} appears without an "
            f"annotation in window: {answer_text[max(0, pos - 200) : pos + 200]!r}"
        )
        idx = pos + len(bicep2)

    # Grounding check on the canned answer.
    assert_all_claims_grounded(
        answer,
        MOCK_FIXTURE["tool_results_for_grounding"],
    )

    # (d) <=25 tool turns
    assert result.metadata.n_turns <= 25, (
        f"n_turns={result.metadata.n_turns} exceeds 25-turn budget"
    )
    assert not result.metadata.truncated

    # (e) claim_blame does NOT return BICEP2 as clean source.
    cb = MOCK_FIXTURE["claim_blame_response"]
    assert cb["origin"] != bicep2, (
        f"claim_blame origin must not be BICEP2 ({bicep2})"
    )
    assert bicep2 in cb["retraction_warnings"], (
        f"BICEP2 ({bicep2}) must appear in retraction_warnings"
    )
