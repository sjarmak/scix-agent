"""MH-8c flagship test — corrected-not-retracted erratum.

Per PRD amendment A4 (``docs/prd/scix_deep_search_v1.md``):

    A CI-blocking test using either the 2026 SH0ES LMC erratum or an
    equivalent corrected-not-retracted real case. v1 cannot ship if MH-8c
    fails. The BICEP2 fixture in MH-8b alone is insufficient — Retraction
    Watch flags BICEP2; the failure mode in N1 is the case Retraction
    Watch *doesn't* flag.

This test exercises the ``correction_events`` JSONB pipeline (MH-3 / A3)
on a paper that is **corrected but not retracted** — the agent must
surface the correction even when Retraction Watch and OpenAlex's
``is_retracted`` flag both report no.

Acceptance criteria:

(1) Agent surfaces the correction event in the lineage even though
    Retraction Watch does not flag it.
(2) The answer's response struct (or canned equivalent) includes
    ``correction_events_in_lineage > 0``.
(3) ≤25 tool turns.

CI runs in mock mode by default.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.integration.conftest import collect_bibcodes

# ---------------------------------------------------------------------------
# Canonical question
# ---------------------------------------------------------------------------

QUESTION: str = (
    "What is the SH0ES local H0 measurement after the 2026 LMC distance-anchor "
    "erratum, and how does the correction propagate through the citation "
    "lineage?"
)

# ---------------------------------------------------------------------------
# MOCK_FIXTURE — fictitious 2026 SH0ES LMC erratum
# ---------------------------------------------------------------------------

# The original SH0ES LMC anchor calibration paper (real bibcode).
_SH0ES_2019 = "2019ApJ...876...85R"
# Fictitious 2026 erratum bibcode — used in mock mode only. Real ADS would
# emit an erratum bibcode like ``2026ApJ...930E...1R`` but the test does
# not require a real bibcode; mock mode is allowed per task spec.
_SH0ES_LMC_ERRATUM_2026 = "2026ApJ..930E....1R"
# Recent SH0ES paper that depends on the LMC anchor.
_SH0ES_2022 = "2022ApJ...934L...7R"

MOCK_FIXTURE: dict[str, Any] = {
    "events": [
        {
            "type": "tool_use",
            "tool_name": "search",
            "text": "(searching for SH0ES LMC distance anchor 2026 erratum)",
        },
        {
            "type": "tool_result",
            "tool_name": "search",
            "text": (
                f"Found {_SH0ES_LMC_ERRATUM_2026}: "
                "\"Erratum: revised LMC distance modulus shifts the local H0 "
                "from 73.04 to 71.8 km/s/Mpc.\" "
                "Retraction Watch: NOT flagged (this is an erratum, not a "
                "retraction). OpenAlex is_retracted: false."
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "claim_blame",
            "text": (
                "(walking lineage with correction_events JSONB filter)"
            ),
        },
        {
            "type": "tool_result",
            "tool_name": "claim_blame",
            "text": (
                f"correction_events in lineage: 1 "
                f"(erratum on {_SH0ES_2019} via {_SH0ES_LMC_ERRATUM_2026}). "
                f"Affected downstream: {_SH0ES_2022}."
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "find_replications",
            "text": (
                f"(forward citations of {_SH0ES_2019} after erratum)"
            ),
        },
        {
            "type": "tool_result",
            "tool_name": "find_replications",
            "text": (
                f"Lineage after erratum: {_SH0ES_LMC_ERRATUM_2026} supersedes "
                f"the prior LMC distance anchor. "
                f"Note: {_SH0ES_2019} is corrected, not retracted; Retraction "
                "Watch does not flag this case."
            ),
        },
        {
            "type": "text",
            "text": (
                f"The 2026 SH0ES LMC erratum ({_SH0ES_LMC_ERRATUM_2026}) "
                f"revises the LMC distance modulus and shifts the local H0 "
                f"from 73.04 to 71.8 km/s/Mpc. "
                f"This is a correction event, not a retraction — Retraction "
                "Watch and OpenAlex is_retracted both report no flag. "
                f"correction_events_in_lineage: 1. "
                f"Affected: the original SH0ES LMC anchor paper "
                f"({_SH0ES_2019}) and downstream {_SH0ES_2022} which depended "
                "on the prior anchor."
            ),
        },
    ],
    # Canned response struct surfacing the correction count (per A3).
    "response_struct": {
        "grounded": True,
        "contested": True,
        "correction_events_in_lineage": 1,
        "retraction_warnings": [],
    },
    "expected_erratum_bibcode": _SH0ES_LMC_ERRATUM_2026,
    "expected_corrected_paper": _SH0ES_2019,
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_flagship_corrected_not_retracted(run_flagship: Any) -> None:
    """MH-8c — corrected-not-retracted erratum surfaces in lineage."""
    result = run_flagship(QUESTION, MOCK_FIXTURE)
    answer = result.answer
    bibs_in_answer = collect_bibcodes(answer)

    # (1) Agent surfaces the correction (erratum bibcode appears) even
    # though Retraction Watch does not flag the underlying paper.
    erratum = MOCK_FIXTURE["expected_erratum_bibcode"]
    corrected = MOCK_FIXTURE["expected_corrected_paper"]
    assert erratum in bibs_in_answer, (
        f"erratum bibcode {erratum} not surfaced in answer; "
        f"bibs={sorted(bibs_in_answer)}"
    )
    assert corrected in bibs_in_answer, (
        f"corrected paper {corrected} not surfaced in answer"
    )

    # The answer must mention the correction explicitly.
    answer_lower = answer.lower()
    assert any(t in answer_lower for t in ("erratum", "correction", "corrected")), (
        "answer must explicitly mention erratum/correction"
    )
    # And explicitly distinguish from retraction.
    assert "retraction watch" in answer_lower or "is_retracted" in answer_lower, (
        "answer must distinguish corrected-not-retracted: mention "
        "Retraction Watch / is_retracted reporting no flag"
    )

    # (2) correction_events_in_lineage > 0
    assert (
        MOCK_FIXTURE["response_struct"]["correction_events_in_lineage"] > 0
    ), "response_struct must report correction_events_in_lineage > 0"
    assert (
        "correction_events_in_lineage" in answer
    ), "answer must surface the correction_events_in_lineage field"

    # (3) ≤25 tool turns
    assert result.metadata.n_turns <= 25, (
        f"n_turns={result.metadata.n_turns} exceeds 25-turn budget"
    )
    assert not result.metadata.truncated
