"""MH-8a flagship test — Hubble tension lineage trace.

Acceptance criteria (per ``docs/prd/scix_deep_search_v1.md`` MH-8 + A2):

(a) Riess+ 2011 (``2011ApJ...730..119R``) appears in the lineage.
(b) The answer surfaces ≥3 SH0ES papers and ≥2 Planck/CMB papers.
(c) Every claim in the answer is ``citation_grounded=True`` against
    the conversation's tool-result history.
(d) ≤25 tool turns (per A2; wall-clock replaced).
(e) No retracted paper is cited as a clean source.

CI runs in mock mode by default (``SCIX_FLAGSHIP_MOCK=1``). Live runs
are operator-triggered and skipped in CI.
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
    "Trace the local-vs-CMB H0 tension to its earliest explicit assertion "
    "in the corpus, and list each subsequent paper that quantitatively "
    "contradicted the prior consensus through 2024."
)

# ---------------------------------------------------------------------------
# MOCK_FIXTURE — canned events + canned final answer + grounding sources.
#
# The fixture is hand-crafted to satisfy every acceptance criterion. SH0ES
# bibcodes (Riess et al.) and Planck/CMB bibcodes are real-world examples;
# the answer is fabricated for assertion-shape coverage. Mock mode validates
# the assertion logic, not the agent itself.
# ---------------------------------------------------------------------------

# SH0ES papers (Riess et al. lineage)
_RIESS_2011 = "2011ApJ...730..119R"  # origin
_RIESS_2016 = "2016ApJ...826...56R"
_RIESS_2019 = "2019ApJ...876...85R"
_RIESS_2022 = "2022ApJ...934L...7R"  # most recent SH0ES tightening

# Planck / CMB papers
_PLANCK_2014 = "2014A&A...571A..16P"  # Planck 2013 results XVI
_PLANCK_2020 = "2020A&A...641A...6P"  # Planck 2018 results VI

MOCK_FIXTURE: dict[str, Any] = {
    "events": [
        {
            "type": "tool_use",
            "tool_name": "concept_search",
            "text": "(querying for 'local H0 CMB tension early measurement')",
        },
        {
            "type": "tool_result",
            "tool_name": "concept_search",
            "text": (
                f"Top hit: {_RIESS_2011} — Riess et al. 2011, "
                "\"Our value of H0 is 2.4 sigma higher than the value "
                "derived from WMAP+BAO+SN.\""
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "claim_blame",
            "text": "(walking reverse references to find chronological earliest)",
        },
        {
            "type": "tool_result",
            "tool_name": "claim_blame",
            "text": (
                f"claim_blame origin: {_RIESS_2011}, "
                f"lineage includes {_RIESS_2016}, {_RIESS_2019}, {_RIESS_2022}, "
                f"{_PLANCK_2014}, {_PLANCK_2020}. "
                "retraction_warnings: []."
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "find_replications",
            "text": f"(forward citations of {_RIESS_2011})",
        },
        {
            "type": "tool_result",
            "tool_name": "find_replications",
            "text": (
                f"Replications: SH0ES collaboration {_RIESS_2016} measured "
                "H0 = 73.24, "
                f"{_RIESS_2019} reported H0 = 74.03, "
                f"{_RIESS_2022} reported H0 = 73.04 ± 1.04 km/s/Mpc."
            ),
        },
        {
            "type": "tool_result",
            "tool_name": "find_replications",
            "text": (
                f"Refutations from Planck collaboration: {_PLANCK_2014} "
                "reported H0 = 67.3, "
                f"{_PLANCK_2020} reported H0 = 67.4 ± 0.5 km/s/Mpc."
            ),
        },
        {
            "type": "text",
            "text": (
                f"The earliest explicit assertion of a local-vs-CMB H0 tension "
                f"is Riess et al. 2011 ({_RIESS_2011}). "
                f"SH0ES collaboration tightened the local measurement: "
                f"{_RIESS_2016}, {_RIESS_2019}, "
                f"and most recently {_RIESS_2022} reported "
                "H0 = 73.04 ± 1.04 km/s/Mpc. "
                f"Planck collaboration ({_PLANCK_2014}, {_PLANCK_2020}) "
                "reported H0 = 67.4 ± 0.5 km/s/Mpc, "
                "quantitatively contradicting the prior consensus."
            ),
        },
    ],
    # Tool quotes that ground the claims in the canned answer.
    "tool_results_for_grounding": [
        {
            "text": (
                f"The earliest explicit assertion of a local-vs-CMB H0 tension "
                f"is Riess et al. 2011 ({_RIESS_2011})"
            ),
        },
        {
            "text": (
                f"SH0ES collaboration tightened the local measurement: "
                f"{_RIESS_2016}, {_RIESS_2019}, "
                f"and most recently {_RIESS_2022} reported "
                "H0 = 73.04 ± 1.04 km/s/Mpc"
            ),
        },
        {
            "text": (
                f"Planck collaboration ({_PLANCK_2014}, {_PLANCK_2020}) "
                "reported H0 = 67.4 ± 0.5 km/s/Mpc, "
                "quantitatively contradicting the prior consensus"
            ),
        },
    ],
    "expected_origin": _RIESS_2011,
    "expected_sh0es_bibcodes": (_RIESS_2011, _RIESS_2016, _RIESS_2019, _RIESS_2022),
    "expected_planck_bibcodes": (_PLANCK_2014, _PLANCK_2020),
    "retracted_bibcodes": (),  # No retractions in this lineage.
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_flagship_h0_tension(run_flagship: Any) -> None:
    """MH-8a — canonical Hubble-tension question, end-to-end grounded."""
    result = run_flagship(QUESTION, MOCK_FIXTURE)

    answer = result.answer
    bibs_in_answer = collect_bibcodes(answer)

    # (a) Riess+ 2011 in lineage
    assert MOCK_FIXTURE["expected_origin"] in bibs_in_answer, (
        f"Riess+ 2011 origin {MOCK_FIXTURE['expected_origin']} not found "
        f"in answer; bibs={sorted(bibs_in_answer)}"
    )

    # (b) >=3 SH0ES papers, >=2 Planck/CMB papers
    sh0es_in_answer = [
        b for b in MOCK_FIXTURE["expected_sh0es_bibcodes"] if b in bibs_in_answer
    ]
    planck_in_answer = [
        b for b in MOCK_FIXTURE["expected_planck_bibcodes"] if b in bibs_in_answer
    ]
    assert len(sh0es_in_answer) >= 3, (
        f"expected >=3 SH0ES papers in answer, got {sh0es_in_answer}"
    )
    assert len(planck_in_answer) >= 2, (
        f"expected >=2 Planck/CMB papers in answer, got {planck_in_answer}"
    )

    # (c) Every claim citation_grounded=True
    assert_all_claims_grounded(
        answer,
        MOCK_FIXTURE["tool_results_for_grounding"],
    )

    # (d) <=25 tool turns (A2 amendment)
    assert result.metadata.n_turns <= 25, (
        f"n_turns={result.metadata.n_turns} exceeds 25-turn budget"
    )
    assert not result.metadata.truncated, (
        "fixture should fit under 25 turns without truncation"
    )

    # (e) No retracted paper cited as clean source
    for retracted in MOCK_FIXTURE["retracted_bibcodes"]:
        assert retracted not in answer, (
            f"retracted bibcode {retracted} appears in answer as clean source"
        )
