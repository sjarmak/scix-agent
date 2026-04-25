"""MH-8d flagship test — paper novelty / random-bibcode summary.

Per PRD amendment A9 (``docs/prd/scix_deep_search_v1.md``):

    Replace MH-8's single flagship with two (contested-trace +
    paper-novelty). Add MH-8d: "Summarize this paper [random ADS
    bibcode], identify what's novel about it relative to the prior
    literature, and surface any contested elements." This is closer
    to what working researchers ask. Both must pass.

Acceptance criteria:

(1) Agent summarizes the paper, identifies novelty, surfaces contested
    elements.
(2) Answer cites the seed paper.
(3) Answer cites ≥3 prior-art bibcodes (distinct from seed) — surfaced
    via tool calls.
(4) ≤25 tool turns.

CI runs in mock mode by default.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.integration.conftest import (
    collect_bibcodes,
    collect_bibcodes_from_events,
)

# ---------------------------------------------------------------------------
# Canonical question — fixed seed bibcode for reproducibility.
# ---------------------------------------------------------------------------

# Same seed bibcode as the persona example; mock mode is allowed.
SEED_BIBCODE: str = "2024MNRAS.527.1234X"

QUESTION: str = (
    f"Summarize {SEED_BIBCODE}, identify what is novel about it relative "
    "to the prior literature, and surface any contested elements."
)

# ---------------------------------------------------------------------------
# MOCK_FIXTURE
# ---------------------------------------------------------------------------

# Three fictitious prior-art bibcodes — must be distinct, must match the
# canonical 19-char ADS pattern.
_PRIOR_1 = "2021MNRAS.500.2345Y"
_PRIOR_2 = "2022ApJ...934..456Z"
_PRIOR_3 = "2023A&A...675A..78W"

MOCK_FIXTURE: dict[str, Any] = {
    "events": [
        {
            "type": "tool_use",
            "tool_name": "get_paper",
            "text": f"(fetching metadata for {SEED_BIBCODE})",
        },
        {
            "type": "tool_result",
            "tool_name": "get_paper",
            "text": (
                f"{SEED_BIBCODE} — title: \"A new ML method for galaxy "
                "morphological classification.\" Abstract: \"We introduce "
                "a transformer-based architecture that improves over prior "
                "CNN approaches.\""
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "read_paper",
            "text": f"(reading body of {SEED_BIBCODE})",
        },
        {
            "type": "tool_result",
            "tool_name": "read_paper",
            "text": (
                f"{SEED_BIBCODE} body §1: \"Our novelty is the use of "
                "self-attention over patches of the spectroscopic feature "
                "map, which prior CNN-based work cannot exploit.\""
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "citation_chain",
            "text": f"(direction=references for {SEED_BIBCODE})",
        },
        {
            "type": "tool_result",
            "tool_name": "citation_chain",
            "text": (
                f"References include: {_PRIOR_1} (CNN-based morphology), "
                f"{_PRIOR_2} (RNN with hand-crafted features), "
                f"{_PRIOR_3} (a transformer applied to photometric — but "
                "not spectroscopic — data)."
            ),
        },
        {
            "type": "tool_use",
            "tool_name": "concept_search",
            "text": "(searching for prior transformer-on-spectroscopy papers)",
        },
        {
            "type": "tool_result",
            "tool_name": "concept_search",
            "text": (
                f"Closest prior art: {_PRIOR_3} applied transformers to "
                "photometric data only; the seed paper extends to "
                "spectroscopic features."
            ),
        },
        {
            "type": "text",
            "text": (
                f"{SEED_BIBCODE} introduces a transformer-based architecture "
                "for galaxy morphological classification. "
                f"The stated novelty is self-attention over spectroscopic "
                f"feature maps, which prior CNN-based work ({_PRIOR_1}) and "
                f"RNN-based work ({_PRIOR_2}) cannot exploit. "
                f"The closest prior art is {_PRIOR_3}, which applied "
                "transformers to photometric data only. "
                "Contested element: the abstract claims architectural "
                "novelty, but the spectroscopic-vs-photometric distinction "
                "is the load-bearing contribution and is not benchmarked "
                "against an ablated photometric-only baseline."
            ),
        },
    ],
    "tool_results_for_grounding": [
        {
            "text": (
                f"{SEED_BIBCODE} introduces a transformer-based architecture "
                "for galaxy morphological classification"
            ),
        },
        {
            "text": (
                f"The stated novelty is self-attention over spectroscopic "
                f"feature maps, which prior CNN-based work ({_PRIOR_1}) and "
                f"RNN-based work ({_PRIOR_2}) cannot exploit"
            ),
        },
        {
            "text": (
                f"The closest prior art is {_PRIOR_3}, which applied "
                "transformers to photometric data only"
            ),
        },
        {
            "text": (
                "Contested element: the abstract claims architectural "
                "novelty, but the spectroscopic-vs-photometric distinction "
                "is the load-bearing contribution and is not benchmarked "
                "against an ablated photometric-only baseline"
            ),
        },
    ],
    "expected_seed_bibcode": SEED_BIBCODE,
    "expected_prior_art_min": 3,
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_flagship_paper_novelty(run_flagship: Any) -> None:
    """MH-8d — paper-novelty random-bibcode summary."""
    result = run_flagship(QUESTION, MOCK_FIXTURE)
    answer = result.answer

    # (2) Seed paper cited
    assert MOCK_FIXTURE["expected_seed_bibcode"] in answer, (
        f"seed bibcode {MOCK_FIXTURE['expected_seed_bibcode']} "
        "not present in answer"
    )

    # (3) ≥3 prior-art bibcodes (distinct from seed)
    bibs_in_tool_calls = collect_bibcodes_from_events(result.tool_calls)
    bibs_in_answer = collect_bibcodes(answer)
    seed = MOCK_FIXTURE["expected_seed_bibcode"]
    prior_art_in_calls = bibs_in_tool_calls - {seed}
    prior_art_in_answer = bibs_in_answer - {seed}

    assert len(prior_art_in_calls) >= MOCK_FIXTURE["expected_prior_art_min"], (
        f"expected >={MOCK_FIXTURE['expected_prior_art_min']} prior-art "
        f"bibcodes in tool calls, got {sorted(prior_art_in_calls)}"
    )
    assert len(prior_art_in_answer) >= MOCK_FIXTURE["expected_prior_art_min"], (
        f"expected >={MOCK_FIXTURE['expected_prior_art_min']} prior-art "
        f"bibcodes in answer, got {sorted(prior_art_in_answer)}"
    )

    # (1) Novelty + contested-elements language present
    answer_lower = answer.lower()
    assert any(t in answer_lower for t in ("novel", "novelty")), (
        "answer must surface the paper's novelty"
    )
    assert any(t in answer_lower for t in ("contested", "ablated", "limitation")), (
        "answer must surface a contested or limiting element"
    )

    # (4) ≤25 tool turns
    assert result.metadata.n_turns <= 25, (
        f"n_turns={result.metadata.n_turns} exceeds 25-turn budget"
    )
    assert not result.metadata.truncated
