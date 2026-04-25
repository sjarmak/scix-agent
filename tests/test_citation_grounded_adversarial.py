"""Adversarial unit test for citation_grounded gate (PRD MH-6).

Twenty hand-curated paraphrase / unsupported-claim cases. Each case is a
dict with:
    - assertion: the candidate sentence in the persona's draft
    - tool_results: the tool-result quotes available to the gate
    - expected_grounded: True for legitimate paraphrases, False for unsupported

Acceptance (per acceptance criterion #7): of the 10 unsupported cases, at most 2
may incorrectly slip through as grounded=True (i.e., ≥80% of unsupported claims
must be flagged as ungrounded). For coverage, we also assert that ≥7 of the 10
legitimate paraphrases are correctly grounded.

Embedder: count-vector over the call's vocab (same as test_citation_grounded.py)
so the test is deterministic.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest

from scix.citation_grounded import (
    DEFAULT_THRESHOLD,
    grounded_check,
    set_embedder,
)


# ---------------------------------------------------------------------------
# Deterministic embedder (shared with test_citation_grounded.py logic)
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
        vectors.append([float(counts.get(v, 0)) for v in vocab])
    return vectors


@pytest.fixture(autouse=True)
def _install_embedder():
    set_embedder(count_vector_embedder)
    yield
    set_embedder(None)


# ---------------------------------------------------------------------------
# 20 adversarial cases
# ---------------------------------------------------------------------------
#
# Cases 1-10: legitimate paraphrases (expected_grounded=True). These share
# substantial content vocabulary with the corresponding tool quote and/or
# overlap as substrings.
#
# Cases 11-20: unsupported claims (expected_grounded=False). These contain
# distinct content vocabulary from the tool quotes — minimal overlap.
#
# All cases are at the default threshold of 0.82.

CASES: list[dict[str, Any]] = [
    # --- Legitimate paraphrases (1-10) ---
    {
        "id": "L01_substring",
        "assertion": "The team measured H0 = 73.0 km/s/Mpc using Cepheid distances.",
        "tool_results": [
            {
                "quote": (
                    "The team measured H0 = 73.0 km/s/Mpc using Cepheid distances "
                    "and SH0ES distance ladder."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L02_paraphrase_dark_matter",
        "assertion": "Galactic rotation curves require dark matter halos.",
        "tool_results": [
            {
                "text": (
                    "Galactic rotation curves require dark matter halos to explain "
                    "the observed flat profiles at large radii."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L03_substring_planck",
        "assertion": "Planck CMB analysis yields H0 = 67.4 km/s/Mpc.",
        "tool_results": [
            {
                "snippet": (
                    "Planck CMB analysis yields H0 = 67.4 km/s/Mpc from a "
                    "six-parameter LCDM fit to TT/TE/EE spectra."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L04_paraphrase_supernova",
        "assertion": "Type Ia supernova observations indicate accelerating cosmic expansion.",
        "tool_results": [
            {
                "text": (
                    "Type Ia supernova observations indicate accelerating cosmic "
                    "expansion driven by dark energy."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L05_substring_neutrino",
        "assertion": "Neutrino oscillation experiments measure mass-squared differences.",
        "tool_results": [
            {
                "quote": (
                    "Neutrino oscillation experiments measure mass-squared differences "
                    "in solar and atmospheric channels."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L06_paraphrase_jwst",
        "assertion": "JWST observations reveal high-redshift galaxies brighter than predicted.",
        "tool_results": [
            {
                "snippet": (
                    "JWST observations reveal high-redshift galaxies brighter than "
                    "predicted by LCDM galaxy formation models."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L07_substring_bicep",
        "assertion": "BICEP2 reported a tensor-to-scalar ratio of r = 0.20.",
        "tool_results": [
            {
                "text": (
                    "BICEP2 reported a tensor-to-scalar ratio of r = 0.20 from B-mode "
                    "polarization measurements at degree scales."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L08_paraphrase_riess",
        "assertion": "Riess and collaborators establish a local distance ladder using Cepheids and supernovae.",
        "tool_results": [
            {
                "quote": (
                    "Riess and collaborators establish a local distance ladder using "
                    "Cepheids and supernovae anchored to NGC 4258 megamasers."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L09_substring_exoplanet",
        "assertion": "Kepler detected over 4000 exoplanet candidates by transit photometry.",
        "tool_results": [
            {
                "snippet": (
                    "Kepler detected over 4000 exoplanet candidates by transit "
                    "photometry over its 4-year primary mission."
                )
            }
        ],
        "expected_grounded": True,
    },
    {
        "id": "L10_paraphrase_gw",
        "assertion": "LIGO measured gravitational waves from a binary black hole merger in 2015.",
        "tool_results": [
            {
                "text": (
                    "LIGO measured gravitational waves from a binary black hole "
                    "merger in 2015 in the GW150914 event detection."
                )
            }
        ],
        "expected_grounded": True,
    },
    # --- Unsupported claims (11-20) ---
    {
        "id": "U11_unrelated",
        "assertion": "Wombats invented general relativity in 1915.",
        "tool_results": [
            {"text": "Cepheid variables are pulsating standard candles."},
            {"text": "Type Ia supernovae trace dark energy."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U12_fabricated_value",
        "assertion": "Pluto orbits inside Mercury at 0.3 AU.",
        "tool_results": [
            {"snippet": "Pluto is a Kuiper-belt object beyond Neptune."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U13_unsupported_extrapolation",
        "assertion": "All quasars host supermassive black holes of exactly 10^9 solar masses.",
        "tool_results": [
            {
                "text": (
                    "Quasars are powered by accretion onto active galactic nuclei "
                    "with a wide range of central black hole masses."
                )
            }
        ],
        "expected_grounded": False,
    },
    {
        "id": "U14_invented_paper",
        "assertion": "Smith 2099 demonstrates faster-than-light neutrinos at CERN.",
        "tool_results": [
            {"snippet": "OPERA reported anomalous neutrino velocity measurements that were later attributed to a loose cable."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U15_pure_speculation",
        "assertion": "Aliens calibrated the cosmic microwave background dipole.",
        "tool_results": [
            {"text": "The CMB dipole arises from solar motion relative to the rest frame defined by photons."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U16_wrong_facts",
        "assertion": "Andromeda galaxy will collide with the Milky Way in 2026.",
        "tool_results": [
            {"snippet": "Andromeda is approaching the Milky Way at 110 km/s and they will merge in roughly 4.5 billion years."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U17_off_topic",
        "assertion": "The price of bitcoin reached 100000 dollars yesterday.",
        "tool_results": [
            {"text": "Cosmological parameter constraints come from CMB, BAO, and SNIa data."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U18_fabricated_quote",
        "assertion": "Einstein famously said cosmology is engineering.",
        "tool_results": [
            {"snippet": "Albert Einstein developed general relativity in 1915 and described gravitation as spacetime curvature."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U19_unsupported_method",
        "assertion": "We trained a 175-billion-parameter LLM on rotation curves overnight.",
        "tool_results": [
            {"text": "Rotation curve fitting is performed via NFW profile chi-squared minimization."},
        ],
        "expected_grounded": False,
    },
    {
        "id": "U20_invented_instrument",
        "assertion": "The Smith Array detected dark matter particles directly in 2025.",
        "tool_results": [
            {
                "snippet": (
                    "Direct dark matter detection experiments such as XENONnT and "
                    "LZ have set increasingly stringent limits on WIMP cross sections."
                )
            }
        ],
        "expected_grounded": False,
    },
]


# Sanity at module-import time: exactly 20 cases, 10 of each polarity.
assert len(CASES) == 20, f"adversarial set must have 20 cases, has {len(CASES)}"
assert sum(1 for c in CASES if c["expected_grounded"]) == 10
assert sum(1 for c in CASES if not c["expected_grounded"]) == 10


# ---------------------------------------------------------------------------
# Per-case test (parametrized for readability) — informational only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_records_outcome(case: dict[str, Any]) -> None:
    """Each case should produce a GroundingReport without errors.

    This test does not assert per-case correctness — the budgeted
    aggregate assertions live in :func:`test_adversarial_aggregate_budget`
    so a single edge case doesn't fail the whole suite.
    """
    report = grounded_check(
        case["assertion"],
        case["tool_results"],
        threshold=DEFAULT_THRESHOLD,
    )
    # A claim-bearing assertion is always detected (these cases all have
    # numeric or named-entity content).
    assert len(report.assertions) >= 1, f"{case['id']}: no assertion detected"


# ---------------------------------------------------------------------------
# Aggregate budget assertions (the actual MH-6 acceptance criterion)
# ---------------------------------------------------------------------------


def test_adversarial_aggregate_budget() -> None:
    """≤2 false-clean outputs on the 10 unsupported claims (PRD MH-6)."""
    false_clean = 0  # unsupported claim slipped through as grounded=True
    correct_clean = 0  # legit paraphrase correctly grounded
    legit_misflagged = 0  # legit paraphrase flagged as ungrounded

    detail: list[str] = []
    for case in CASES:
        report = grounded_check(
            case["assertion"],
            case["tool_results"],
            threshold=DEFAULT_THRESHOLD,
        )
        actual = report.grounded
        expected = case["expected_grounded"]
        if expected and actual:
            correct_clean += 1
        elif expected and not actual:
            legit_misflagged += 1
        elif (not expected) and actual:
            false_clean += 1
            detail.append(f"FALSE-CLEAN {case['id']}: assertion={case['assertion']!r}")
        # not-expected, not-actual → correctly flagged

    # PRD MH-6 acceptance: ≤2 false-clean outputs.
    assert false_clean <= 2, (
        f"{false_clean} unsupported claims slipped through as grounded; "
        f"budget is 2.\n" + "\n".join(detail)
    )

    # Coverage: ≥7 of the 10 legit paraphrases must be grounded — otherwise
    # the gate is too strict to ship.
    assert correct_clean >= 7, (
        f"only {correct_clean}/10 legit paraphrases grounded; "
        f"the gate is too strict at threshold={DEFAULT_THRESHOLD}."
    )
