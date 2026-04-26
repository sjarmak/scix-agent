"""Schema and span integrity tests for the claim-extraction gold standard.

The gold standard lives at ``eval/claim_extraction_gold_standard.jsonl``.
These tests are the canonical contract: they must pass for any change that
touches the gold-standard file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

GOLD_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "claim_extraction_gold_standard.jsonl"
)

REQUIRED_ENTRY_KEYS = {
    "bibcode",
    "section_index",
    "paragraph_index",
    "paragraph_text",
    "expected_claims",
    "discipline",
}

REQUIRED_CLAIM_KEYS = {
    "claim_text",
    "claim_type",
    "subject",
    "predicate",
    "object",
    "char_span_start",
    "char_span_end",
}

ALLOWED_CLAIM_TYPES = {
    "factual",
    "methodological",
    "comparative",
    "speculative",
    "cited_from_other",
}

ALLOWED_DISCIPLINES = {
    "astrophysics",
    "planetary_science",
    "earth_science",
}

MIN_TOTAL_ENTRIES = 12
MIN_PER_DISCIPLINE = 4


def _load_entries() -> list[dict]:
    assert GOLD_PATH.exists(), f"gold standard JSONL missing: {GOLD_PATH}"
    entries: list[dict] = []
    with GOLD_PATH.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                pytest.fail(f"line {lineno} is not valid JSON: {exc}")
    return entries


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return _load_entries()


def test_file_exists() -> None:
    assert GOLD_PATH.is_file(), f"gold standard file not found at {GOLD_PATH}"


def test_minimum_entry_count(entries: list[dict]) -> None:
    assert len(entries) >= MIN_TOTAL_ENTRIES, (
        f"gold standard has {len(entries)} entries; "
        f"need at least {MIN_TOTAL_ENTRIES}"
    )


def test_per_discipline_minimum(entries: list[dict]) -> None:
    counts = Counter(entry["discipline"] for entry in entries)
    for discipline in ALLOWED_DISCIPLINES:
        assert counts.get(discipline, 0) >= MIN_PER_DISCIPLINE, (
            f"discipline {discipline!r} has {counts.get(discipline, 0)} entries; "
            f"need at least {MIN_PER_DISCIPLINE}"
        )


def test_required_entry_keys(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
        assert not missing, (
            f"entry {idx} (bibcode={entry.get('bibcode')!r}) "
            f"missing keys: {sorted(missing)}"
        )


def test_discipline_enum(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        assert entry["discipline"] in ALLOWED_DISCIPLINES, (
            f"entry {idx}: discipline {entry['discipline']!r} "
            f"not in {sorted(ALLOWED_DISCIPLINES)}"
        )


def test_section_paragraph_indices_are_ints(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        assert isinstance(entry["section_index"], int), (
            f"entry {idx}: section_index must be int"
        )
        assert isinstance(entry["paragraph_index"], int), (
            f"entry {idx}: paragraph_index must be int"
        )


def test_paragraph_text_is_nonempty_string(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        text = entry["paragraph_text"]
        assert isinstance(text, str) and text.strip(), (
            f"entry {idx}: paragraph_text must be a non-empty string"
        )


def test_expected_claims_is_nonempty_list(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        claims = entry["expected_claims"]
        assert isinstance(claims, list) and len(claims) > 0, (
            f"entry {idx}: expected_claims must be a non-empty list"
        )


def test_claim_required_keys(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        for cidx, claim in enumerate(entry["expected_claims"]):
            missing = REQUIRED_CLAIM_KEYS - set(claim.keys())
            assert not missing, (
                f"entry {idx} claim {cidx}: missing keys {sorted(missing)}"
            )


def test_claim_type_enum(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        for cidx, claim in enumerate(entry["expected_claims"]):
            assert claim["claim_type"] in ALLOWED_CLAIM_TYPES, (
                f"entry {idx} claim {cidx}: claim_type "
                f"{claim['claim_type']!r} not in {sorted(ALLOWED_CLAIM_TYPES)}"
            )


def test_span_is_nonempty_substring_of_paragraph(entries: list[dict]) -> None:
    for idx, entry in enumerate(entries):
        paragraph = entry["paragraph_text"]
        for cidx, claim in enumerate(entry["expected_claims"]):
            start = claim["char_span_start"]
            end = claim["char_span_end"]
            assert isinstance(start, int) and isinstance(end, int), (
                f"entry {idx} claim {cidx}: span offsets must be ints"
            )
            assert 0 <= start < end <= len(paragraph), (
                f"entry {idx} claim {cidx}: span [{start}, {end}) "
                f"out of bounds for paragraph of length {len(paragraph)}"
            )
            substring = paragraph[start:end]
            assert substring, (
                f"entry {idx} claim {cidx}: span yields empty substring"
            )
            # Sanity: substring is actually a substring of the paragraph
            assert substring in paragraph, (
                f"entry {idx} claim {cidx}: span content not found in paragraph"
            )


def test_claim_type_diversity(entries: list[dict]) -> None:
    """Soft check that the corpus represents all 5 claim types."""
    seen: set[str] = set()
    for entry in entries:
        for claim in entry["expected_claims"]:
            seen.add(claim["claim_type"])
    missing = ALLOWED_CLAIM_TYPES - seen
    assert not missing, (
        f"gold standard does not exercise claim_type values: {sorted(missing)}"
    )
