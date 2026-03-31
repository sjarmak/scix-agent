"""Unit tests for JSONL → SQL field mapping and record transformation."""

from __future__ import annotations

import json

import pytest

from scix.field_mapping import COLUMN_ORDER, transform_record

# Column name -> index lookup for readable assertions.
COL = {name: i for i, name in enumerate(COLUMN_ORDER)}


# --- Fixtures ---

FULL_RECORD: dict = {
    "bibcode": "2024ApJ...962L..15J",
    "title": ["Predicting the Timing of the Solar Cycle 25 Polar Field Reversal"],
    "abstract": "We present a novel analysis of gravitational wave signals.",
    "year": "2024",
    "doctype": "article",
    "pub": "The Astrophysical Journal",
    "pub_raw": "The Astrophysical Journal Letters, Volume 962, Issue 1",
    "volume": "962",
    "issue": "1",
    "page": ["L15"],
    "author": ["Jha, Bibhuti Kumar", "Upton, Lisa A."],
    "first_author": "Jha, Bibhuti Kumar",
    "aff": ["Southwest Research Institute", "Southwest Research Institute"],
    "keyword": ["Sunspots", "Solar cycle"],
    "arxiv_class": ["astro-ph.SR"],
    "database": ["astronomy"],
    "doi": ["10.3847/2041-8213/ad20d2"],
    "identifier": ["arXiv:2401.10502", "2024ApJ...962L..15J"],
    "alternate_bibcode": ["2024arXiv240110502J"],
    "bibstem": ["ApJL", "ApJL..962"],
    "bibgroup": [],
    "orcid_pub": ["0000-0003-3191-4625", "0000-0003-0621-4803"],
    "orcid_user": ["-", "0000-0003-0621-4803"],
    "property": ["ARTICLE", "REFEREED"],
    "copyright": "(c) 2024 AAS",
    "lang": "en",
    "pubdate": "2024-02-00",
    "entry_date": "2024-02-11T00:00:00Z",
    "indexstamp": "2025-04-29T11:02:13.932Z",
    "citation_count": 8,
    "read_count": 79,
    "reference_count": 35,
    "reference": ["1919ApJ....49..153H", "1959ApJ...130..364B", "2024AdSpR..74.1518J"],
    "citation": ["2024AdSpR..74.1518J"],
    "body": "Full text here...",
    "ack": "We thank the reviewer.",
    "id": "26840679",
    "grant": [{"agency": "NASA", "id": "123"}],
}


MINIMAL_RECORD: dict = {
    "bibcode": "2026KIsMS.336..114S",
    "title": ["Computer simulation of copper monosulfide"],
    "year": "2026",
    "doctype": "article",
    "author": ["Shevko, V."],
    "first_author": "Shevko, V.",
}


class TestTransformRecord:
    def test_bibcode_preserved(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["bibcode"]] == "2024ApJ...962L..15J"

    def test_title_extracted_from_list(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["title"]] == (
            "Predicting the Timing of the Solar Cycle 25 Polar Field Reversal"
        )

    def test_title_string_passthrough(self) -> None:
        rec = {**MINIMAL_RECORD, "title": "Plain string title"}
        row, _ = transform_record(rec)
        assert row[COL["title"]] == "Plain string title"

    def test_title_empty_list(self) -> None:
        rec = {**MINIMAL_RECORD, "title": []}
        row, _ = transform_record(rec)
        assert row[COL["title"]] is None

    def test_year_converted_to_int(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        year = row[COL["year"]]
        assert year == 2024
        assert isinstance(year, int)

    def test_year_invalid_string(self) -> None:
        rec = {**MINIMAL_RECORD, "year": "unknown"}
        row, _ = transform_record(rec)
        assert row[COL["year"]] is None

    def test_year_missing(self) -> None:
        rec = {k: v for k, v in MINIMAL_RECORD.items() if k != "year"}
        row, _ = transform_record(rec)
        assert row[COL["year"]] is None

    def test_renamed_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        authors = row[COL["authors"]]
        affiliations = row[COL["affiliations"]]
        keywords = row[COL["keywords"]]
        assert authors == ["Jha, Bibhuti Kumar", "Upton, Lisa A."]
        assert affiliations == ["Southwest Research Institute", "Southwest Research Institute"]
        assert keywords == ["Sunspots", "Solar cycle"]

    def test_direct_array_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["arxiv_class"]] == ["astro-ph.SR"]
        assert row[COL["doi"]] == ["10.3847/2041-8213/ad20d2"]
        assert row[COL["page"]] == ["L15"]
        assert row[COL["property"]] == ["ARTICLE", "REFEREED"]

    def test_integer_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["citation_count"]] == 8
        assert row[COL["read_count"]] == 79
        assert row[COL["reference_count"]] == 35

    def test_unmapped_fields_in_raw_jsonb(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        raw_str = row[COL["raw"]]
        assert raw_str is not None
        raw = json.loads(raw_str)
        assert raw["body"] == "Full text here..."
        assert raw["ack"] == "We thank the reviewer."
        assert raw["id"] == "26840679"
        assert "reference" in raw  # preserved in raw for provenance
        assert "citation" in raw

    def test_unmapped_fields_exclude_mapped(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        raw = json.loads(row[COL["raw"]])
        # Mapped fields should NOT appear in raw
        assert "bibcode" not in raw
        assert "title" not in raw
        assert "author" not in raw
        assert "year" not in raw
        assert "abstract" not in raw

    def test_raw_none_when_no_unmapped(self) -> None:
        row, _ = transform_record(MINIMAL_RECORD)
        assert row[COL["raw"]] is None

    def test_missing_optional_fields_are_none(self) -> None:
        row, _ = transform_record(MINIMAL_RECORD)
        assert row[COL["abstract"]] is None
        assert row[COL["keywords"]] is None
        assert row[COL["citation_count"]] is None
        assert row[COL["copyright"]] is None

    def test_missing_bibcode_raises(self) -> None:
        with pytest.raises(ValueError, match="missing bibcode"):
            transform_record({"title": ["No bibcode"], "year": "2024"})

    def test_empty_bibcode_raises(self) -> None:
        with pytest.raises(ValueError, match="missing bibcode"):
            transform_record({"bibcode": "", "title": ["Empty"], "year": "2024"})


class TestEdgeExtraction:
    def test_edges_from_references(self) -> None:
        _, edges = transform_record(FULL_RECORD)
        assert len(edges) == 3
        assert ("2024ApJ...962L..15J", "1919ApJ....49..153H") in edges
        assert ("2024ApJ...962L..15J", "1959ApJ...130..364B") in edges
        assert ("2024ApJ...962L..15J", "2024AdSpR..74.1518J") in edges

    def test_no_edges_when_no_references(self) -> None:
        _, edges = transform_record(MINIMAL_RECORD)
        assert edges == []

    def test_edges_skip_empty_strings(self) -> None:
        rec = {**MINIMAL_RECORD, "reference": ["2024ApJ...001", "", "2024ApJ...002"]}
        _, edges = transform_record(rec)
        assert len(edges) == 2

    def test_edges_skip_non_string_entries(self) -> None:
        rec = {**MINIMAL_RECORD, "reference": ["2024ApJ...001", None, 12345]}
        _, edges = transform_record(rec)
        assert len(edges) == 1

    def test_citation_field_not_used_for_edges(self) -> None:
        """Only reference[] is used for edges; citation[] goes to raw JSONB."""
        rec = {**MINIMAL_RECORD, "citation": ["2025xyz...001"], "reference": []}
        _, edges = transform_record(rec)
        assert edges == []


class TestColumnOrder:
    def test_column_count_matches_schema(self) -> None:
        # papers table has 33 columns (32 named + raw)
        assert len(COLUMN_ORDER) == 33

    def test_bibcode_is_first(self) -> None:
        assert COLUMN_ORDER[0] == "bibcode"

    def test_raw_is_last(self) -> None:
        assert COLUMN_ORDER[-1] == "raw"

    def test_no_duplicates(self) -> None:
        assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER))

    def test_row_length_matches_column_count(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert len(row) == len(COLUMN_ORDER)
