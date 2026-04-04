"""Unit tests for JSONL → SQL field mapping and record transformation."""

from __future__ import annotations

import json

import pytest

from scix.field_mapping import (
    COLUMN_ORDER,
    DIRECT_ARRAY_FIELDS,
    DIRECT_FLOAT_FIELDS,
    DIRECT_INT_FIELDS,
    DIRECT_TEXT_FIELDS,
    RENAMES,
    transform_record,
)

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
    # --- New full-coverage fields (migration 012) ---
    "ack": "We thank the reviewer.",
    "date": "2024-02-00",
    "eid": "L15",
    "entdate": "2024-02-11",
    "first_author_norm": "Jha, B",
    "page_range": "L15",
    "pubnote": "5 pages, 3 figures",
    "series": "ApJL",
    "aff_id": ["60001234", "60001234"],
    "alternate_title": ["Predicting Solar Cycle 25 Reversal"],
    "author_norm": ["Jha, B", "Upton, L"],
    "caption": ["Figure 1: Solar cycle phases"],
    "comment": ["Accepted for publication"],
    "data": ["SIMBAD:4", "ESO:13"],
    "esources": ["PUB_PDF", "PUB_HTML"],
    "facility": ["SDO", "GONG"],
    "grant": ["NASA NNX12AB34G", "NSF AST-1234567"],
    "grant_agencies": ["NASA", "NSF"],
    "grant_id": ["NNX12AB34G", "AST-1234567"],
    "isbn": [],
    "issn": ["2041-8213"],
    "keyword_norm": ["sunspot", "solar cycle"],
    "keyword_schema": ["Unified Astronomy Thesaurus"],
    "links_data": ['{"title":"", "type":"electr", "instances":""}'],
    "nedid": ["MESSIER_031"],
    "nedtype": ["G"],
    "orcid_other": ["0000-0003-3191-4625"],
    "simbid": ["1575544"],
    "vizier": ["J/ApJ/962/L15"],
    "author_count": 2,
    "page_count": 5,
    "citation_count_norm": 0.53,
    "cite_read_boost": 0.42,
    "classic_factor": 0.0,
    # --- Fields that stay in raw JSONB ---
    "id": "26840679",
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

    def test_body_as_dedicated_column(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["body"]] == "Full text here..."

    def test_body_not_in_raw(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        raw = json.loads(row[COL["raw"]])
        assert "body" not in raw

    def test_body_none_when_missing(self) -> None:
        row, _ = transform_record(MINIMAL_RECORD)
        assert row[COL["body"]] is None

    def test_body_null_bytes_stripped(self) -> None:
        rec = {**MINIMAL_RECORD, "body": "Full\x00text\x00here"}
        row, _ = transform_record(rec)
        assert row[COL["body"]] == "Fulltexthere"

    def test_unmapped_fields_in_raw_jsonb(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        raw_str = row[COL["raw"]]
        assert raw_str is not None
        raw = json.loads(raw_str)
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
        assert "body" not in raw
        # New mapped fields should also NOT appear in raw
        assert "ack" not in raw
        assert "data" not in raw
        assert "facility" not in raw
        assert "grant" not in raw
        assert "author_count" not in raw
        assert "citation_count_norm" not in raw

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

    def test_null_bytes_stripped_from_text(self) -> None:
        rec = {
            **MINIMAL_RECORD,
            "abstract": "Text with \x00null\x00 bytes",
            "title": ["Title\x00with\x00nulls"],
        }
        row, _ = transform_record(rec)
        assert row[COL["abstract"]] == "Text with null bytes"
        assert row[COL["title"]] == "Titlewithnulls"

    def test_null_bytes_stripped_from_arrays(self) -> None:
        rec = {**MINIMAL_RECORD, "author": ["Author\x00, A."]}
        row, _ = transform_record(rec)
        assert row[COL["authors"]] == ["Author, A."]

    def test_null_bytes_stripped_from_raw_jsonb(self) -> None:
        rec = {**MINIMAL_RECORD, "some_unmapped_field": "Thanks\x00reviewer"}
        row, _ = transform_record(rec)
        raw = json.loads(row[COL["raw"]])
        assert "\x00" not in raw["some_unmapped_field"]


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


class TestNewFullCoverageFields:
    """Tests for fields added in migration 012 (full ADS field coverage)."""

    def test_new_text_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["ack"]] == "We thank the reviewer."
        assert row[COL["date"]] == "2024-02-00"
        assert row[COL["eid"]] == "L15"
        assert row[COL["entdate"]] == "2024-02-11"
        assert row[COL["first_author_norm"]] == "Jha, B"
        assert row[COL["page_range"]] == "L15"
        assert row[COL["pubnote"]] == "5 pages, 3 figures"
        assert row[COL["series"]] == "ApJL"

    def test_new_array_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["data"]] == ["SIMBAD:4", "ESO:13"]
        assert row[COL["facility"]] == ["SDO", "GONG"]
        assert row[COL["esources"]] == ["PUB_PDF", "PUB_HTML"]
        assert row[COL["nedid"]] == ["MESSIER_031"]
        assert row[COL["simbid"]] == ["1575544"]
        assert row[COL["author_norm"]] == ["Jha, B", "Upton, L"]
        assert row[COL["keyword_norm"]] == ["sunspot", "solar cycle"]
        assert row[COL["vizier"]] == ["J/ApJ/962/L15"]

    def test_new_integer_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["author_count"]] == 2
        assert row[COL["page_count"]] == 5

    def test_new_float_fields(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["citation_count_norm"]] == pytest.approx(0.53)
        assert row[COL["cite_read_boost"]] == pytest.approx(0.42)
        assert row[COL["classic_factor"]] == pytest.approx(0.0)

    def test_float_field_invalid_value(self) -> None:
        rec = {**MINIMAL_RECORD, "citation_count_norm": "not_a_number"}
        row, _ = transform_record(rec)
        assert row[COL["citation_count_norm"]] is None

    def test_float_field_missing(self) -> None:
        row, _ = transform_record(MINIMAL_RECORD)
        assert row[COL["citation_count_norm"]] is None
        assert row[COL["cite_read_boost"]] is None
        assert row[COL["classic_factor"]] is None

    def test_grant_renamed_to_grant_facet(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert row[COL["grant_facet"]] == ["NASA NNX12AB34G", "NSF AST-1234567"]

    def test_grant_dict_elements_serialized(self) -> None:
        """ADS API may return grant as list of dicts; these are JSON-serialized."""
        rec = {**MINIMAL_RECORD, "grant": [{"agency": "NASA", "id": "123"}]}
        row, _ = transform_record(rec)
        grant_facet = row[COL["grant_facet"]]
        assert len(grant_facet) == 1
        # Dict elements are serialized to JSON strings for TEXT[] compatibility
        parsed = json.loads(grant_facet[0])
        assert parsed == {"agency": "NASA", "id": "123"}

    def test_new_text_fields_none_when_missing(self) -> None:
        row, _ = transform_record(MINIMAL_RECORD)
        assert row[COL["ack"]] is None
        assert row[COL["eid"]] is None
        assert row[COL["facility"]] is None
        assert row[COL["data"]] is None

    def test_new_text_fields_null_bytes_stripped(self) -> None:
        rec = {**MINIMAL_RECORD, "ack": "Thanks\x00reviewer", "series": "Ap\x00JL"}
        row, _ = transform_record(rec)
        assert row[COL["ack"]] == "Thanksreviewer"
        assert row[COL["series"]] == "ApJL"

    def test_new_array_fields_null_bytes_stripped(self) -> None:
        rec = {**MINIMAL_RECORD, "data": ["SIMBAD\x00:4"]}
        row, _ = transform_record(rec)
        assert row[COL["data"]] == ["SIMBAD:4"]

    def test_ack_not_in_raw(self) -> None:
        """ack was previously unmapped and in raw; now it's a dedicated column."""
        rec = {**MINIMAL_RECORD, "ack": "We thank the reviewer."}
        row, _ = transform_record(rec)
        assert row[COL["ack"]] == "We thank the reviewer."
        # raw should be None since ack is the only extra field and it's now mapped
        assert row[COL["raw"]] is None


class TestFieldSetConsistency:
    """Verify that field sets are internally consistent."""

    def test_all_column_order_entries_are_mapped_or_special(self) -> None:
        """Every column in COLUMN_ORDER must be either a direct field, a rename
        target, a special-cased field (title, year), or 'raw'."""
        direct_cols = (
            DIRECT_TEXT_FIELDS | DIRECT_ARRAY_FIELDS | DIRECT_INT_FIELDS | DIRECT_FLOAT_FIELDS
        )
        rename_targets = set(RENAMES.values())
        special = {"title", "year", "raw"}
        all_covered = direct_cols | rename_targets | special
        for col in COLUMN_ORDER:
            assert col in all_covered, f"Column {col!r} not covered by any field set"

    def test_no_overlap_between_field_sets(self) -> None:
        """Field sets should be disjoint (no double-processing)."""
        sets = [DIRECT_TEXT_FIELDS, DIRECT_ARRAY_FIELDS, DIRECT_INT_FIELDS, DIRECT_FLOAT_FIELDS]
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                if i < j:
                    overlap = a & b
                    assert not overlap, f"Overlap between sets {i} and {j}: {overlap}"


class TestColumnOrder:
    def test_column_count_matches_schema(self) -> None:
        # papers table: 33 original + 34 new full-coverage + raw = 68 columns
        assert len(COLUMN_ORDER) == 68

    def test_bibcode_is_first(self) -> None:
        assert COLUMN_ORDER[0] == "bibcode"

    def test_raw_is_last(self) -> None:
        assert COLUMN_ORDER[-1] == "raw"

    def test_no_duplicates(self) -> None:
        assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER))

    def test_row_length_matches_column_count(self) -> None:
        row, _ = transform_record(FULL_RECORD)
        assert len(row) == len(COLUMN_ORDER)
