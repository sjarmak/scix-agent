"""Unit tests for PaperStub dataclass."""

from __future__ import annotations

import pytest

from scix.stubs import PaperStub


class TestPaperStubFromRow:
    def test_full_row(self) -> None:
        row = {
            "bibcode": "2024ApJ...962L..15J",
            "title": "Predicting Solar Cycle 25",
            "first_author": "Jha, B.",
            "year": 2024,
            "citation_count": 8,
            "abstract": "We present a novel analysis of gravitational wave signals "
            "detected by advanced LIGO during the third observing run, "
            "focusing on template-based matched filtering techniques.",
        }
        stub = PaperStub.from_row(row)
        assert stub.bibcode == "2024ApJ...962L..15J"
        assert stub.title == "Predicting Solar Cycle 25"
        assert stub.first_author == "Jha, B."
        assert stub.year == 2024
        assert stub.citation_count == 8
        assert stub.abstract_snippet is not None
        assert len(stub.abstract_snippet) <= 153  # 150 + "..."
        assert stub.abstract_snippet.endswith("...")

    def test_short_abstract_no_ellipsis(self) -> None:
        row = {
            "bibcode": "2024test...001A",
            "title": "Short Paper",
            "first_author": "Author, A.",
            "year": 2024,
            "citation_count": 0,
            "abstract": "Short abstract.",
        }
        stub = PaperStub.from_row(row)
        assert stub.abstract_snippet == "Short abstract."
        assert not stub.abstract_snippet.endswith("...")

    def test_missing_abstract(self) -> None:
        row = {
            "bibcode": "2024test...002B",
            "title": "No Abstract Paper",
            "first_author": "Author, B.",
            "year": 2024,
            "citation_count": None,
            "abstract": None,
        }
        stub = PaperStub.from_row(row)
        assert stub.abstract_snippet is None

    def test_missing_optional_fields(self) -> None:
        row = {"bibcode": "2024test...003C"}
        stub = PaperStub.from_row(row)
        assert stub.bibcode == "2024test...003C"
        assert stub.title is None
        assert stub.first_author is None
        assert stub.year is None
        assert stub.citation_count is None
        assert stub.abstract_snippet is None

    def test_frozen(self) -> None:
        row = {
            "bibcode": "2024test...001A",
            "title": "Test",
            "first_author": "A",
            "year": 2024,
            "citation_count": 0,
            "abstract": None,
        }
        stub = PaperStub.from_row(row)
        with pytest.raises(AttributeError):
            stub.title = "Modified"  # type: ignore[misc]


class TestPaperStubToDict:
    def test_roundtrip(self) -> None:
        stub = PaperStub(
            bibcode="2024test...001A",
            title="Test Paper",
            first_author="Author, A.",
            year=2024,
            citation_count=5,
            abstract_snippet="First 150 chars...",
        )
        d = stub.to_dict()
        assert d["bibcode"] == "2024test...001A"
        assert d["title"] == "Test Paper"
        assert d["first_author"] == "Author, A."
        assert d["year"] == 2024
        assert d["citation_count"] == 5
        assert d["abstract_snippet"] == "First 150 chars..."
        assert len(d) == 6

    def test_none_values_preserved(self) -> None:
        stub = PaperStub(
            bibcode="2024test...001A",
            title=None,
            first_author=None,
            year=None,
            citation_count=None,
            abstract_snippet=None,
        )
        d = stub.to_dict()
        assert d["title"] is None
        assert d["abstract_snippet"] is None
