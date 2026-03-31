"""PaperStub: compact paper representation shared across all tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperStub:
    """Compact paper summary returned by search and listing tools.

    Designed to be token-efficient for agent consumption while providing
    enough context for the agent to decide whether to fetch full metadata.
    """

    bibcode: str
    title: str | None
    first_author: str | None
    year: int | None
    citation_count: int | None
    abstract_snippet: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PaperStub:
        """Construct from a database row (dict-style cursor result).

        Expects keys: bibcode, title, first_author, year, citation_count, abstract.
        The abstract is truncated to 150 characters for the snippet.
        """
        abstract = row.get("abstract")
        snippet: str | None = None
        if abstract:
            snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract

        return cls(
            bibcode=row["bibcode"],
            title=row.get("title"),
            first_author=row.get("first_author"),
            year=row.get("year"),
            citation_count=row.get("citation_count"),
            abstract_snippet=snippet,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON responses."""
        return {
            "bibcode": self.bibcode,
            "title": self.title,
            "first_author": self.first_author,
            "year": self.year,
            "citation_count": self.citation_count,
            "abstract_snippet": self.abstract_snippet,
        }
