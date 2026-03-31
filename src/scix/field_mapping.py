"""Map ADS JSONL records to PostgreSQL papers table columns and citation edges."""

from __future__ import annotations

import json
from typing import Any


def _sanitize_text(val: str | None) -> str | None:
    """Remove null bytes that PostgreSQL rejects in text columns."""
    if val is None:
        return None
    return val.replace("\x00", "")

# Column order for COPY INTO papers. Must match the INSERT/COPY column list exactly.
COLUMN_ORDER: tuple[str, ...] = (
    "bibcode",
    "title",
    "abstract",
    "year",
    "doctype",
    "pub",
    "pub_raw",
    "volume",
    "issue",
    "page",
    "authors",
    "first_author",
    "affiliations",
    "keywords",
    "arxiv_class",
    "database",
    "doi",
    "identifier",
    "alternate_bibcode",
    "bibstem",
    "bibgroup",
    "orcid_pub",
    "orcid_user",
    "property",
    "copyright",
    "lang",
    "pubdate",
    "entry_date",
    "indexstamp",
    "citation_count",
    "read_count",
    "reference_count",
    "raw",
)

# JSONL fields that map directly to SQL columns (same name, same type).
DIRECT_TEXT_FIELDS: frozenset[str] = frozenset({
    "bibcode", "abstract", "doctype", "pub", "pub_raw", "volume", "issue",
    "first_author", "copyright", "lang", "pubdate", "entry_date", "indexstamp",
})

# JSONL fields that map directly as arrays (same name).
DIRECT_ARRAY_FIELDS: frozenset[str] = frozenset({
    "page", "arxiv_class", "database", "doi", "identifier",
    "alternate_bibcode", "bibstem", "bibgroup", "orcid_pub", "orcid_user", "property",
})

# JSONL fields that map directly as integers.
DIRECT_INT_FIELDS: frozenset[str] = frozenset({
    "citation_count", "read_count", "reference_count",
})

# JSONL field -> SQL column renames.
RENAMES: dict[str, str] = {
    "author": "authors",
    "aff": "affiliations",
    "keyword": "keywords",
}

# All JSONL fields that have a dedicated SQL column (after rename).
# NOTE: "reference" and "citation" are intentionally NOT here.
# "reference" is consumed for edge extraction and also preserved in raw for provenance.
# "citation" is kept only in raw (incomplete per ADS API; derived from reference[] inverse).
_MAPPED_JSONL_FIELDS: frozenset[str] = (
    DIRECT_TEXT_FIELDS
    | DIRECT_ARRAY_FIELDS
    | DIRECT_INT_FIELDS
    | frozenset(RENAMES.keys())
    | {"title", "year"}  # special-cased transforms
)


def transform_record(rec: dict[str, Any]) -> tuple[tuple[Any, ...], list[tuple[str, str]]]:
    """Transform a JSONL record into a (paper_row, edges) pair.

    Returns:
        paper_row: tuple of values in COLUMN_ORDER, ready for COPY write_row.
        edges: list of (source_bibcode, target_bibcode) from the reference[] field.

    Raises:
        ValueError: if the record is missing a bibcode.
    """
    bibcode = rec.get("bibcode")
    if not bibcode:
        raise ValueError("Record missing bibcode")

    row: dict[str, Any] = dict.fromkeys(COLUMN_ORDER)

    # Direct text fields — sanitize to strip null bytes PostgreSQL rejects
    for field in DIRECT_TEXT_FIELDS:
        row[field] = _sanitize_text(rec.get(field))

    # Direct array fields — sanitize string elements within arrays
    for field in DIRECT_ARRAY_FIELDS:
        val = rec.get(field)
        if isinstance(val, list):
            row[field] = [_sanitize_text(v) if isinstance(v, str) else v for v in val]
        else:
            row[field] = None

    # Direct integer fields (guard against non-numeric values in source data)
    for field in DIRECT_INT_FIELDS:
        val = rec.get(field)
        if val is not None:
            try:
                row[field] = int(val)
            except (ValueError, TypeError):
                row[field] = None

    # Renamed array fields: author -> authors, aff -> affiliations, keyword -> keywords
    for jsonl_name, sql_name in RENAMES.items():
        val = rec.get(jsonl_name)
        if isinstance(val, list):
            row[sql_name] = [_sanitize_text(v) if isinstance(v, str) else v for v in val]
        else:
            row[sql_name] = None

    # Special transforms
    # title: list[str] in JSONL -> first element as str
    title_val = rec.get("title")
    if isinstance(title_val, list) and title_val:
        row["title"] = _sanitize_text(title_val[0])
    elif isinstance(title_val, str):
        row["title"] = _sanitize_text(title_val)
    else:
        row["title"] = None

    # year: str in JSONL -> SMALLINT
    try:
        row["year"] = int(rec["year"]) if "year" in rec else None
    except (ValueError, TypeError):
        row["year"] = None

    # Collect unmapped fields into raw JSONB (sanitize string values before serialization)
    raw_fields = {}
    for k, v in rec.items():
        if k not in _MAPPED_JSONL_FIELDS:
            raw_fields[k] = _sanitize_text(v) if isinstance(v, str) else v
    row["raw"] = json.dumps(raw_fields) if raw_fields else None

    # Build the tuple in COLUMN_ORDER
    paper_row = tuple(row[col] for col in COLUMN_ORDER)

    # Extract citation edges from reference[] (outgoing references)
    edges: list[tuple[str, str]] = []
    references = rec.get("reference")
    if isinstance(references, list):
        for ref_bibcode in references:
            if isinstance(ref_bibcode, str) and ref_bibcode:
                edges.append((bibcode, ref_bibcode))

    return paper_row, edges
