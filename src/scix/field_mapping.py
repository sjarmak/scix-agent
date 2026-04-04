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
# Grouped: original columns first, then new full-coverage columns, raw JSONB last.
COLUMN_ORDER: tuple[str, ...] = (
    # --- Original columns (migrations 001 + 010) ---
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
    "body",
    # --- New full-coverage columns (migration 012) ---
    # Text fields
    "ack",
    "date",
    "eid",
    "entdate",
    "first_author_norm",
    "page_range",
    "pubnote",
    "series",
    # Array fields
    "aff_id",
    "alternate_title",
    "author_norm",
    "caption",
    "comment",
    "data",
    "esources",
    "facility",
    "grant_facet",
    "grant_agencies",
    "grant_id",
    "isbn",
    "issn",
    "keyword_norm",
    "keyword_schema",
    "links_data",
    "nedid",
    "nedtype",
    "orcid_other",
    "simbid",
    "vizier",
    # Integer fields
    "author_count",
    "page_count",
    # Real (float) fields
    "citation_count_norm",
    "cite_read_boost",
    "classic_factor",
    # --- Always last ---
    "raw",
)

# JSONL fields that map directly to SQL columns (same name, same type).
DIRECT_TEXT_FIELDS: frozenset[str] = frozenset(
    {
        "bibcode",
        "abstract",
        "body",
        "doctype",
        "pub",
        "pub_raw",
        "volume",
        "issue",
        "first_author",
        "copyright",
        "lang",
        "pubdate",
        "entry_date",
        "indexstamp",
        # New in migration 012
        "ack",
        "date",
        "eid",
        "entdate",
        "first_author_norm",
        "page_range",
        "pubnote",
        "series",
    }
)

# JSONL fields that map directly as arrays (same name).
DIRECT_ARRAY_FIELDS: frozenset[str] = frozenset(
    {
        "page",
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
        # New in migration 012
        "aff_id",
        "alternate_title",
        "author_norm",
        "caption",
        "comment",
        "data",
        "esources",
        "facility",
        "grant_agencies",
        "grant_id",
        "isbn",
        "issn",
        "keyword_norm",
        "keyword_schema",
        "links_data",
        "nedid",
        "nedtype",
        "orcid_other",
        "simbid",
        "vizier",
    }
)

# JSONL fields that map directly as integers.
DIRECT_INT_FIELDS: frozenset[str] = frozenset(
    {
        "citation_count",
        "read_count",
        "reference_count",
        # New in migration 012
        "author_count",
        "page_count",
    }
)

# JSONL fields that map directly as floats (REAL in SQL).
DIRECT_FLOAT_FIELDS: frozenset[str] = frozenset(
    {
        "citation_count_norm",
        "cite_read_boost",
        "classic_factor",
    }
)

# JSONL field -> SQL column renames.
RENAMES: dict[str, str] = {
    "author": "authors",
    "aff": "affiliations",
    "keyword": "keywords",
    "grant": "grant_facet",  # ADS "grant" is list[str]; renamed to avoid SQL keyword
}

# All JSONL fields that have a dedicated SQL column (after rename).
# NOTE: "reference" and "citation" are intentionally NOT here.
# "reference" is consumed for edge extraction and also preserved in raw for provenance.
# "citation" is kept only in raw (incomplete per ADS API; derived from reference[] inverse).
_MAPPED_JSONL_FIELDS: frozenset[str] = (
    DIRECT_TEXT_FIELDS
    | DIRECT_ARRAY_FIELDS
    | DIRECT_INT_FIELDS
    | DIRECT_FLOAT_FIELDS
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

    # Direct float fields (REAL in SQL — normalized scores from ADS)
    for field in DIRECT_FLOAT_FIELDS:
        val = rec.get(field)
        if val is not None:
            try:
                row[field] = float(val)
            except (ValueError, TypeError):
                row[field] = None

    # Renamed fields: author -> authors, aff -> affiliations,
    #                  keyword -> keywords, grant -> grant_facet
    for jsonl_name, sql_name in RENAMES.items():
        val = rec.get(jsonl_name)
        if isinstance(val, list):
            # Coerce non-string elements (e.g., grant dicts) to JSON strings
            # so TEXT[] columns always receive strings.
            sanitized: list[Any] = []
            for v in val:
                if isinstance(v, str):
                    sanitized.append(_sanitize_text(v))
                elif isinstance(v, dict):
                    sanitized.append(json.dumps(v, ensure_ascii=False))
                else:
                    sanitized.append(v)
            row[sql_name] = sanitized
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
