"""Docling PDF parser for the non-arXiv tail.

Last-resort fallback for papers without LaTeX source, S2ORC body, or ADS body.
Uses IBM Docling with its pretrained Heron layout model + TableFormer + equation
recognition. NO custom training is performed.

The module:
    1. Resolves a PDF URL from OpenAlex ``best_oa_location.pdf_url``
    2. Parses with Docling's DocumentConverter (lazy import)
    3. Normalizes the result into ``papers_fulltext`` rows with ``source='docling'``
    4. Records failures to ``papers_fulltext_failures`` with R15 exponential backoff

SAFETY:
    * Docling is imported lazily — the module is importable without it installed.
    * Production DSN guard via ``check_production_guard()`` (same pattern as ar5iv.py).
    * PDF URLs are validated (http/https only) to prevent SSRF — both in
      ``resolve_pdf_url()`` and inside ``parse_pdf_with_docling()``.
    * All tables are LOGGED (never UNLOGGED).
    * DB functions accept ``psycopg.Connection`` and commit per-call (autocommit style).

See also:
    - ``src/scix/sources/route.py`` (routing decision tree)
    - ``migrations/047_papers_fulltext_failures.sql`` (failure queue schema)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn

logger = logging.getLogger(__name__)

__all__ = [
    "PARSER_VERSION",
    "ProductionGuardError",
    "SOURCE_TAG",
    "DoclingConfig",
    "DoclingResult",
    "FailureRecord",
    "check_production_guard",
    "compute_retry_after",
    "normalize_to_fulltext_row",
    "parse_pdf_with_docling",
    "record_failure",
    "resolve_pdf_url",
    "upsert_fulltext_row",
]

SOURCE_TAG = "docling"
PARSER_VERSION = "docling@v1"

# R15 exponential backoff schedule (attempt -> timedelta).
_BACKOFF_SCHEDULE: dict[int, dt.timedelta] = {
    1: dt.timedelta(hours=24),
    2: dt.timedelta(days=3),
    3: dt.timedelta(days=7),
}
_BACKOFF_DEFAULT = dt.timedelta(days=30)

# Allowed URL schemes for PDF downloads (SSRF prevention).
_ALLOWED_SCHEMES = frozenset({"http", "https"})


# ---------------------------------------------------------------------------
# Data classes (all frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoclingResult:
    """Immutable result of parsing a PDF with Docling."""

    sections: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    equations: list[dict[str, Any]]
    parser_version: str
    page_count: int
    source_pdf_url: str


@dataclass(frozen=True)
class FailureRecord:
    """Immutable record of a failed parse attempt."""

    bibcode: str
    parser_version: str
    failure_reason: str
    attempts: int
    retry_after: dt.datetime


@dataclass(frozen=True)
class DoclingConfig:
    """Immutable configuration for the Docling PDF pipeline."""

    dsn: str | None
    batch_size: int = 100
    dry_run: bool = False
    yes_production: bool = False


class ProductionGuardError(RuntimeError):
    """Raised when attempting to write to production without authorization."""


def check_production_guard(cfg: DoclingConfig) -> None:
    """Refuse to run against production unless explicitly authorized."""
    effective_dsn = cfg.dsn or DEFAULT_DSN
    if is_production_dsn(effective_dsn) and not cfg.yes_production:
        raise ProductionGuardError(
            "Refusing to run Docling pipeline against production DSN "
            f"({redact_dsn(effective_dsn)}). Pass yes_production=True "
            "to override."
        )


# ---------------------------------------------------------------------------
# Pure functions (no IO)
# ---------------------------------------------------------------------------


def compute_retry_after(
    attempts: int,
    now: dt.datetime | None = None,
) -> dt.datetime:
    """Compute the next retry timestamp using R15 exponential backoff.

    Schedule: attempt 1 -> +24h, 2 -> +3d, 3 -> +7d, >=4 -> +30d.
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    delta = _BACKOFF_SCHEDULE.get(attempts, _BACKOFF_DEFAULT)
    return now + delta


def resolve_pdf_url(openalex_meta: dict[str, Any]) -> str | None:
    """Extract a usable PDF URL from OpenAlex metadata.

    Checks ``best_oa_location.pdf_url``. Returns None if absent or if the
    URL scheme is not http/https (SSRF prevention).
    """
    best_oa = openalex_meta.get("best_oa_location")
    if not isinstance(best_oa, dict):
        return None
    pdf_url = best_oa.get("pdf_url")
    if not isinstance(pdf_url, str) or not pdf_url:
        return None
    # Validate URL scheme to prevent SSRF
    scheme = pdf_url.split("://", 1)[0].lower() if "://" in pdf_url else ""
    if scheme not in _ALLOWED_SCHEMES:
        logger.warning("Rejected PDF URL with disallowed scheme: %s", scheme)
        return None
    return pdf_url


def normalize_to_fulltext_row(
    bibcode: str,
    result: DoclingResult,
) -> dict[str, Any]:
    """Convert a DoclingResult into a dict matching the papers_fulltext schema.

    Returns a dict with keys: bibcode, source, sections, inline_cites,
    figures, tables, equations, parser_version. Values for JSON columns are
    serialized strings.
    """
    return {
        "bibcode": bibcode,
        "source": SOURCE_TAG,
        "sections": json.dumps(result.sections, ensure_ascii=False),
        "inline_cites": json.dumps([], ensure_ascii=False),  # Docling does not extract inline cites
        "figures": json.dumps(result.figures, ensure_ascii=False),
        "tables": json.dumps(result.tables, ensure_ascii=False),
        "equations": json.dumps(result.equations, ensure_ascii=False),
        "parser_version": result.parser_version,
    }


# ---------------------------------------------------------------------------
# Docling PDF parser (lazy import of docling)
# ---------------------------------------------------------------------------


def _import_docling() -> Any:
    """Lazily import docling. Raises ImportError with install hint."""
    try:
        from docling.document_converter import DocumentConverter

        return DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "Docling is required for PDF parsing. "
            "Install it with: pip install 'docling>=2.0'"
        ) from exc


def parse_pdf_with_docling(pdf_path_or_url: str) -> DoclingResult:
    """Parse a PDF file or URL with Docling's DocumentConverter.

    Uses the pretrained Heron layout model, TableFormer, and equation
    recognition. NO custom training.

    Parameters
    ----------
    pdf_path_or_url : str
        Local file path or HTTPS URL to a PDF.

    Returns
    -------
    DoclingResult
        Parsed sections, figures, tables, equations.

    Raises
    ------
    ValueError
        If the input contains a URL with a disallowed scheme (SSRF prevention).
    """
    if "://" in pdf_path_or_url:
        scheme = pdf_path_or_url.split("://", 1)[0].lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"Disallowed URL scheme {scheme!r}. "
                "Only http/https are permitted."
            )
    DocumentConverter = _import_docling()
    converter = DocumentConverter()
    result = converter.convert(pdf_path_or_url)
    doc = result.document

    # Single pass over document items — classify by type name.
    sections: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    offset = 0

    for item in doc.iterate_items():
        # Docling v2 returns (path, item) tuples from iterate_items.
        obj = item[1] if isinstance(item, tuple) else item
        type_name = type(obj).__name__

        if type_name == "SectionHeaderItem":
            heading = obj.text if hasattr(obj, "text") else ""
            sections.append({
                "heading": heading,
                "level": getattr(obj, "level", 1),
                "text": "",
                "offset": offset,
            })

        elif type_name == "TextItem":
            text = obj.text if hasattr(obj, "text") else ""
            sections.append({
                "heading": "",
                "level": 1,
                "text": text,
                "offset": offset,
            })
            offset += len(text)

        elif type_name == "PictureItem":
            caption = "".join(
                getattr(cap, "text", "") for cap in getattr(obj, "captions", [])
            )
            figures.append({"id": str(getattr(obj, "self_ref", "")), "caption": caption})

        elif type_name == "TableItem":
            caption = "".join(
                getattr(cap, "text", "") for cap in getattr(obj, "captions", [])
            )
            tables.append({"id": str(getattr(obj, "self_ref", "")), "caption": caption})

        elif type_name == "FormulaItem":
            equations.append({
                "id": str(getattr(obj, "self_ref", "")),
                "latex": getattr(obj, "text", ""),
            })

    page_count = len(doc.pages) if hasattr(doc, "pages") else 0

    return DoclingResult(
        sections=sections,
        figures=figures,
        tables=tables,
        equations=equations,
        parser_version=PARSER_VERSION,
        page_count=page_count,
        source_pdf_url=str(pdf_path_or_url),
    )


# ---------------------------------------------------------------------------
# DB operations (failure logging)
# ---------------------------------------------------------------------------


_UPSERT_FAILURE_SQL = """\
INSERT INTO papers_fulltext_failures
    (bibcode, parser_version, failure_reason, attempts, retry_after, last_attempt)
VALUES (%(bibcode)s, %(parser_version)s, %(failure_reason)s, %(attempts)s, %(retry_after)s, now())
ON CONFLICT (bibcode) DO UPDATE SET
    parser_version = EXCLUDED.parser_version,
    failure_reason = EXCLUDED.failure_reason,
    attempts = papers_fulltext_failures.attempts + 1,
    last_attempt = now(),
    retry_after = %(retry_after)s
"""

_UPSERT_FULLTEXT_SQL = """\
INSERT INTO papers_fulltext
    (bibcode, source, sections, inline_cites, figures, tables, equations, parser_version)
VALUES
    (%(bibcode)s, %(source)s, %(sections)s::jsonb, %(inline_cites)s::jsonb,
     %(figures)s::jsonb, %(tables)s::jsonb, %(equations)s::jsonb, %(parser_version)s)
ON CONFLICT (bibcode) DO UPDATE SET
    source = EXCLUDED.source,
    sections = EXCLUDED.sections,
    inline_cites = EXCLUDED.inline_cites,
    figures = EXCLUDED.figures,
    tables = EXCLUDED.tables,
    equations = EXCLUDED.equations,
    parser_version = EXCLUDED.parser_version,
    parsed_at = now()
"""


def record_failure(
    conn: psycopg.Connection,  # type: ignore[type-arg]
    bibcode: str,
    failure_reason: str,
    attempts: int = 1,
) -> FailureRecord:
    """Record a parse failure to papers_fulltext_failures.

    Uses UPSERT: first failure inserts, subsequent failures increment attempts
    and extend retry_after per R15 backoff. Commits immediately (autocommit style).
    """
    retry_after = compute_retry_after(attempts)
    params = {
        "bibcode": bibcode,
        "parser_version": PARSER_VERSION,
        "failure_reason": failure_reason,
        "attempts": attempts,
        "retry_after": retry_after,
    }
    with conn.cursor() as cur:
        cur.execute(_UPSERT_FAILURE_SQL, params)
    conn.commit()

    return FailureRecord(
        bibcode=bibcode,
        parser_version=PARSER_VERSION,
        failure_reason=failure_reason,
        attempts=attempts,
        retry_after=retry_after,
    )


def upsert_fulltext_row(
    conn: psycopg.Connection,  # type: ignore[type-arg]
    row: dict[str, Any],
) -> None:
    """Insert or update a papers_fulltext row. Commits immediately."""
    with conn.cursor() as cur:
        cur.execute(_UPSERT_FULLTEXT_SQL, row)
    conn.commit()
