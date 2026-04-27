"""Citation context extraction pipeline.

Extracts ~250-word context windows around inline citation markers [N] in paper
body text, resolves numbered markers to target bibcodes via the paper's
reference[] array, and stores results in the citation_contexts table.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import psycopg

from scix.db import get_connection
from scix.section_parser import parse_sections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures (frozen for immutability)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationMarker:
    """A citation marker found in body text.

    Numbered ``[N]`` markers populate ``marker_numbers`` and leave the
    author-year fields empty.  Author-year markers (e.g. ``Hong et al. 2001``)
    populate ``marker_authors`` and ``marker_year`` and leave
    ``marker_numbers`` empty.  Both shapes are unified into the same
    dataclass so downstream stages handle them uniformly.
    """

    marker_text: str  # e.g. "[1]", "[1, 2, 3]", "[1-3]", "Hong et al. 2001"
    marker_numbers: tuple[int, ...]  # numbered refs (1-indexed); () for author-year
    char_start: int  # start offset in body
    char_end: int  # end offset in body
    context_text: str  # ~250-word window around the marker
    context_start: int  # char offset where context window begins
    section_name: str | None = None  # which section this appears in
    # Author-year fields. Empty/None for numbered markers.
    marker_authors: tuple[str, ...] = ()  # surnames in citation order; first is primary
    marker_year: int | None = None


@dataclass(frozen=True)
class CitationContext:
    """A resolved citation context ready for DB insertion."""

    source_bibcode: str
    target_bibcode: str
    context_text: str
    char_offset: int
    section_name: str | None = None
    intent: str | None = None


# ---------------------------------------------------------------------------
# Regex patterns for citation markers
# ---------------------------------------------------------------------------

# Matches [N], [N, M, ...], and [N-M] patterns where N, M are digits.
# Excludes markers that look like author-year (contain letters).
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")


def _parse_marker_numbers(inner: str) -> tuple[int, ...]:
    """Parse the interior of a citation marker into a tuple of 1-indexed ints.

    Handles:
      "1"         -> (1,)
      "1, 2, 3"   -> (1, 2, 3)
      "1-3"       -> (1, 2, 3)
      "1, 3-5"    -> (1, 3, 4, 5)
    """
    numbers: list[int] = []
    parts = [p.strip() for p in inner.split(",")]
    for part in parts:
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                lo = int(bounds[0].strip())
                hi = int(bounds[1].strip())
                numbers.extend(range(lo, hi + 1))
            except (ValueError, IndexError):
                continue
        else:
            try:
                numbers.append(int(part))
            except ValueError:
                continue
    return tuple(numbers)


def _word_boundary_window(
    text: str, char_start: int, char_end: int, words: int = 125
) -> tuple[int, int]:
    """Find a ~words-before and ~words-after window around a span.

    Returns (window_start, window_end) as char offsets into text.
    """
    # Walk backward from char_start to find ~`words` word boundaries
    before_text = text[:char_start]
    before_tokens = before_text.split()
    if len(before_tokens) <= words:
        window_start = 0
    else:
        # Find the position of the (words)th-from-last token
        kept = before_tokens[-words:]
        window_start = before_text.rfind(kept[0], 0, char_start)
        if window_start < 0:
            window_start = 0

    # Walk forward from char_end to find ~`words` word boundaries
    after_text = text[char_end:]
    after_tokens = after_text.split()
    if len(after_tokens) <= words:
        window_end = len(text)
    else:
        # Accumulate words forward
        consumed = 0
        for i, token in enumerate(after_tokens):
            if i >= words:
                break
            pos = after_text.find(token, consumed)
            consumed = pos + len(token)
        window_end = char_end + consumed

    return window_start, window_end


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------


def extract_citation_contexts(body: str) -> list[CitationMarker]:
    """Find [N] patterns in text and return ~250-word context windows.

    Parameters
    ----------
    body : str
        Plain-text body of a paper.

    Returns
    -------
    list[CitationMarker]
        One entry per citation marker found, with context window and offsets.
    """
    if not body:
        return []

    markers: list[CitationMarker] = []
    for m in _CITATION_RE.finditer(body):
        inner = m.group(1)
        nums = _parse_marker_numbers(inner)
        if not nums:
            continue

        char_start = m.start()
        char_end = m.end()
        win_start, win_end = _word_boundary_window(body, char_start, char_end)
        context = body[win_start:win_end]

        markers.append(
            CitationMarker(
                marker_text=m.group(0),
                marker_numbers=nums,
                char_start=char_start,
                char_end=char_end,
                context_text=context,
                context_start=win_start,
            )
        )

    return markers


def resolve_citation_markers(
    markers: list[CitationMarker],
    references: list[str],
    source_bibcode: str,
) -> list[CitationContext]:
    """Map [N] markers to target bibcodes using a reference list.

    Parameters
    ----------
    markers : list[CitationMarker]
        Markers extracted from body text.
    references : list[str]
        Ordered list of reference bibcodes (0-indexed array; markers are 1-indexed).
    source_bibcode : str
        Bibcode of the citing paper.

    Returns
    -------
    list[CitationContext]
        One context per (marker, resolved bibcode) pair.  Markers with
        N > len(references) are silently skipped.
    """
    contexts: list[CitationContext] = []
    for marker in markers:
        for n in marker.marker_numbers:
            idx = n - 1  # markers are 1-indexed
            if idx < 0 or idx >= len(references):
                continue
            target = references[idx]
            if not isinstance(target, str) or not target:
                continue
            contexts.append(
                CitationContext(
                    source_bibcode=source_bibcode,
                    target_bibcode=target,
                    context_text=marker.context_text,
                    char_offset=marker.char_start,
                    section_name=marker.section_name,
                )
            )
    return contexts


# ---------------------------------------------------------------------------
# Author-year citation marker extraction
# ---------------------------------------------------------------------------

# Year range considered plausible for a citation. Values outside this range
# (e.g. 'Section 2099' or 'Smith 1066') are rejected as noise.
_MIN_CITATION_YEAR = 1500
_MAX_CITATION_YEAR = 2099

# Surname-shaped tokens that are almost always false positives in author-year
# context (capitalized common nouns followed by a number that happens to fall
# inside the year range, e.g. 'Section 2020 reports').
_SURNAME_FALSE_POSITIVES = frozenset(
    {
        "Figure",
        "Fig",
        "Table",
        "Section",
        "Sect",
        "Equation",
        "Eq",
        "Eqn",
        "Ref",
        "Refs",
        "Vol",
        "Volume",
        "Page",
        "Chapter",
        "Chap",
        "Appendix",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Sept",
        "Oct",
        "Nov",
        "Dec",
    }
)

# Surname pattern: capitalized word, optionally hyphenated (e.g. "Smith-Jones").
# The leading character is `[A-Z]` and the rest is `[a-z]+` so a single-letter
# initial like "J." won't match — we strip leading initials separately.
_SURNAME = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"

# Year token: 4 digits. Range is validated post-match.
_YEAR = r"(\d{4})"

# Pattern A — narrative form: "Surname (YYYY)" or "Surname and Other (YYYY)".
# The optional second author after "and" / "&" is captured but not strictly
# required. We anchor with a word boundary so we don't match within tokens.
_AY_NARRATIVE = re.compile(
    rf"\b({_SURNAME})"
    rf"(?:\s+(?:and|&)\s+(?:{_SURNAME}))?"
    rf"\s*\(\s*{_YEAR}\s*\)"
)

# Pattern B — "Surname et al. YYYY" / "Surname et al., YYYY".
_AY_ET_AL = re.compile(
    rf"\b({_SURNAME})\s+et\s+al\.?,?\s*\(?\s*{_YEAR}\s*\)?"
)

# Pattern C — fully parenthetical: "(Surname, YYYY)" / "(Surname & Other, YYYY)"
# / "(Surname et al., YYYY)" / "(Surname, Other, & Third, YYYY)".
# Capture only the first surname inside the parens — that's what the bibcode
# initial encodes.
_AY_PAREN = re.compile(
    rf"\(\s*({_SURNAME})"
    rf"(?:\s+et\s+al\.?)?"
    rf"(?:(?:\s+(?:and|&)\s+|,\s+|,\s*&\s+){_SURNAME})*"
    rf",?\s*{_YEAR}\s*\)"
)

# Pattern D — sub-citation inside a multi-cite paren block. Matches "Surname
# [et al.] [& Other], YYYY" only when preceded by '(' or '; ' (i.e. inside a
# paren-separated list like "(Adams, 2020; Smith & Jones, 2003)"). The leading
# delimiter is consumed as part of group(0) but group(1) is the surname only.
_AY_SUBCITE = re.compile(
    rf"(?<=[\(;])\s*({_SURNAME})"
    rf"(?:\s+et\s+al\.?)?"
    rf"(?:\s+(?:and|&)\s+{_SURNAME})?"
    rf",\s*{_YEAR}(?=\s*[;\)])"
)


def _is_valid_year(year: int) -> bool:
    return _MIN_CITATION_YEAR <= year <= _MAX_CITATION_YEAR


def _is_surname_candidate(token: str) -> bool:
    """Reject capitalized common-noun false positives."""
    return token not in _SURNAME_FALSE_POSITIVES


def extract_author_year_citations(body: str) -> list[CitationMarker]:
    """Find author-year citations in body text.

    Handles patterns:

      * ``Surname et al. YYYY`` / ``Surname et al., YYYY``
      * ``Surname (YYYY)`` / ``Surname and Other (YYYY)``
      * ``(Surname, YYYY)`` / ``(Surname & Other, YYYY)`` /
        ``(Surname et al., YYYY)`` / ``(Surname, Other, & Third, YYYY)``

    Returns one :class:`CitationMarker` per match with ``marker_authors`` set
    to a 1-tuple ``(first_surname,)`` and ``marker_year`` set to an
    in-range year.  Numbered ``[N]`` markers are not produced here; use
    :func:`extract_citation_contexts` for those.

    Overlapping matches across the three patterns are de-duplicated by
    ``char_start``.
    """
    if not body:
        return []

    spans: list[tuple[int, int]] = []  # accepted (start, end) for overlap check
    out: list[CitationMarker] = []

    def _overlaps(start: int, end: int) -> bool:
        for s, e in spans:
            if start < e and end > s:
                return True
        return False

    # Order matters: et-al pattern is tried before narrative because
    # narrative would otherwise mis-capture "Smith et al" as just "Smith".
    # The sub-citation pattern is last because it's the most permissive and
    # should only fire when none of the structured patterns already covered
    # the span.
    for pattern in (_AY_ET_AL, _AY_NARRATIVE, _AY_PAREN, _AY_SUBCITE):
        for m in pattern.finditer(body):
            char_start = m.start()
            char_end = m.end()
            if _overlaps(char_start, char_end):
                continue

            surname = m.group(1)
            if not _is_surname_candidate(surname):
                continue
            try:
                year = int(m.group(2))
            except (ValueError, IndexError):
                continue
            if not _is_valid_year(year):
                continue

            win_start, win_end = _word_boundary_window(body, char_start, char_end)
            context = body[win_start:win_end]

            spans.append((char_start, char_end))
            out.append(
                CitationMarker(
                    marker_text=m.group(0),
                    marker_numbers=(),
                    char_start=char_start,
                    char_end=char_end,
                    context_text=context,
                    context_start=win_start,
                    marker_authors=(surname,),
                    marker_year=year,
                )
            )

    out.sort(key=lambda mk: mk.char_start)
    return out


def resolve_author_year_markers(
    markers: list[CitationMarker],
    references: list[str],
    source_bibcode: str,
    min_confidence: float = 0.5,
) -> list[CitationContext]:
    """Resolve author-year markers to target bibcodes via name+year disambiguation.

    A reference bibcode encodes the first author's surname initial as its last
    character (uppercase) and the publication year as its first four
    characters.  For each marker, we filter the reference list to entries
    whose bibcode starts with the marker year *and* ends with the marker's
    first surname initial (case-insensitive).

    Resolution rules:

      * 0 candidates  -> drop (no citation emitted).
      * 1 candidate   -> resolve at confidence 1.0.
      * N candidates  -> confidence = 1/N.  All N candidates are emitted
        if confidence >= ``min_confidence``; otherwise the marker is
        dropped (deterministic, no LLM disambiguation).

    References whose last char is non-alphabetic (e.g. arXiv-style
    ``2020arXiv200112345.``) are excluded from the initial-match filter
    because they don't encode a surname-initial — over-resolving on year
    alone would inflate false positives.
    """
    if not markers or not references:
        return []

    contexts: list[CitationContext] = []

    for marker in markers:
        if marker.marker_year is None or not marker.marker_authors:
            continue

        first_surname = marker.marker_authors[0]
        if not first_surname:
            continue
        target_initial = first_surname[0].upper()
        year_prefix = f"{marker.marker_year:04d}"

        candidates: list[str] = []
        for ref in references:
            if not isinstance(ref, str) or len(ref) < 5:
                continue
            if not ref.startswith(year_prefix):
                continue
            last_char = ref[-1]
            if not last_char.isalpha():
                # Non-alphabetic terminator (e.g. arXiv refs) cannot be
                # disambiguated by surname initial — skip rather than
                # accept on year-only.
                continue
            if last_char.upper() != target_initial:
                continue
            candidates.append(ref)

        n = len(candidates)
        if n == 0:
            continue
        confidence = 1.0 if n == 1 else 1.0 / n
        if confidence < min_confidence:
            continue

        for target in candidates:
            contexts.append(
                CitationContext(
                    source_bibcode=source_bibcode,
                    target_bibcode=target,
                    context_text=marker.context_text,
                    char_offset=marker.char_start,
                    section_name=marker.section_name,
                )
            )

    return contexts


def _enrich_with_sections(
    markers: list[CitationMarker],
    sections: list[tuple[str, int, int, str]],
) -> list[CitationMarker]:
    """Annotate markers with the section they appear in."""
    enriched: list[CitationMarker] = []
    for marker in markers:
        section_name: str | None = None
        for sec_name, sec_start, sec_end, _sec_text in sections:
            if sec_start <= marker.char_start < sec_end:
                section_name = sec_name
                break
        enriched.append(
            CitationMarker(
                marker_text=marker.marker_text,
                marker_numbers=marker.marker_numbers,
                char_start=marker.char_start,
                char_end=marker.char_end,
                context_text=marker.context_text,
                context_start=marker.context_start,
                section_name=section_name,
                marker_authors=marker.marker_authors,
                marker_year=marker.marker_year,
            )
        )
    return enriched


def process_paper(
    bibcode: str,
    body: str,
    references: list[str],
) -> list[CitationContext]:
    """Extract and resolve citation contexts for a single paper.

    Runs both the numbered ``[N]`` extractor and the author-year extractor,
    enriches both with section names, and merges resolved contexts.
    Duplicate (target_bibcode, char_offset) pairs are de-duplicated so a
    single physical marker never produces two rows even if both extractors
    happen to fire on overlapping spans.

    Parameters
    ----------
    bibcode : str
        Bibcode of the source paper.
    body : str
        Plain-text body of the paper.
    references : list[str]
        Ordered reference bibcodes from the paper's metadata.

    Returns
    -------
    list[CitationContext]
        Resolved citation contexts ready for DB insertion.
    """
    if not body or not references:
        return []

    numbered_markers = extract_citation_contexts(body)
    author_year_markers = extract_author_year_citations(body)
    if not numbered_markers and not author_year_markers:
        return []

    sections = parse_sections(body)

    contexts: list[CitationContext] = []
    if numbered_markers:
        enriched = _enrich_with_sections(numbered_markers, sections)
        contexts.extend(resolve_citation_markers(enriched, references, bibcode))
    if author_year_markers:
        enriched = _enrich_with_sections(author_year_markers, sections)
        contexts.extend(resolve_author_year_markers(enriched, references, bibcode))

    if len(contexts) <= 1:
        return contexts

    seen: set[tuple[str, int]] = set()
    deduped: list[CitationContext] = []
    for ctx in contexts:
        key = (ctx.target_bibcode, ctx.char_offset)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ctx)
    return deduped


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

# Max chars of context_text persisted to the citation_contexts table.
# v_claim_edges (migrations/057) already truncates to 1000 chars at view-build
# time, so any additional bytes here are storage overhead with no downstream
# consumer. Cap matches the view to halve per-row size at scale; before this
# cap, full ~250-word windows averaged ~1900 bytes (1.45 GB at 825K rows).
_CONTEXT_TEXT_MAX_CHARS = 1000

_SELECT_PAPERS = """
    SELECT p.bibcode, p.body, p.raw
    FROM papers p
    WHERE p.body IS NOT NULL
      AND p.raw IS NOT NULL
      AND p.raw::jsonb ? 'reference'
      AND NOT EXISTS (
          SELECT 1 FROM citation_contexts cc
          WHERE cc.source_bibcode = p.bibcode
      )
"""

_CITCTX_STAGING_DDL = (
    "CREATE TEMP TABLE IF NOT EXISTS _citctx_staging "
    "(LIKE citation_contexts INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
)

_CITCTX_COPY = (
    "COPY _citctx_staging "
    "(source_bibcode, target_bibcode, context_text, char_offset, section_name, intent) "
    "FROM STDIN"
)

_CITCTX_MERGE = (
    "INSERT INTO citation_contexts "
    "(source_bibcode, target_bibcode, context_text, char_offset, section_name, intent) "
    "SELECT source_bibcode, target_bibcode, context_text, char_offset, section_name, intent "
    "FROM _citctx_staging"
)


def _flush_contexts(
    conn: psycopg.Connection,
    rows: list[tuple[str, str, str, int, str | None, str | None]],
) -> int:
    """COPY citation context rows into the DB. Returns row count."""
    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.execute(_CITCTX_STAGING_DDL)
        with cur.copy(_CITCTX_COPY) as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute(_CITCTX_MERGE)
        inserted = cur.rowcount

    conn.commit()
    return inserted


def run_pipeline(
    dsn: str | None = None,
    batch_size: int = 1000,
    limit: int | None = None,
) -> int:
    """Process papers from DB, extracting citation contexts in batches.

    Parameters
    ----------
    dsn : str | None
        Database connection string.  Falls back to DEFAULT_DSN.
    batch_size : int
        Number of context rows to accumulate before flushing via COPY.
    limit : int | None
        Maximum number of papers to process (None = all).

    Returns
    -------
    int
        Total number of citation context rows inserted.
    """
    read_conn = get_connection(dsn)
    write_conn = get_connection(dsn)
    total_inserted = 0
    papers_processed = 0
    t_start = time.monotonic()

    try:
        query = _SELECT_PAPERS
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        with read_conn.cursor(name="citctx_papers") as cur:
            cur.execute(query, params)

            batch: list[tuple[str, str, str, int, str | None, str | None]] = []

            for bibcode, body, raw_val in cur:
                # Parse references from raw JSONB
                if isinstance(raw_val, str):
                    try:
                        raw_dict = json.loads(raw_val)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(raw_val, dict):
                    raw_dict = raw_val
                else:
                    continue

                refs = raw_dict.get("reference")
                if not isinstance(refs, list):
                    continue

                contexts = process_paper(bibcode, body, refs)
                for ctx in contexts:
                    batch.append(
                        (
                            ctx.source_bibcode,
                            ctx.target_bibcode,
                            ctx.context_text[:_CONTEXT_TEXT_MAX_CHARS],
                            ctx.char_offset,
                            ctx.section_name,
                            ctx.intent,
                        )
                    )

                    if len(batch) >= batch_size:
                        inserted = _flush_contexts(write_conn, batch)
                        total_inserted += inserted
                        batch.clear()

                papers_processed += 1
                if papers_processed % 1000 == 0:
                    elapsed = time.monotonic() - t_start
                    rate = papers_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Processed %d papers, %d contexts inserted, %.0f papers/s",
                        papers_processed,
                        total_inserted,
                        rate,
                    )

            # Flush remaining
            if batch:
                inserted = _flush_contexts(write_conn, batch)
                total_inserted += inserted

        elapsed = time.monotonic() - t_start
        logger.info(
            "Pipeline complete: %d papers, %d contexts, %.1fs",
            papers_processed,
            total_inserted,
            elapsed,
        )

    finally:
        read_conn.close()
        write_conn.close()

    return total_inserted
