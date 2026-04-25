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

from scix.db import DEFAULT_DSN, get_connection
from scix.section_parser import parse_sections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures (frozen for immutability)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationMarker:
    """A citation marker found in body text."""

    marker_text: str  # e.g. "[1]", "[1, 2, 3]", "[1-3]"
    marker_numbers: tuple[int, ...]  # resolved integer references (1-indexed)
    char_start: int  # start offset in body
    char_end: int  # end offset in body
    context_text: str  # ~250-word window around the marker
    context_start: int  # char offset where context window begins
    section_name: str | None = None  # which section this appears in


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
            )
        )
    return enriched


def process_paper(
    bibcode: str,
    body: str,
    references: list[str],
) -> list[CitationContext]:
    """Extract and resolve citation contexts for a single paper.

    Combines extraction, section enrichment, and resolution.

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

    markers = extract_citation_contexts(body)
    if not markers:
        return []

    sections = parse_sections(body)
    enriched_markers = _enrich_with_sections(markers, sections)

    return resolve_citation_markers(enriched_markers, references, bibcode)


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
