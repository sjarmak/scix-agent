"""Citation context extraction pipeline.

Extracts ~250-word context windows around inline citation markers [N] in paper
body text, resolves numbered markers to target bibcodes via the paper's
reference[] array, and stores results in the citation_contexts table.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import psycopg

from scix.db import IngestLog, get_connection
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


_WORD_RE = re.compile(r"\S+")


# Hard cap on body length passed to the public extractors. _word_offsets builds
# two list[int]s sized to the token count of the body — at ~8 chars/token, a
# 100 MB body materialises ~12.5M ints and ~200 MB peak RAM, which OOM-kills
# the shard worker even under scix-batch's MemoryMax. Truncate at the entry of
# the public functions so the rest of the module can assume bounded input. The
# 10M-char cap is well above the body-size distribution we actually see in
# papers_fulltext (P99.9 ≈ 1.5M chars per spot-checks) so legitimate bodies
# pass through untouched. Surfaced from wave-k6w0-u5gz-i315-3ozn1 review (bead
# scix_experiments-2wbx).
_MAX_BODY_CHARS: int = 10_000_000


def _word_offsets(text: str) -> tuple[list[int], list[int]]:
    """Pre-compute per-word char offsets in ``text``.

    Returns parallel sorted lists ``(starts, ends)`` of half-open char ranges
    for each whitespace-delimited token, equivalent to ``text.split()`` token
    positions. Computed once per body so callers can resolve word-window
    boundaries via O(log N) bisect rather than re-splitting an O(L) prefix
    on every match.
    """
    starts: list[int] = []
    ends: list[int] = []
    for m in _WORD_RE.finditer(text):
        starts.append(m.start())
        ends.append(m.end())
    return starts, ends


def _word_boundary_window(
    text: str,
    char_start: int,
    char_end: int,
    words: int = 125,
    *,
    word_starts: list[int] | None = None,
    word_ends: list[int] | None = None,
) -> tuple[int, int]:
    """Find a ~words-before and ~words-after window around a span.

    Returns (window_start, window_end) as char offsets into text.

    ``word_starts`` and ``word_ends`` are optional pre-computed parallel
    lists from :func:`_word_offsets` over ``text``. When the same body is
    queried for many citation spans, reusing the pre-computed arrays makes
    each lookup O(log N) instead of O(L) per call. When omitted, the
    arrays are computed inline (caller pays O(L) once).
    """
    if word_starts is None or word_ends is None:
        word_starts, word_ends = _word_offsets(text)

    # Words strictly before char_start: those whose start offset < char_start.
    # bisect_left returns the first index with value >= char_start, which is
    # exactly the count of words before the span.
    before_count = bisect.bisect_left(word_starts, char_start)
    if before_count <= words:
        window_start = 0
    else:
        window_start = word_starts[before_count - words]

    # Words at or after char_end: those whose start offset >= char_end.
    after_first = bisect.bisect_left(word_starts, char_end)
    after_count = len(word_starts) - after_first
    if after_count <= words:
        window_end = len(text)
    else:
        # End of the (words)th word after the span. word_ends is parallel
        # to word_starts, so index `after_first + words - 1` is the end of
        # the words-th token at or after char_end.
        window_end = word_ends[after_first + words - 1]

    return window_start, window_end


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------


def extract_citation_contexts(body: str) -> list[CitationMarker]:
    """Find [N] patterns in text and return ~250-word context windows.

    Parameters
    ----------
    body : str
        Plain-text body of a paper. Truncated at :data:`_MAX_BODY_CHARS`
        with a single warning log if exceeded — see the constant's
        rationale for why.

    Returns
    -------
    list[CitationMarker]
        One entry per citation marker found, with context window and offsets.
    """
    if not body:
        return []
    if len(body) > _MAX_BODY_CHARS:
        logger.warning(
            "extract_citation_contexts: body length %d exceeds cap %d; truncating",
            len(body),
            _MAX_BODY_CHARS,
        )
        body = body[:_MAX_BODY_CHARS]

    word_starts, word_ends = _word_offsets(body)
    markers: list[CitationMarker] = []
    for m in _CITATION_RE.finditer(body):
        inner = m.group(1)
        nums = _parse_marker_numbers(inner)
        if not nums:
            continue

        char_start = m.start()
        char_end = m.end()
        win_start, win_end = _word_boundary_window(
            body, char_start, char_end, word_starts=word_starts, word_ends=word_ends
        )
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
    rf"\b({_SURNAME})" rf"(?:\s+(?:and|&)\s+(?:{_SURNAME}))?" rf"\s*\(\s*{_YEAR}\s*\)"
)

# Pattern B — "Surname et al. YYYY" / "Surname et al., YYYY".
# Whitespace and parens are grouped so trailing space is consumed only when a
# closing paren follows — otherwise m.end() inflates past the year.
_AY_ET_AL = re.compile(
    rf"\b({_SURNAME})\s+et\s+al\.?,?\s*(?:\(\s*)?{_YEAR}(?:\s*\))?"
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

    Overlapping matches across the four patterns are de-duplicated by
    half-open interval overlap: once a match is accepted, any later match
    whose ``[char_start, char_end)`` range intersects it is rejected. The
    pattern iteration order (et-al → narrative → paren → sub-cite) decides
    which match wins on conflict.

    Body length is capped at :data:`_MAX_BODY_CHARS` with a warning log
    when truncation fires.
    """
    if not body:
        return []
    if len(body) > _MAX_BODY_CHARS:
        logger.warning(
            "extract_author_year_citations: body length %d exceeds cap %d; truncating",
            len(body),
            _MAX_BODY_CHARS,
        )
        body = body[:_MAX_BODY_CHARS]

    # Pre-compute word-start/word-end offsets once per body so each
    # _word_boundary_window call is O(log N) via bisect rather than
    # re-splitting an O(L) prefix per match — was ~75% of cumtime on
    # review-paper-scale bodies.
    word_starts, word_ends = _word_offsets(body)

    # Accepted spans form a disjoint, sorted-by-start interval set (the
    # overlap rejection below maintains that invariant). We keep parallel
    # sorted lists so each overlap check is O(log N) via bisect rather than
    # an O(N) linear scan over a tuple list — the linear form was 162ms at
    # n=3000 spans and would blow up on a degenerate review-paper body.
    accepted_starts: list[int] = []
    accepted_ends: list[int] = []
    out: list[CitationMarker] = []

    def _overlaps(start: int, end: int) -> bool:
        # Half-open intervals: [a, b) overlaps [s, e) iff a < e and b > s.
        # With disjoint sorted intervals, only the two neighbors matter:
        #   - left neighbor (largest start <= start): overlap iff its end > start
        #   - right neighbor (smallest start > start): overlap iff its start < end
        i = bisect.bisect_right(accepted_starts, start)
        if i > 0 and accepted_ends[i - 1] > start:
            return True
        if i < len(accepted_starts) and accepted_starts[i] < end:
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

            win_start, win_end = _word_boundary_window(
                body,
                char_start,
                char_end,
                word_starts=word_starts,
                word_ends=word_ends,
            )
            context = body[win_start:win_end]

            # Insert into both lists at the same index so they stay
            # aligned and accepted_starts remains sorted for the next
            # bisect lookup.
            i = bisect.bisect_right(accepted_starts, char_start)
            accepted_starts.insert(i, char_start)
            accepted_ends.insert(i, char_end)
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

_SELECT_PAPERS_BASE = """
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


def _build_papers_select(
    shard: tuple[int, int] | None,
    limit: int | None,
    *,
    oa_only: bool = True,
) -> tuple[str, list[Any]]:
    """Compose the streaming SELECT for the extraction pipeline.

    Optionally appends a ``mod(hashtext(p.bibcode), n) = i`` predicate so
    multiple worker processes can carve up the eligible-paper population
    without locking. Validates ``shard`` defensively even though the CLI
    parses it: callers that bypass the CLI (tests, future schedulers)
    must still get an error on bad input rather than a silently-empty
    result set.

    ``oa_only`` (default True) appends ``papers_is_oa_or_preprint(p)`` to
    the WHERE clause — the OA/preprint gate from migration 068. Set to
    False (CLI: ``--include-closed``) to process closed-access papers as
    well, with explicit operator approval.
    """
    sql = _SELECT_PAPERS_BASE
    if oa_only:
        sql = sql + "      AND papers_is_oa_or_preprint(p)\n"
    params: list[Any] = []
    if shard is not None:
        index, total = shard
        if total <= 0 or index < 0 or index >= total:
            raise ValueError(
                f"invalid shard spec ({index}/{total}); require 0 <= index < total and total > 0"
            )
        sql = sql + "      AND mod(hashtext(p.bibcode), %s) = %s\n"
        params.extend([total, index])
    if limit is not None:
        sql = sql + " LIMIT %s"
        params.append(limit)
    return sql, params


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
    *,
    shard: tuple[int, int] | None = None,
    ingest_log_filename: str | None = None,
    oa_only: bool = True,
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
    shard : tuple[int, int] | None
        ``(index, total)`` shard spec. Restricts the SELECT to papers
        where ``mod(hashtext(bibcode), total) = index`` so independent
        worker processes can run in parallel without locking.
    ingest_log_filename : str | None
        Logical filename to record progress under in ``ingest_log``. When
        set, the pipeline writes a ``status='in_progress'`` row at start,
        updates counts at each checkpoint, and writes ``status='complete'``
        on success or ``status='failed'`` if the pipeline raises.

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

    log: IngestLog | None = None
    if ingest_log_filename is not None:
        log = IngestLog(write_conn)
        log.start(ingest_log_filename)

    pipeline_failed = False

    try:
        query, params = _build_papers_select(
            shard=shard, limit=limit, oa_only=oa_only
        )

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
                    if log is not None and ingest_log_filename is not None:
                        log.update_counts(
                            ingest_log_filename,
                            records=papers_processed,
                            errors=0,
                            edges=total_inserted,
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

    except BaseException:
        pipeline_failed = True
        raise
    finally:
        if log is not None and ingest_log_filename is not None:
            try:
                log.update_counts(
                    ingest_log_filename,
                    records=papers_processed,
                    errors=0,
                    edges=total_inserted,
                )
                if pipeline_failed:
                    log.mark_failed(ingest_log_filename)
                else:
                    log.finish(ingest_log_filename)
            except Exception:  # noqa: BLE001 — bookkeeping must never mask the real error
                logger.exception("ingest_log finalize for %s failed", ingest_log_filename)
        read_conn.close()
        write_conn.close()

    return total_inserted
