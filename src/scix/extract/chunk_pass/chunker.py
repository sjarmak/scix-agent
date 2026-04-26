"""Pure-function chunker that walks ``papers_fulltext.sections`` in order.

Emits 512-token windows over each section. Consecutive windows in a section
overlap by ``stride`` (=64) tokens — i.e. the start of the next window is
``window - stride`` tokens past the start of the previous one. Chunks NEVER
cross section boundaries. The trailing window in a section is kept iff it
has at least ``stride`` (=64) tokens; smaller tails are dropped.

Each chunk carries the enclosing section's heading, level, and char_offset,
plus a per-paper sequential ``chunk_id`` stable within
``(bibcode, parser_version)`` ordering.

The tokenizer is injected, never imported here at module load — importing
this module must NOT pull in torch or transformers. Production callers pass
a HuggingFace fast tokenizer (``return_offsets_mapping=True`` supported);
tests can pass a stub exposing only ``.encode(text, add_special_tokens=False)``
and the fallback whitespace offset helper kicks in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 64


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One tokenizer window over a single section.

    Frozen for immutability (see ``rules/common/coding-style.md``). The
    ``chunk_id`` is sequential per paper across all sections in input order.
    """

    bibcode: str
    section_idx: int
    section_heading: str
    section_level: int
    char_offset: int
    chunk_id: int
    n_tokens: int
    text: str


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _tokenize_with_offsets(
    tokenizer: Any, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return ``(token_ids, offsets)`` for ``text``.

    Strategy (in order):

    1. HuggingFace fast tokenizer — call
       ``tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)``
       and unpack ``input_ids`` + ``offset_mapping``.
    2. Stub providing ``encode_with_offsets`` — returns ``(ids, offsets)``.
    3. Stub providing only ``encode`` — fall back to whitespace tokenization
       to derive offsets, while still using ``encode`` for token ids so the
       caller's stub controls ``n_tokens``.

    The two outputs are guaranteed to have the same length.
    """
    # Strategy 1: HuggingFace fast tokenizer call signature.
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = list(encoded["input_ids"])
        offsets = [tuple(o) for o in encoded["offset_mapping"]]
        if len(ids) == len(offsets):
            return ids, offsets
    except (TypeError, KeyError, AttributeError):
        pass

    # Strategy 2: explicit helper on a stub.
    if hasattr(tokenizer, "encode_with_offsets"):
        ids, offsets = tokenizer.encode_with_offsets(text)
        return list(ids), [tuple(o) for o in offsets]

    # Strategy 3: ``encode`` plus whitespace-derived offsets.
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    ws_offsets = _whitespace_offsets(text)
    if len(ws_offsets) != len(ids):
        # The stub's token count disagrees with whitespace tokenization; this
        # is fine for synthetic tests — pad/truncate offsets to align so we
        # never index out of range. Logged at DEBUG to keep test noise low.
        logger.debug(
            "tokenizer/whitespace token count mismatch (ids=%d, ws=%d); aligning",
            len(ids),
            len(ws_offsets),
        )
        if len(ws_offsets) < len(ids):
            tail = ws_offsets[-1] if ws_offsets else (0, 0)
            ws_offsets = ws_offsets + [tail] * (len(ids) - len(ws_offsets))
        else:
            ws_offsets = ws_offsets[: len(ids)]
    return ids, ws_offsets


def _whitespace_offsets(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` char offsets for each whitespace-delimited word."""
    offsets: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        # Skip whitespace.
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        offsets.append((start, i))
    return offsets


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def iter_chunks(
    bibcode: str,
    sections: list[dict],
    tokenizer: Any,
    *,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
) -> Iterator[Chunk]:
    """Yield :class:`Chunk` objects for one paper.

    Parameters
    ----------
    bibcode:
        Paper identifier; copied onto each emitted chunk.
    sections:
        Ordered list of section dicts with keys ``heading``, ``level``,
        ``text``, ``offset`` (character offset of the section within the
        full body).
    tokenizer:
        Any object satisfying :func:`_tokenize_with_offsets` (see module
        docstring).
    window:
        Window size in tokens. Default ``512``.
    stride:
        Token overlap between consecutive windows. Default ``64``. The step
        between window starts is ``window - stride``. Also used as the
        minimum-tail threshold: trailing windows shorter than ``stride``
        tokens are dropped.

    Yields
    ------
    Chunk
        One per kept window. ``chunk_id`` is sequential ``0..N-1`` across
        the entire paper in section-then-window order. Chunks NEVER cross
        section boundaries.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride >= window:
        raise ValueError(f"stride ({stride}) must be < window ({window})")

    step = window - stride

    chunk_id = 0
    for section_idx, section in enumerate(sections):
        text = section.get("text") or ""
        if not text:
            continue

        ids, offsets = _tokenize_with_offsets(tokenizer, text)
        n = len(ids)
        if n < stride:
            # Section too short to yield any kept window.
            continue

        section_offset = int(section.get("offset", 0))
        heading = str(section.get("heading", ""))
        level = int(section.get("level", 0))

        t_start = 0
        while t_start < n:
            t_end = min(t_start + window, n)
            n_tokens = t_end - t_start

            if n_tokens < stride:
                # Trailing window too small — drop and stop walking this section.
                break

            char_start = offsets[t_start][0]
            char_end = offsets[t_end - 1][1]
            chunk_text = text[char_start:char_end].strip()
            char_offset = section_offset + char_start

            yield Chunk(
                bibcode=bibcode,
                section_idx=section_idx,
                section_heading=heading,
                section_level=level,
                char_offset=char_offset,
                chunk_id=chunk_id,
                n_tokens=n_tokens,
                text=chunk_text,
            )
            chunk_id += 1

            if t_end == n:
                # Reached end of section — done with this section.
                break
            t_start += step
