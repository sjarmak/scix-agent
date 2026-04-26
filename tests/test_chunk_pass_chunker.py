"""Tests for the section-aware sliding-window chunker.

Covers all 10 acceptance criteria from the PRD work unit "chunker":

1. Package init exists.
2. ``Chunk`` is a frozen dataclass with the documented fields.
3. ``iter_chunks`` signature matches the spec.
4. Chunks never cross section boundaries.
5. Trailing windows shorter than ``stride`` are dropped; trailing windows
   ``>= stride`` are kept.
6. ``chunk_id`` is sequential per paper across all sections.
7. ``char_offset`` accounts for the section's offset and is monotonically
   non-decreasing within a section.
8. The chunker module loads without importing torch or transformers.
9. Edge cases — empty sections, empty text, text shorter than ``stride``,
   large sections.
10. The full file passes under pytest.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Stub tokenizer used by all tests
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Minimal stub: each whitespace-separated word is one token.

    Exposes only the ``encode(text, add_special_tokens=False) -> list[int]``
    interface, exercising the chunker's whitespace-offset fallback path.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def _make_text(n_tokens: int) -> str:
    """Return a string of ``n_tokens`` whitespace-separated single-letter words."""
    return " ".join(["w"] * n_tokens)


# ---------------------------------------------------------------------------
# AC1 + AC2 + AC3 — package shape + dataclass + iter_chunks signature
# ---------------------------------------------------------------------------


def test_package_init_importable():
    """AC1: ``scix.extract.chunk_pass`` package init exists and is importable."""
    pkg = importlib.import_module("scix.extract.chunk_pass")
    assert hasattr(pkg, "Chunk")
    assert hasattr(pkg, "iter_chunks")


def test_chunk_dataclass_is_frozen_with_required_fields():
    """AC2: ``Chunk`` is frozen and has the documented fields."""
    from scix.extract.chunk_pass import Chunk

    assert dataclasses.is_dataclass(Chunk)
    # Frozen dataclasses raise FrozenInstanceError on mutation.
    chunk = Chunk(
        bibcode="2024arXiv.0001",
        section_idx=0,
        section_heading="Intro",
        section_level=1,
        char_offset=0,
        chunk_id=0,
        n_tokens=10,
        text="hello world",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.bibcode = "other"  # type: ignore[misc]

    field_names = {f.name for f in dataclasses.fields(Chunk)}
    expected = {
        "bibcode",
        "section_idx",
        "section_heading",
        "section_level",
        "char_offset",
        "chunk_id",
        "n_tokens",
        "text",
    }
    assert field_names == expected


def test_iter_chunks_signature_matches_spec():
    """AC3: ``iter_chunks(bibcode, sections, tokenizer, *, window=512, stride=64)``."""
    from scix.extract.chunk_pass import iter_chunks

    sig = inspect.signature(iter_chunks)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ["bibcode", "sections", "tokenizer", "window", "stride"]

    # window and stride are keyword-only with the documented defaults.
    window_p = sig.parameters["window"]
    stride_p = sig.parameters["stride"]
    assert window_p.kind is inspect.Parameter.KEYWORD_ONLY
    assert stride_p.kind is inspect.Parameter.KEYWORD_ONLY
    assert window_p.default == 512
    assert stride_p.default == 64


# ---------------------------------------------------------------------------
# AC4 — chunks never cross section boundaries
# ---------------------------------------------------------------------------


def test_chunks_never_cross_section_boundary_two_600_token_sections():
    """AC4: two 600-token sections produce exactly 4 chunks (2 per section).

    With window=512 and stride=64 (overlap=64 -> step=448):

    * Section 0: window [0, 512] (512 tokens), trailing [448, 600] (152 tokens).
    * Section 1: same shape.

    No bridging chunk spanning both sections is permitted.
    """
    from scix.extract.chunk_pass import iter_chunks

    sections = [
        {"heading": "S0", "level": 1, "text": _make_text(600), "offset": 0},
        {"heading": "S1", "level": 1, "text": _make_text(600), "offset": 5000},
    ]
    chunks = list(iter_chunks("paper-A", sections, StubTokenizer()))
    assert len(chunks) == 4

    # Every chunk belongs to exactly one section_idx (0 or 1) — none span both.
    by_section = {0: [], 1: []}
    for c in chunks:
        by_section[c.section_idx].append(c)
    assert len(by_section[0]) == 2
    assert len(by_section[1]) == 2


# ---------------------------------------------------------------------------
# AC5 — trailing-window threshold
# ---------------------------------------------------------------------------


def test_540_token_section_yields_two_chunks_with_trailing_kept():
    """AC5 (keep): 540 tokens -> [0,512] full + [448,540]=92 tokens (>=64, kept)."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": _make_text(540), "offset": 0}]
    chunks = list(iter_chunks("paper-B", sections, StubTokenizer()))
    assert len(chunks) == 2
    assert chunks[0].n_tokens == 512
    assert chunks[1].n_tokens == 92  # 540 - 448 = 92, >= 64 -> kept


def test_section_below_stride_drops_entirely():
    """AC5 (drop): a section shorter than ``stride`` tokens yields 0 chunks.

    The "last window kept iff >= stride tokens" rule means that any section
    too short to produce even one valid window is skipped entirely. A
    63-token section (one below stride=64) is dropped.
    """
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": _make_text(63), "offset": 0}]
    chunks = list(iter_chunks("paper-C", sections, StubTokenizer()))
    assert chunks == []


def test_trailing_window_just_above_stride_is_kept():
    """A trailing window with stride+1 tokens is kept (above threshold)."""
    from scix.extract.chunk_pass import iter_chunks

    # window=4, stride=2, step=2. n=7: [0,4], [2,6]=full (t_end=6!=7),
    # [4,7]=3 tokens (>=2 -> kept). 3 chunks.
    sections = [{"heading": "S0", "level": 1, "text": _make_text(7), "offset": 0}]
    chunks = list(iter_chunks(
        "paper-D2", sections, StubTokenizer(), window=4, stride=2
    ))
    assert len(chunks) == 3
    assert chunks[-1].n_tokens == 3


def test_short_section_just_above_stride_yields_one_chunk():
    """A section just above ``stride`` tokens but below ``window`` produces
    a single window with ``n_tokens == n``."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": _make_text(65), "offset": 0}]
    chunks = list(iter_chunks("paper-D3", sections, StubTokenizer()))
    assert len(chunks) == 1
    assert chunks[0].n_tokens == 65


# ---------------------------------------------------------------------------
# AC6 — sequential chunk_id per paper
# ---------------------------------------------------------------------------


def test_chunk_ids_are_sequential_across_sections():
    """AC6: chunk_id runs 0..N-1 across all chunks in section order."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [
        {"heading": "S0", "level": 1, "text": _make_text(600), "offset": 0},
        {"heading": "S1", "level": 1, "text": _make_text(600), "offset": 5000},
        {"heading": "S2", "level": 1, "text": _make_text(600), "offset": 10000},
    ]
    chunks = list(iter_chunks("paper-E", sections, StubTokenizer()))
    assert len(chunks) == 6
    assert [c.chunk_id for c in chunks] == [0, 1, 2, 3, 4, 5]
    # And the section_idx sequence respects input order.
    assert [c.section_idx for c in chunks] == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# AC7 — char_offset = section.offset + token_start_char_within_section
# ---------------------------------------------------------------------------


def test_char_offsets_are_section_relative_and_monotonic_within_section():
    """AC7: char_offset = section.offset + char-of-first-token-in-window;
    char_offset is monotonically non-decreasing within a single section."""
    from scix.extract.chunk_pass import iter_chunks

    section_offset = 12345
    sections = [
        {"heading": "S0", "level": 1, "text": _make_text(600), "offset": section_offset}
    ]
    chunks = list(iter_chunks("paper-F", sections, StubTokenizer()))
    assert len(chunks) == 2
    # First chunk starts at section offset (token 0 -> char 0 within section).
    assert chunks[0].char_offset == section_offset
    # Second chunk starts at section_offset + char-of-token-448.
    # With "w " repetition each token is 1 char + 1 space = 2 chars; token i
    # starts at char 2*i. So token 448 starts at char 896.
    assert chunks[1].char_offset == section_offset + 2 * 448

    # Monotonic non-decreasing within the section.
    offsets = [c.char_offset for c in chunks]
    assert offsets == sorted(offsets)


def test_char_offsets_across_sections_use_section_offset():
    """char_offset jumps by section.offset, never crossing into another section's text."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [
        {"heading": "S0", "level": 1, "text": _make_text(600), "offset": 0},
        {"heading": "S1", "level": 1, "text": _make_text(600), "offset": 9999},
    ]
    chunks = list(iter_chunks("paper-G", sections, StubTokenizer()))
    s1_chunks = [c for c in chunks if c.section_idx == 1]
    # Every section-1 chunk has char_offset >= 9999 (the section's start).
    for c in s1_chunks:
        assert c.char_offset >= 9999


# ---------------------------------------------------------------------------
# AC8 — lazy tokenizer load (no torch / transformers at module import)
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_in_torch_or_transformers():
    """AC8: importing the chunker must NOT import torch or transformers.

    Run in a fresh Python subprocess to get a clean ``sys.modules``.
    """
    import os
    import pathlib

    # Locate the worktree's ``src`` directory so the subprocess imports the
    # in-tree package (not any older site-packages copy).
    src_dir = pathlib.Path(__file__).resolve().parent.parent / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing}" if existing else str(src_dir)
    )

    code = textwrap.dedent(
        """
        import sys
        import scix.extract.chunk_pass.chunker  # noqa: F401
        bad = [m for m in ('torch', 'transformers') if m in sys.modules]
        if bad:
            raise SystemExit(f'imported: {bad}')
        """
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"chunker import pulled in heavy deps: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# AC9 — edge cases
# ---------------------------------------------------------------------------


def test_empty_sections_list_yields_no_chunks():
    """AC9 (empty list)."""
    from scix.extract.chunk_pass import iter_chunks

    chunks = list(iter_chunks("paper-H", [], StubTokenizer()))
    assert chunks == []


def test_section_with_empty_text_yields_no_chunks():
    """AC9 (empty text)."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": "", "offset": 0}]
    chunks = list(iter_chunks("paper-I", sections, StubTokenizer()))
    assert chunks == []


def test_section_with_text_too_short_for_min_window_yields_no_chunks():
    """AC9 (text < stride tokens => 0 chunks)."""
    from scix.extract.chunk_pass import iter_chunks

    # 32 tokens < stride=64 -> dropped entirely.
    sections = [{"heading": "S0", "level": 1, "text": _make_text(32), "offset": 0}]
    chunks = list(iter_chunks("paper-J", sections, StubTokenizer()))
    assert chunks == []


def test_section_with_exactly_stride_tokens_yields_one_chunk():
    """A section with exactly ``stride`` tokens passes the >=stride threshold."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": _make_text(64), "offset": 0}]
    chunks = list(iter_chunks("paper-K", sections, StubTokenizer()))
    assert len(chunks) == 1
    assert chunks[0].n_tokens == 64


def test_large_section_produces_correct_chunk_count():
    """AC9 (large section). With step=window-stride=448, a section of length
    ``n`` yields chunks at starts 0, 448, 896, ... while the window has
    ``>= stride`` tokens.

    For n=2048: starts at 0 [0,512], 448 [448,960], 896 [896,1408],
    1344 [1344,1856], 1792 [1792,2048]=256. 256>=64 -> kept. Total: 5.
    """
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": _make_text(2048), "offset": 0}]
    chunks = list(iter_chunks("paper-L", sections, StubTokenizer()))
    assert len(chunks) == 5
    assert [c.n_tokens for c in chunks] == [512, 512, 512, 512, 256]


# ---------------------------------------------------------------------------
# AC2 (additional) — Chunk text is a substring of the section text
# ---------------------------------------------------------------------------


def test_chunk_text_is_substring_of_section_text():
    """Each chunk's text is the substring of its section's text spanning
    the window's tokens."""
    from scix.extract.chunk_pass import iter_chunks

    section_text = _make_text(600)
    sections = [{"heading": "S0", "level": 1, "text": section_text, "offset": 0}]
    chunks = list(iter_chunks("paper-M", sections, StubTokenizer()))
    for c in chunks:
        assert c.text in section_text


# ---------------------------------------------------------------------------
# Validation of arguments
# ---------------------------------------------------------------------------


def test_invalid_window_or_stride_raise_value_error():
    """Defensive: nonsensical window/stride combinations raise."""
    from scix.extract.chunk_pass import iter_chunks

    sections = [{"heading": "S0", "level": 1, "text": _make_text(100), "offset": 0}]
    with pytest.raises(ValueError):
        list(iter_chunks("p", sections, StubTokenizer(), window=0))
    with pytest.raises(ValueError):
        list(iter_chunks("p", sections, StubTokenizer(), stride=0))
    with pytest.raises(ValueError):
        list(iter_chunks("p", sections, StubTokenizer(), window=64, stride=64))
