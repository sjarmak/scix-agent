"""Surface-form normalization for entity canonical names.

Pre-canonicalization step that strips markup artifacts and Unicode
variants leaking through from upstream HTML/LaTeX/PDF text. Sits before
``canonicalize()`` in ``ner_pass`` so two GLiNER mentions that differ
only in tag wrapping (``co<sub>2</sub>`` vs ``co2``), HTML entity
encoding (``ar&amp;d`` vs ``ar&d``), or Unicode dash variant
(``iron–59`` vs ``iron-59``) collapse into a single ``entities`` row
instead of fragmenting the canonical space.

Intentionally conservative: only strips/maps are applied, never any
semantic substitution. ``co<sub>2</sub>`` becomes ``co2``; mapping
``co2`` to ``carbon dioxide`` is a chemistry-knowledge step and is out
of scope here (see follow-up bead in scix_experiments-06xc notes).
"""

from __future__ import annotations

import re
import unicodedata

# Inline-formatting tags that frequently leak through ar5iv/ADS HTML scrapes.
# Stripped open- and close-tag wholesale; their text content is preserved
# (e.g. ``co<sub>2</sub>`` -> ``co2``).
_INLINE_TAG_RE = re.compile(
    r"</?(?:sub|sup|i|b|em|strong|tt|small|span|p|br|u|s|big)\b[^>]*>",
    re.IGNORECASE,
)

# Numeric / hex character references (e.g. ``&#946;`` for greek beta,
# ``&#x3B2;`` likewise).
_ENTITY_NUM_RE = re.compile(r"&#(\d+);")
_ENTITY_HEX_RE = re.compile(r"&#x([0-9a-fA-F]+);")

# Named HTML entities the corpus has been observed to contain. Limited to
# punctuation/whitespace mappings so we never accidentally turn a Greek
# letter named entity into an ASCII look-alike (those should reach NFKC).
_NAMED_ENTITIES: dict[str, str] = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "ensp": " ",
    "emsp": " ",
    "thinsp": " ",
    "ndash": "-",
    "mdash": "-",
    "minus": "-",
    "hyphen": "-",
    "lsquo": "'",
    "rsquo": "'",
    "ldquo": '"',
    "rdquo": '"',
    "sbquo": "'",
    "bdquo": '"',
    "prime": "'",
    "Prime": '"',
    "lsaquo": "<",
    "rsaquo": ">",
    "laquo": '"',
    "raquo": '"',
    "shy": "",
    "zwnj": "",
    "zwj": "",
}

_NAMED_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]*);")

# Unicode -> ASCII fold for punctuation that NFKC leaves alone but that
# fragments canonicals (the en/em-dash family is the worst offender,
# accounting for ~10% of chemistry mentions in our corpus).
#
# Stored as explicit ord() codepoints to keep the source file
# round-trip safe through tooling that may strip or re-encode literal
# wide characters.
_PUNCT_FOLD: dict[int, str] = {
    # Dash family
    0x2010: "-",  # HYPHEN
    0x2011: "-",  # NB HYPHEN
    0x2012: "-",  # FIGURE DASH
    0x2013: "-",  # EN DASH
    0x2014: "-",  # EM DASH
    0x2015: "-",  # HORIZONTAL BAR
    0x2212: "-",  # MINUS SIGN
    # Quote family
    0x2018: "'",  # LEFT SINGLE QUOTATION MARK
    0x2019: "'",  # RIGHT SINGLE QUOTATION MARK
    0x201A: "'",  # SINGLE LOW-9 QUOTATION MARK
    0x201B: "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    0x201C: '"',  # LEFT DOUBLE QUOTATION MARK
    0x201D: '"',  # RIGHT DOUBLE QUOTATION MARK
    0x201E: '"',  # DOUBLE LOW-9 QUOTATION MARK
    0x201F: '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    0x2032: "'",  # PRIME
    0x2033: '"',  # DOUBLE PRIME
    # Whitespace block - fold to ASCII space; canonicalize() collapses runs.
    0x00A0: " ",  # NO-BREAK SPACE
    0x2000: " ",  # EN QUAD
    0x2001: " ",  # EM QUAD
    0x2002: " ",  # EN SPACE
    0x2003: " ",  # EM SPACE
    0x2004: " ",  # THREE-PER-EM SPACE
    0x2005: " ",  # FOUR-PER-EM SPACE
    0x2006: " ",  # SIX-PER-EM SPACE
    0x2007: " ",  # FIGURE SPACE
    0x2008: " ",  # PUNCTUATION SPACE
    0x2009: " ",  # THIN SPACE
    0x200A: " ",  # HAIR SPACE
    0x202F: " ",  # NARROW NO-BREAK SPACE
    0x205F: " ",  # MEDIUM MATHEMATICAL SPACE
    0x3000: " ",  # IDEOGRAPHIC SPACE
    # Zero-width / format chars - drop entirely.
    0x200B: "",  # ZERO WIDTH SPACE
    0x200C: "",  # ZERO WIDTH NON-JOINER
    0x200D: "",  # ZERO WIDTH JOINER
    0xFEFF: "",  # ZERO WIDTH NO-BREAK SPACE / BOM
    0x00AD: "",  # SOFT HYPHEN
}


def _decode_html_entities(s: str) -> str:
    s = _ENTITY_NUM_RE.sub(lambda m: chr(int(m.group(1))), s)
    s = _ENTITY_HEX_RE.sub(lambda m: chr(int(m.group(1), 16)), s)
    s = _NAMED_ENTITY_RE.sub(
        lambda m: _NAMED_ENTITIES.get(m.group(1), m.group(0)),
        s,
    )
    return s


def normalize_surface(s: str) -> str:
    """Strip markup, decode entities, and fold Unicode variants.

    Runs before ``canonicalize()`` (which lowercases + collapses
    whitespace). Pure string transformation: no DB lookups, no
    semantic substitution.

    Steps:
        1. Strip inline HTML/XML formatting tags (``<sub>``, ``<sup>``,
           ``<i>``, etc.). Content is preserved.
        2. Decode named/numeric/hex HTML entities (``&amp;`` -> ``&``,
           ``&#946;`` -> greek beta).
        3. Apply NFKC. This collapses fullwidth/halfwidth, decomposes
           compatibility ligatures, and pulls subscript / superscript
           digits down to ASCII (``₂`` -> ``2``).
        4. Fold the dash / quote / Unicode-space families to ASCII.
           NFKC alone does not handle these.
    """
    s = _INLINE_TAG_RE.sub("", s)
    s = _decode_html_entities(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_PUNCT_FOLD)
    return s
