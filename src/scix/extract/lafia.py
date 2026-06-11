"""Lafia-style informal software/dataset reference detector (bead dbl.18).

GLiNER (``scix.extract.ner_pass``) tags *named* mentions: it sees a token that
looks like a software or dataset name and labels it. It systematically misses
mentions where the name itself is unremarkable but the *surrounding context*
makes the reference unambiguous — e.g. "implemented in TOPCAT", "data from the
LAMOST survey", "the emcee package". This module adopts the cue-phrase approach
of Lafia et al. 2022 (arXiv:2205.11651) and the informal-mention literature
(SciER arXiv:2410.21155, Falcon-7b SOMD arXiv:2405.08514) to recover those
context-disambiguated mentions.

Design contract
---------------
A mention is emitted iff BOTH:

  (a) a curated *cue* fires — a verb/preposition phrase ("implemented in",
      "data from") that precedes a name, or a head-noun ("package", "survey")
      that follows one; and

  (b) the adjacent token run validates as a *name* — capitalised, an acronym,
      or carrying an internal digit / punctuation (``scikit-learn``, ``DR16``),
      and not a common English word.

Cue strength sets the base confidence; ``--min-confidence`` on the batch driver
filters the weak generic cues out by default. The two-stage split (mechanical
regex anchor, then directional name extraction in Python) keeps the matcher
robust to greedy capture and keeps every decision unit-testable.

ZFC note: this is a mechanical pattern detector, not a semantic judge. Cues are
regexes, the name filter is structural token shape, and confidence is a fixed
per-cue constant — no model scoring, no hidden thresholds on meaning.

Provenance
----------
Rows land in ``document_entities`` under a dedicated namespace so they never
collide with or clobber GLiNER's ``(link_type='mentions', tier=4,
match_method='gliner')`` rows:

- ``link_type   = 'informal_mention'``
- ``match_method= 'lafia_informal'``
- ``tier        = 3``  (contextual signal: above lexical keyword/aho, below ML NER)

Entities are upserted under ``source='lafia'`` (cross-source canonicalisation is
handled downstream by ``scix.resolve_entities``), mirroring how each extraction
source owns its own ``entities`` rows.

Public surface
--------------
- ``InformalMention``           — frozen dataclass (one detected mention)
- ``detect_informal_references``— pure detector over body text
- ``insert_mentions``           — upsert entities + document_entities for a paper
- provenance constants ``LINK_TYPE`` / ``MATCH_METHOD`` / ``SOURCE`` /
  ``SOURCE_VERSION`` / ``LAFIA_TIER``
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from psycopg.types.json import Jsonb

if TYPE_CHECKING:  # pragma: no cover — typing-only import
    import psycopg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------

LINK_TYPE: str = "informal_mention"
MATCH_METHOD: str = "lafia_informal"
SOURCE: str = "lafia"
SOURCE_VERSION: str = "lafia/v1"
# Contextual pattern signal. Distinct link_type already isolates these rows
# from every other producer, so the value only needs to order sensibly against
# the global tier ladder (0 lexical-exact, 1 keyword, 2 aho-corasick, 4 ML/NER).
LAFIA_TIER: int = 3

ENTITY_TYPE_SOFTWARE: str = "software"
ENTITY_TYPE_DATASET: str = "dataset"

# Default confidence floor applied by the batch driver. Strong suffix and
# prepositional cues clear it; the generic "we used X" / "via X" cues do not.
DEFAULT_MIN_CONFIDENCE: float = 0.7

# How far past a cue we look for the adjacent name.
_NAME_WINDOW_CHARS: int = 48
# Longest multi-word name we accept ("Sloan Digital Sky Survey" is 4).
_MAX_NAME_TOKENS: int = 4
# Context snippet stored in ``evidence`` for spot-checking precision.
_EVIDENCE_SPAN_CHARS: int = 160

# A name immediately followed by "et al" is an author citation, not a
# software/dataset reference ("data from Cohen et al." / "Cohen, et al."). The
# separator class admits a trailing comma because the span now ends at the
# cleaned core, before any punctuation the raw token carried (dbl.23).
_AUTHOR_CITATION_RE = re.compile(r"[\s,;:.]+et\s+al", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------
# Common words that pass the shape test (often sentence-initial and therefore
# capitalised) but are never software/dataset names. Lower-cased membership.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "this", "that", "these", "those", "our", "their",
        "its", "his", "her", "we", "they", "it", "all", "both", "each",
        "such", "same", "above", "below", "following", "previous", "new",
        "standard", "default", "various", "several", "many", "two", "three",
        "four", "one", "data", "dataset", "method", "methods", "model",
        "models", "result", "results", "analysis", "approach", "approaches",
        "technique", "techniques", "code", "codes", "software", "package",
        "library", "toolkit", "pipeline", "framework", "module", "routine",
        "algorithm", "algorithms", "tool", "tools", "sample", "samples",
        "survey", "catalog", "catalogue", "database", "archive", "simulation",
        "simulations", "version", "figure", "table", "section", "equation",
        "value", "values", "function", "set", "test", "paper", "work",
        "study", "and", "or", "of", "for", "with", "from", "via", "using",
        "used", "use", "which", "what", "where", "when", "i", "ii", "iii",
        # Descriptive / method / quantity terms that recur as the candidate in
        # "X simulation" / "X catalog" / "X survey" but never name a resource on
        # their own. Curated from a real-corpus precision sweep (bead dbl.18);
        # named simulations (IllustrisTNG, EAGLE) survive — only the generic
        # method words are blocked.
        "monte", "carlo", "numerical", "computer", "microlensing", "velocity",
        "galaxy", "galaxies", "luminosity", "wavelength", "multiwavelength",
        "multi-wavelength", "large", "area", "cosmological", "hydrodynamic",
        "hydrodynamical", "spectroscopic", "photometric", "stellar", "magnitude",
    }
)

# Internal characters that mark a token as a plausible name even when it is
# otherwise lower-case (e.g. ``scikit-learn``, ``word2vec``, ``v4.0``).
_NAME_PUNCT: str = "-+./_"
_LEADING_STRIP = " \t\n\r\"'`([{“”‘’"
_TRAILING_STRIP = " \t\n\r\"'`)]}.,;:!?“”‘’"
# Surface punctuation peeled off both ends of a token to reach its name core.
_STRIP = _LEADING_STRIP + _TRAILING_STRIP

# Sentence/clause-terminating marks. A name run must not be extended across one:
# 'implemented in TOPCAT. See docs' otherwise yields 'TOPCAT See' (bead 23w).
_SENTENCE_PUNCT: str = ".!?;:"
# Closing wrappers that may sit AFTER the terminator ('GADGET).', 'emcee."'), so
# the run-boundary test strips them before inspecting the final character.
_SENTENCE_WRAPPERS: str = " \t\n\r\"'`)]}“”‘’"

# A real software/dataset name is ASCII and uses only this charset. Scanned
# 19th-century ADS bodies produce OCR garbage ("Obf~", "i8óO", "3,7165") that
# would otherwise slip through ``X Catalog`` / ``X Library`` cues; this rejects
# it mechanically (no semantic judgement — just a charset gate).
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]*\Z")

# ADS digitization / institutional provenance that recurs in scanned-page
# headers and footers (e.g. the "John G. Wolbach Library" stamp on every
# Harvard plate). These are never the software/dataset being referenced, so a
# cue like "... Library" must not promote them. Matched per-token, lower-cased.
_DENY_TOKENS: frozenset[str] = frozenset(
    {
        "wolbach", "harvard", "observatory", "college", "university",
        "library", "institute", "department", "microfilmed", "sternwarte",
        "smithsonian", "nasa", "ads",
    }
)

# Leading generic descriptive adjectives (bead dbl.21). These DESCRIBE a
# resource ("high-quality dataset", "open-source code") but never name one. The
# residual false positives after GLiNER type-confirmation (bead dbl.20) are
# dominated by such an adjective being captured as the candidate: hyphenated
# compounds slip past ``_is_namey_token`` via the internal-punctuation rule, and
# capitalised participles slip past via the leading-capital rule, so GLiNER then
# legitimately confirms the full noun phrase and cannot reject them.
#
# This is a SPAN-BOUNDARY fix, not a name-token stopword: a leading adjective is
# trimmed off the *front* of a candidate run before it is normalised, so the
# span starts at the real name ("high-resolution VLA" -> "VLA") or, when the
# adjective was the whole run ("high-quality dataset"), no candidate is emitted.
# It is deliberately restricted to generic quality / resolution / scale /
# availability / processing descriptors that never lead a resource name; domain
# compounds that can be part of one (e.g. "high-mass", "X-ray") are excluded.
_LEADING_ADJECTIVES: frozenset[str] = frozenset(
    {
        # quality / fidelity
        "high-quality", "low-quality", "high-fidelity", "high-precision",
        # resolution / dimensionality
        "high-resolution", "low-resolution", "high-dimensional",
        "low-dimensional", "high-contrast",
        # performance / throughput
        "high-performance", "high-throughput", "high-speed", "high-level",
        "low-level", "low-cost", "general-purpose", "special-purpose",
        # scale
        "large-scale", "small-scale", "full-scale", "web-scale",
        # availability / openness
        "open-source", "closed-source", "open-access", "publicly-available",
        "freely-available", "ready-to-use", "easy-to-use",
        # maturity / familiarity
        "state-of-the-art", "well-known", "well-studied", "well-tested",
        "well-defined", "well-established", "well-calibrated", "well-sampled",
        "up-to-date", "real-time", "real-world",
        # granularity
        "fine-grained", "coarse-grained", "ground-truth",
        # single-word processing / quality descriptors (namey only when
        # sentence-initial-capitalised; never a standalone resource name)
        "standardized", "standardised", "normalized", "normalised",
        "augmented", "synthetic", "annotated", "curated", "preprocessed",
        "pre-processed", "labeled", "labelled", "unlabeled", "unlabelled",
        "balanced", "cleaned",
    }
)


def _is_leading_adjective(token: str) -> bool:
    """True if ``token`` is a generic descriptive adjective (span-boundary trim)."""
    core = token.strip(_STRIP)
    return core.lower() in _LEADING_ADJECTIVES


def _ends_sentence(token: str) -> bool:
    """True if ``token``'s raw form terminates a sentence/clause (bead 23w).

    The marker is a sentence-terminating mark (``.!?;:``) as the final character
    of the raw token, possibly behind a closing quote or bracket (``TOPCAT.``,
    ``(GADGET).``, ``emcee."``). Internal dots in versions or paths (``v4.0``,
    ``healpy.io``) are not terminal once wrappers are stripped, so they do not
    count — only a boundary that a name run must never cross.
    """
    stripped = token.rstrip(_SENTENCE_WRAPPERS)
    return bool(stripped) and stripped[-1] in _SENTENCE_PUNCT


def _is_namey_token(token: str) -> bool:
    """Return True if ``token`` has the surface shape of a proper name."""
    core = token.strip(_STRIP)
    if len(core) < 2:
        return False
    if not core.isascii() or _TOKEN_RE.match(core) is None:
        return False  # OCR garbage / non-name punctuation
    low = core.lower()
    if low in _STOPWORDS or low in _DENY_TOKENS:
        return False
    if not any(ch.isalpha() for ch in core):  # pure number / section index
        return False
    if core[0].isupper():  # Capitalised or ALLCAPS
        return True
    if any(ch.isdigit() for ch in core):  # word2vec, DR16, v4
        return True
    if any(p in core[1:] for p in _NAME_PUNCT):  # scikit-learn, healpy.io
        return True
    return False


def _validate_cores(cores: list[str]) -> str | None:
    """Validate a run of already-stripped name cores as a single name.

    Returns the canonical surface (cores joined by single spaces) or ``None``
    if the run does not validate. The single source of truth for "does this
    token run name a software/dataset" — both raw tokens (via
    :func:`_normalise_name`) and offset-tracked cores (via :func:`_clean_run`)
    funnel through here, so surface and span can never disagree on what counts.
    """
    if not cores or len(cores) > _MAX_NAME_TOKENS:
        return None
    if all(not _is_namey_token(t) for t in cores):
        return None
    if any(t.lower() in _DENY_TOKENS for t in cores):
        return None
    surface = " ".join(cores)
    if surface.lower() in _STOPWORDS:
        return None
    if sum(ch.isalpha() for ch in surface) < 2:  # reject pure numbers/symbols
        return None
    return surface


def _normalise_name(tokens: list[str]) -> str | None:
    """Trim surface punctuation off raw tokens and validate them as a name."""
    return _validate_cores([core for core in (t.strip(_STRIP) for t in tokens) if core])


def _clean_run(run: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Strip surface punctuation off each token, tightening offsets to the core.

    Tokens that strip to empty (a lone bracket or quote) are dropped. Every
    surviving core carries the offsets that bound *its own* characters, so the
    first and last cores delimit the real name — excluding leading ``(`` /
    trailing ``.`` that validation would otherwise discard from the surface
    while leaving them inside the raw span (dbl.23).
    """
    cleaned: list[tuple[str, int, int]] = []
    for tok, start, _end in run:
        lead = len(tok) - len(tok.lstrip(_STRIP))
        core = tok.strip(_STRIP)
        if core:
            cleaned.append((core, start + lead, start + lead + len(core)))
    return cleaned


def _finalise_run(run: list[tuple[str, int, int]]) -> tuple[str, int, int] | None:
    """Trim leading generic adjectives, normalise, and return the name span.

    ``run`` is the collected token sequence in left-to-right order, each token
    carrying its ``(start, end)`` offsets within the window. Leading generic
    descriptive adjectives are dropped from the front (the dbl.21 span-boundary
    fix), which both shifts the start offset to the real name and lets a run
    that was *only* adjectives ("high-quality dataset") collapse to ``None``.

    Returns ``(surface, start, end)`` with window-relative offsets, or ``None``.
    ``surface`` is the canonical (whitespace-collapsed) name; the offsets bound
    the cleaned cores in the source text, so a verbatim slice over them recovers
    the matched text even when its internal spacing is irregular (dbl.23).
    """
    while run and _is_leading_adjective(run[0][0]):
        run.pop(0)
    cleaned = _clean_run(run)
    surface = _validate_cores([core for core, _s, _e in cleaned])
    if surface is None:
        return None
    return surface, cleaned[0][1], cleaned[-1][2]


def _leading_name_span(window: str) -> tuple[str, int, int] | None:
    """Name span (surface + window-relative offsets) at the *start* of ``window``.

    The run stops *after* a token that ends its sentence (bead 23w): the
    terminator sits to that token's right, so the token itself stays in the name
    but nothing past the boundary can join it.
    """
    run: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", window):
        if not (_is_namey_token(m.group()) and len(run) < _MAX_NAME_TOKENS):
            break
        run.append((m.group(), m.start(), m.end()))
        if _ends_sentence(m.group()):
            break
    return _finalise_run(run)


def _trailing_name_span(window: str) -> tuple[str, int, int] | None:
    """Name span (surface + window-relative offsets) at the *end* of ``window``.

    Walking right-to-left, a token that ends its sentence is a boundary to the
    *left* of everything already collected (bead 23w). Once the run is non-empty
    that token belongs to the previous sentence and must not be absorbed, so the
    run stops *before* it.
    """
    run: list[tuple[str, int, int]] = []
    for m in reversed(list(re.finditer(r"\S+", window))):
        if not (_is_namey_token(m.group()) and len(run) < _MAX_NAME_TOKENS):
            break
        if run and _ends_sentence(m.group()):
            break
        run.insert(0, (m.group(), m.start(), m.end()))
    return _finalise_run(run)


def _take_leading_name(window: str) -> str | None:
    """Collect the run of name-shaped tokens at the *start* of ``window``."""
    span = _leading_name_span(window)
    return span[0] if span is not None else None


def _take_trailing_name(window: str) -> str | None:
    """Collect the run of name-shaped tokens at the *end* of ``window``."""
    span = _trailing_name_span(window)
    return span[0] if span is not None else None


# ---------------------------------------------------------------------------
# Cue catalog
# ---------------------------------------------------------------------------
# Each cue: (cue_id, compiled_regex, entity_type, base_confidence, position).
# ``position='prefix'`` → the name follows the cue; ``'suffix'`` → it precedes.
# Regexes are IGNORECASE on the cue keywords only; name shape is enforced in
# Python against the original-case body text, so case information survives.


@dataclass(frozen=True)
class _Cue:
    cue_id: str
    regex: "re.Pattern[str]"
    entity_type: str
    confidence: float
    position: str  # 'prefix' | 'suffix'
    # Generic verb/preposition cues ("we used X", "via X") fire on far more
    # phrasings than head-noun cues, so they additionally require the candidate
    # to be uppercase-initial — a proper name or acronym, not a lower-case
    # method phrase ("gas-phase", "1D adaptive-kernel").
    strict_candidate: bool = False


def _c(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.IGNORECASE)


_CUES: tuple[_Cue, ...] = (
    # --- software, prefix cues -------------------------------------------
    _Cue("implemented_in",
         _c(r"\bimplemented\s+(?:in|with|using)\s+(?:the\s+)?"),
         ENTITY_TYPE_SOFTWARE, 0.88, "prefix"),
    _Cue("run_with",
         _c(r"\b(?:run|ran|executed?|performed)\s+(?:with|using)\s+(?:the\s+)?"),
         ENTITY_TYPE_SOFTWARE, 0.78, "prefix"),
    _Cue("we_used",
         _c(r"\bwe\s+(?:use|used|employ|employed|adopt|adopted|"
            r"utilis\w+|utiliz\w+|appl(?:y|ied))\s+(?:the\s+)?"),
         ENTITY_TYPE_SOFTWARE, 0.70, "prefix", strict_candidate=True),
    _Cue("via",
         _c(r"\bvia\s+(?:the\s+)?"),
         ENTITY_TYPE_SOFTWARE, 0.62, "prefix", strict_candidate=True),
    # --- software, suffix cues -------------------------------------------
    _Cue("software_head",
         _c(r"\s+(?:software|package|library|toolkit|pipeline|"
            r"codebase|framework)\b"),
         ENTITY_TYPE_SOFTWARE, 0.90, "suffix"),
    _Cue("code_head",
         _c(r"\s+code\b"),
         ENTITY_TYPE_SOFTWARE, 0.74, "suffix"),
    # --- dataset, prefix cues --------------------------------------------
    _Cue("data_from",
         _c(r"\bdata\s+(?:from|drawn\s+from|taken\s+from|obtained\s+from|"
            r"retrieved\s+from)\s+(?:the\s+)?"),
         ENTITY_TYPE_DATASET, 0.82, "prefix"),
    # --- dataset, suffix cues --------------------------------------------
    _Cue("dataset_head",
         _c(r"\s+(?:dataset|data\s+set|catalog|catalogue|survey|database|"
            r"archive|simulations?)\b"),
         ENTITY_TYPE_DATASET, 0.90, "suffix"),
    _Cue("data_release",
         _c(r"\s+(?:DR\s*\d|data\s+release\s*\d)"),
         ENTITY_TYPE_DATASET, 0.88, "suffix"),
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformalMention:
    """One context-disambiguated informal reference.

    ``start_char`` / ``end_char`` are absolute offsets of the name span in the
    original body text. ``surface`` is the matched name; ``canonical_name`` is
    the upsert key into ``entities`` (currently identical to ``surface`` —
    canonicalisation across sources happens downstream).
    """

    surface: str
    canonical_name: str
    entity_type: str
    cue_id: str
    confidence: float
    start_char: int
    end_char: int
    evidence_span: str


def _evidence_snippet(body: str, start: int, end: int) -> str:
    """Return a context window centred on the name span, for spot-checking."""
    pad = (_EVIDENCE_SPAN_CHARS - (end - start)) // 2
    lo = max(0, start - max(pad, 0))
    hi = min(len(body), end + max(pad, 0))
    return body[lo:hi].strip()


def _locate_name(body: str, cue: _Cue, match: "re.Match[str]") -> tuple[str, int, int] | None:
    """Resolve the name span adjacent to a fired cue, or ``None``.

    Offsets are recomputed against ``body`` (not the sliced window) so they are
    absolute and usable as evidence.
    """
    if cue.position == "prefix":
        win_lo = match.end()
        window = body[win_lo:win_lo + _NAME_WINDOW_CHARS]
        span = _leading_name_span(window)
        if span is None:
            return None
        name, ws, we = span
        if cue.strict_candidate and not name[0].isupper():
            return None  # weak cue + lower-case method phrase
        start = win_lo + ws
        end = win_lo + we
        if _AUTHOR_CITATION_RE.match(body[end:end + 8]):
            return None  # "data from Cohen et al." — an author, not a source
        return name, start, end

    # suffix: the name precedes the cue keyword.
    win_hi = match.start()
    win_lo = max(0, win_hi - _NAME_WINDOW_CHARS)
    window = body[win_lo:win_hi]
    span = _trailing_name_span(window)
    if span is None:
        return None
    name, ws, we = span
    return name, win_lo + ws, win_lo + we


def detect_informal_references(body: str) -> list[InformalMention]:
    """Detect informal software/dataset references in ``body``.

    Pure function — no DB, no model. Returns one :class:`InformalMention` per
    distinct ``(canonical_name, entity_type)`` pair, keeping the highest-cue
    confidence when several cues point at the same name.
    """
    if not body:
        return []

    # Keyed by (canonical_name, entity_type); keep the strongest evidence.
    best: dict[tuple[str, str], InformalMention] = {}
    for cue in _CUES:
        for match in cue.regex.finditer(body):
            located = _locate_name(body, cue, match)
            if located is None:
                continue
            name, start, end = located
            key = (name, cue.entity_type)
            existing = best.get(key)
            if existing is not None and existing.confidence >= cue.confidence:
                continue
            best[key] = InformalMention(
                surface=body[start:end],
                canonical_name=name,
                entity_type=cue.entity_type,
                cue_id=cue.cue_id,
                confidence=cue.confidence,
                start_char=start,
                end_char=end,
                evidence_span=_evidence_snippet(body, start, end),
            )
    return sorted(best.values(), key=lambda m: (m.start_char, m.entity_type))


# ---------------------------------------------------------------------------
# DB writers (mirror scix.extract.ner_pass)
# ---------------------------------------------------------------------------


_ENTITY_UPSERT_SQL = """
    INSERT INTO entities
        (canonical_name, entity_type, source, source_version)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
        SET updated_at = now(),
            source_version = EXCLUDED.source_version
    RETURNING id
"""

# Tier-3 informal-mention writer: owns its own (link_type='informal_mention',
# match_method='lafia_informal') namespace in document_entities and never
# clobbers other producers. Mirrors scix.extract.ner_pass; the M13 resolver
# (scix.resolve_entities) is a read-side resolver with no write path, so tier
# producers write directly. u10 will route tier writes through the resolver.
# resolver-lint: bypass
_DOC_ENTITY_UPSERT_SQL = """
    INSERT INTO document_entities
        (bibcode, entity_id, link_type, confidence, match_method, tier, evidence)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (bibcode, entity_id, link_type, tier) DO UPDATE
        SET confidence = GREATEST(document_entities.confidence, EXCLUDED.confidence),
            match_method = EXCLUDED.match_method,
            evidence = EXCLUDED.evidence
"""


def _upsert_entities(
    conn: "psycopg.Connection",
    mentions: list[InformalMention],
) -> dict[tuple[str, str], int]:
    """Upsert distinct ``(canonical_name, entity_type)`` pairs; return id map."""
    distinct = sorted({(m.canonical_name, m.entity_type) for m in mentions})
    id_map: dict[tuple[str, str], int] = {}
    if not distinct:
        return id_map
    with conn.cursor() as cur:
        for canon, etype in distinct:
            cur.execute(_ENTITY_UPSERT_SQL, (canon, etype, SOURCE, SOURCE_VERSION))
            row = cur.fetchone()
            if row is not None:
                id_map[(canon, etype)] = row[0]
    return id_map


def insert_mentions(
    conn: "psycopg.Connection",
    bibcode: str,
    mentions: list[InformalMention],
) -> int:
    """Upsert ``entities`` + ``document_entities`` rows for one paper.

    Returns the number of ``document_entities`` rows written. Does nothing and
    returns 0 when ``mentions`` is empty (no "scanned-nothing" sentinel — this
    pass tracks progress via ``ingest_log`` in the driver, not per-paper rows).
    """
    if not mentions:
        return 0
    id_map = _upsert_entities(conn, mentions)
    rows: list[tuple[str, int, str, float, str, int, Jsonb]] = []
    for m in mentions:
        eid = id_map.get((m.canonical_name, m.entity_type))
        if eid is None:
            continue
        evidence = Jsonb(
            {
                "cue": m.cue_id,
                "surface": m.surface,
                "snippet": m.evidence_span,
                "start": m.start_char,
                "end": m.end_char,
            }
        )
        rows.append(
            (bibcode, eid, LINK_TYPE, m.confidence, MATCH_METHOD, LAFIA_TIER, evidence)
        )
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_DOC_ENTITY_UPSERT_SQL, rows)
    return len(rows)
