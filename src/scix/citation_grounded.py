"""Citation-grounded gate (PRD MH-6 + amendment A6).

Post-draft mechanical check that every assertion in a persona's draft answer
traces to a tool-result quote. Unmatched assertions are flagged; the persona is
given up to 2 revision turns to substantiate or remove them; residual unmatched
sentences after the budget is spent are auto-stripped from the final output.

This module is **mechanical, not epistemic**. It verifies that the draft's
sentences map to retrieved evidence — it cannot verify that the evidence is
true, that the paper's claim was correct, or that the corpus contains the
right sources. See ``docs/citation_grounded.md`` "Grounding != Truth" for the
distinction documented in the SciX Deep Search v1 PRD (MH-6, Tension 2).

Assertion parsing
-----------------
A sentence is treated as an assertion (and therefore subject to grounding)
when:

1. It ends in ``.``, ``!``, or ``?`` (or end-of-text), AND
2. It contains at least one of:

   * A content verb from a small lexicon (``find``, ``show``, ``demonstrate``,
     ``conclude``, ``report``, ``observe``, ``measure``, ``detect``, ``derive``,
     ``predict``, ``confirm``, ``refute``, ``suggest``, ``indicate``,
     ``present``, ``propose``, ``claim``, ``argue``, ``establish``,
     ``identify``, ``infer``, ``support``, ``reveal``, plus inflections);
   * A digit (numeric content);
   * An entity-like uppercase mid-sentence token (e.g., ``Riess`` in
     ``We confirm Riess+ 2011 results``);
   * A parenthetical citation (``(Smith 2011)``, ``(2011)``, or ``[1]``).

Sentences without claim content (questions, transitions, headers) are passed
through ungrounded-checked.

Substring short-circuit
-----------------------
If any tool-result quote contains the assertion's normalized text as a
substring, the grounding score short-circuits to ``0.95`` without computing
embeddings. This dominates the embedding path for exact-quote citations and
saves the GPU round-trip.

Embedding path
--------------
Otherwise the assertion and tool quotes are embedded via INDUS (the
NASA-IBM scientific-domain encoder, 768d). Cosine similarity between the
assertion and each quote is computed; the maximum is the assertion's score.

The embedder is injectable via :func:`set_embedder` so tests can substitute
a deterministic fake without loading INDUS.

Revision protocol
-----------------
:class:`RevisionDispatcher` is a Protocol with a single ``revise(draft,
unmatched)`` method. Production wraps a Claude Code OAuth subagent; tests
inject a canned-output fake. The bounded-revision loop in
:func:`revise_with_gate` calls ``revise`` at most ``max_revisions`` times.
Residuals are stripped per amendment A6:

* Default mode (``rigor_mode=False``) — superscript markers (¹²³…) inline,
  with a "Footnotes" block appended listing the stripped sentences.
* Rigor mode (``rigor_mode=True``) — literal ``[ungrounded claim removed]``
  inline (the original MH-6 wording).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD: float = 0.82
"""PRD MH-6 starting threshold; tuned via shadow-mode in production."""

DEFAULT_MAX_REVISIONS: int = 2
"""PRD MH-6: bounded revision budget."""

SUBSTRING_SCORE: float = 0.95
"""Score returned by the substring short-circuit. Exceeds DEFAULT_THRESHOLD
so substring-matched assertions are always grounded, regardless of any
embedding-cosine result."""

INLINE_MARKER: str = "[ungrounded claim removed]"
"""Rigor-mode marker (PRD MH-6 original literal)."""

# Superscripts U+00B9, U+00B2, U+00B3 are out of order in Unicode for
# historical reasons; the rest follow at U+2074..2079.
_SUPERSCRIPTS: tuple[str, ...] = (
    "¹",  # 1
    "²",  # 2
    "³",  # 3
    "⁴",  # 4
    "⁵",  # 5
    "⁶",  # 6
    "⁷",  # 7
    "⁸",  # 8
    "⁹",  # 9
)

_CONTENT_VERBS: frozenset[str] = frozenset(
    {
        # Discovery / reporting
        "find", "found", "finds", "finding",
        "show", "shows", "showed", "shown",
        "demonstrate", "demonstrates", "demonstrated",
        "conclude", "concluded", "concludes",
        "report", "reported", "reports",
        "observe", "observed", "observes",
        "measure", "measured", "measures",
        "detect", "detected", "detects",
        "derive", "derived", "derives",
        "predict", "predicted", "predicts",
        "confirm", "confirmed", "confirms",
        "refute", "refuted", "refutes",
        "suggest", "suggested", "suggests",
        "indicate", "indicated", "indicates",
        "present", "presented", "presents",
        "propose", "proposed", "proposes",
        "claim", "claimed", "claims",
        "argue", "argued", "argues",
        "establish", "established", "establishes",
        "identify", "identified", "identifies",
        "infer", "inferred", "infers",
        "support", "supported", "supports",
        "reveal", "revealed", "reveals",
        # Causal / requirement / state verbs that often appear in
        # research-prose claims.
        "require", "requires", "required",
        "imply", "implies", "implied",
        "yield", "yields", "yielded",
        "produce", "produces", "produced",
        "explain", "explains", "explained",
        "host", "hosts", "hosted",
        "contain", "contains", "contained",
        "trace", "traces", "traced",
        "drive", "drives", "drove", "driven",
        "calibrate", "calibrates", "calibrated",
        "invent", "invents", "invented",
        "discover", "discovers", "discovered",
        "develop", "develops", "developed",
        "orbit", "orbits", "orbited",
        "collide", "collides", "collided",
        "trained", "train", "trains",
        "showed", "showcase", "showcases",
        "build", "builds", "built",
        # Linking verbs ("the X was Y").
        "is", "are", "was", "were", "has", "have", "had",
    }
)

# Sentence boundary: end punctuation followed by whitespace + capital letter.
_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])")
_DIGIT_RE: re.Pattern[str] = re.compile(r"\d")
# Mid-sentence uppercase token: a capitalized word that isn't the
# first token of the sentence.
_MID_UPPER_RE: re.Pattern[str] = re.compile(r"\s[A-Z][A-Za-z]{2,}")
# Parenthetical citation patterns: (Author 2011), (2011), [1], [12]
_CITATION_RE: re.Pattern[str] = re.compile(
    r"\([A-Z][A-Za-z]+\s+\d{4}\)|\(\d{4}\)|\[\d+\]"
)
_WORD_RE: re.Pattern[str] = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# DTOs and Protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingReport:
    """Result of a single :func:`grounded_check` call.

    Attributes:
        assertions: Tuple of sentences flagged as assertions (subject to gate).
        unmatched: Tuple of assertions whose grounding score < threshold.
        grounded: True iff ``unmatched`` is empty (every assertion matched).
        threshold_used: The threshold that was applied.
    """

    assertions: tuple[str, ...]
    unmatched: tuple[str, ...]
    grounded: bool
    threshold_used: float


@dataclass(frozen=True)
class RevisedDraft:
    """Result of a bounded :func:`revise_with_gate` loop.

    Attributes:
        answer: Final answer text (with stripped sentences replaced per A6).
        stripped: Tuple of sentences removed because they remained unmatched
            after the revision budget was exhausted.
        revision_count: Number of revisions actually performed
            (``0..max_revisions``).
        grounded: True iff no sentences had to be stripped.
        threshold_used: Threshold that was applied throughout.
    """

    answer: str
    stripped: tuple[str, ...]
    revision_count: int
    grounded: bool
    threshold_used: float


class RevisionDispatcher(Protocol):
    """A callable that takes the current draft and the list of unmatched
    sentences, and returns a revised draft string.

    Tests inject a deterministic fake; production wraps a Claude Code
    OAuth subagent revision call.
    """

    def revise(self, draft: str, unmatched: list[str]) -> str: ...


# ---------------------------------------------------------------------------
# Embedder injection seam
# ---------------------------------------------------------------------------

Embedder = Callable[[list[str]], list[list[float]]]

_embedder_override: Embedder | None = None


def set_embedder(fn: Embedder | None) -> None:
    """Override the embedder (test seam).

    Pass ``None`` to restore the default INDUS embedder.
    """
    global _embedder_override
    _embedder_override = fn


def _default_embedder(texts: list[str]) -> list[list[float]]:
    """Default embedder — wraps INDUS via :mod:`scix.embed`.

    This loads the INDUS model on first call (cached for process lifetime).
    Tests should call :func:`set_embedder` with a deterministic fake to
    avoid the model load.
    """
    if not texts:
        return []
    # Local import: avoid eager torch/transformers load at module import.
    from scix.embed import MODEL_POOLING, embed_batch, load_model

    model, tokenizer = load_model("indus", device="cpu")
    return embed_batch(
        model,
        tokenizer,
        texts,
        batch_size=min(32, max(1, len(texts))),
        pooling=MODEL_POOLING.get("indus", "mean"),
    )


def _get_embedder() -> Embedder:
    return _embedder_override if _embedder_override is not None else _default_embedder


# ---------------------------------------------------------------------------
# Sentence parsing
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence-like segments.

    The split is regex-based and imperfect on edge cases (Mr., 1.5,
    e.g.), but adequate for grounding-gate detection — the cost of a
    misclassified sentence is at most one revision turn.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _is_assertion(sentence: str) -> bool:
    """Heuristic: should this sentence be subject to the grounding gate?

    See module docstring for the rule.
    """
    if not sentence:
        return False
    # Any digit: numeric claim (often quantitative result).
    if _DIGIT_RE.search(sentence):
        return True
    # Parenthetical citation pattern.
    if _CITATION_RE.search(sentence):
        return True
    # Content verb anywhere in the sentence (lowercase token match).
    lowered = sentence.lower()
    for word in _WORD_RE.findall(lowered):
        if word in _CONTENT_VERBS:
            return True
    # Mid-sentence uppercase token (named entity).
    if _MID_UPPER_RE.search(sentence):
        return True
    return False


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and strip trailing sentence punctuation.

    Used for substring matching: the assertion's terminal ``.``/``!``/``?``
    should not block a substring hit when the assertion appears mid-quote.
    """
    return " ".join(text.lower().split()).rstrip(".!?,;:")


# ---------------------------------------------------------------------------
# Tool-result extraction
# ---------------------------------------------------------------------------

_QUOTE_KEYS: tuple[str, ...] = (
    "quote",
    "text",
    "snippet",
    "context_snippet",
    "body",
    "content",
    "result",
    "answer",
    "abstract",
    "title",
)


def _extract_quotes(tool_results: list[dict]) -> list[str]:
    """Pull quotable strings out of tool-result dicts.

    Walks each dict for known string-bearing keys, plus any nested list
    under ``results``. Empty/whitespace strings are dropped.
    """
    quotes: list[str] = []

    def _add(s: Any) -> None:
        if isinstance(s, str):
            stripped = s.strip()
            if stripped:
                quotes.append(stripped)
        elif isinstance(s, list):
            for item in s:
                _add(item)
        elif isinstance(s, dict):
            _walk_dict(s)

    def _walk_dict(d: dict) -> None:
        for key in _QUOTE_KEYS:
            if key in d:
                _add(d[key])
        # Common nested-results pattern.
        if "results" in d and isinstance(d["results"], list):
            for item in d["results"]:
                _add(item)

    for result in tool_results or []:
        if isinstance(result, dict):
            _walk_dict(result)
        elif isinstance(result, str):
            _add(result)

    return quotes


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length numeric vectors.

    Returns ``0.0`` when either vector is all zeros (avoids div-by-zero).
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _score_assertion(
    assertion: str,
    tool_quotes: list[str],
    embedder: Embedder,
) -> float:
    """Compute the max grounding score for an assertion.

    First applies the substring short-circuit. Otherwise embeds both sides
    and returns the maximum cosine.
    """
    if not tool_quotes:
        return 0.0

    norm_assert = _normalize(assertion)
    if not norm_assert:
        return 0.0

    # Substring short-circuit. Bidirectional: assertion contained in a
    # quote OR a quote contained in the assertion both count as exact-quote
    # citations.
    for quote in tool_quotes:
        norm_q = _normalize(quote)
        if not norm_q:
            continue
        if norm_assert in norm_q or norm_q in norm_assert:
            return SUBSTRING_SCORE

    # Embedding path: embed assertion + all quotes in one batch.
    vectors = embedder([assertion, *tool_quotes])
    if not vectors:
        return 0.0
    a_vec = vectors[0]
    best = 0.0
    for q_vec in vectors[1:]:
        score = _cosine(a_vec, q_vec)
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Public API: grounded_check
# ---------------------------------------------------------------------------


def grounded_check(
    draft: str,
    tool_results: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> GroundingReport:
    """Score every assertion in ``draft`` against ``tool_results``.

    Args:
        draft: The persona's candidate answer text.
        tool_results: List of tool-call result dicts. Common string-bearing
            keys are extracted (``quote``, ``text``, ``snippet``,
            ``context_snippet``, ``body``, ``content``, ``result``,
            ``answer``, ``abstract``, ``title``); nested ``results``
            lists are flattened.
        threshold: Minimum cosine similarity for an assertion to count as
            grounded. Default is :data:`DEFAULT_THRESHOLD` (0.82).

    Returns:
        :class:`GroundingReport` with the assertions detected, the subset
        unmatched, the boolean ``grounded`` flag, and the threshold used.
    """
    embedder = _get_embedder()
    tool_quotes = _extract_quotes(tool_results)

    sentences = _split_sentences(draft)
    assertions = [s for s in sentences if _is_assertion(s)]

    unmatched: list[str] = []
    for assertion in assertions:
        score = _score_assertion(assertion, tool_quotes, embedder)
        if score < threshold:
            unmatched.append(assertion)

    return GroundingReport(
        assertions=tuple(assertions),
        unmatched=tuple(unmatched),
        grounded=len(unmatched) == 0,
        threshold_used=threshold,
    )


# ---------------------------------------------------------------------------
# Residual stripping (A6)
# ---------------------------------------------------------------------------


def _superscript(n: int) -> str:
    """Return a superscript marker for footnote ``n`` (1-indexed).

    Uses Unicode superscripts for 1..9; falls back to ``[n]`` for n>=10.
    """
    if 1 <= n <= 9:
        return _SUPERSCRIPTS[n - 1]
    return f"[{n}]"


def _strip_residuals(
    draft: str,
    stripped: list[str],
    *,
    rigor_mode: bool,
) -> str:
    """Remove ``stripped`` sentences from ``draft`` and append A6 markers.

    In rigor mode, each stripped sentence is replaced inline with the
    literal :data:`INLINE_MARKER`. In default mode, each is replaced with
    a superscript marker and a "Footnotes" block is appended listing the
    stripped sentence text.
    """
    if not stripped:
        return draft

    result = draft
    footnote_entries: list[tuple[str, str]] = []
    for idx, sentence in enumerate(stripped, start=1):
        if rigor_mode:
            replacement = INLINE_MARKER
        else:
            marker = _superscript(idx)
            replacement = marker
            footnote_entries.append((marker, sentence))
        if sentence in result:
            result = result.replace(sentence, replacement, 1)
        else:
            # The sentence wasn't found verbatim — append the replacement so
            # the user still sees the marker. (This branch is rare; happens
            # if the persona normalized whitespace between draft and gate.)
            result = result.rstrip() + " " + replacement

    if footnote_entries:
        block_lines = ["Footnotes"]
        for marker, sentence in footnote_entries:
            block_lines.append(f"{marker} {sentence}")
        result = result.rstrip() + "\n\n" + "\n".join(block_lines) + "\n"

    return result


# ---------------------------------------------------------------------------
# Public API: revise_with_gate
# ---------------------------------------------------------------------------


def revise_with_gate(
    draft: str,
    tool_results: list[dict],
    dispatcher: RevisionDispatcher,
    *,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    rigor_mode: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> RevisedDraft:
    """Apply the bounded grounding-revision loop (PRD MH-6).

    Steps:
        1. Run :func:`grounded_check` on ``draft``.
        2. If ``grounded``, return immediately.
        3. Otherwise call ``dispatcher.revise(draft, unmatched)`` to get a
           revised draft, and re-check.
        4. Repeat at most ``max_revisions`` times.
        5. After the budget is spent, residual unmatched sentences are
           auto-stripped per amendment A6 (footnoted by default,
           ``[ungrounded claim removed]`` inline if ``rigor_mode``).

    Args:
        draft: Candidate answer text.
        tool_results: Tool-call result dicts (see :func:`grounded_check`).
        dispatcher: A :class:`RevisionDispatcher` (production or fake).
        max_revisions: Upper bound on revision turns (PRD: 2).
        rigor_mode: If True, replace residuals with the inline literal
            instead of footnoted superscripts.
        threshold: Cosine threshold (default :data:`DEFAULT_THRESHOLD`).

    Returns:
        :class:`RevisedDraft` with the final answer, stripped sentences,
        and revision count.
    """
    current = draft
    report = grounded_check(current, tool_results, threshold=threshold)
    revision_count = 0

    while not report.grounded and revision_count < max_revisions:
        revision_count += 1
        current = dispatcher.revise(current, list(report.unmatched))
        report = grounded_check(current, tool_results, threshold=threshold)

    if report.grounded:
        return RevisedDraft(
            answer=current,
            stripped=(),
            revision_count=revision_count,
            grounded=True,
            threshold_used=threshold,
        )

    # Residuals remain — auto-strip per A6.
    stripped = list(report.unmatched)
    final_answer = _strip_residuals(current, stripped, rigor_mode=rigor_mode)
    logger.warning(
        "citation_grounded: stripped %d residual unmatched sentence(s) "
        "after %d revision turn(s) (threshold=%.3f, rigor_mode=%s)",
        len(stripped),
        revision_count,
        threshold,
        rigor_mode,
    )
    return RevisedDraft(
        answer=final_answer,
        stripped=tuple(stripped),
        revision_count=revision_count,
        grounded=False,
        threshold_used=threshold,
    )
