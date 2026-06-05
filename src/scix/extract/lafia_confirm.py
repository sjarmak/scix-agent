"""GLiNER type-confirmation pass over Lafia cue-extracted spans (bead dbl.20).

The Lafia detector (``scix.extract.lafia``) is a high-recall cue heuristic: a
regex cue ("implemented in", "… survey") plus a name-shape filter. On real arXiv
bodies its precision plateaus at ~60-67% (bead dbl.18) — cue + morphology
heuristics are empirically exhausted there and more regex does not help.

This module adds the cheap ML discriminator dbl.18 recommends: re-run the
*trained* GLiNER NER model (``scix.extract.ner_pass.GlinerExtractor``) over the
local context of each cue-extracted candidate span and keep only the candidates
GLiNER independently tags as a named research resource of a compatible type. The
heuristic supplies recall (it spots context-disambiguated mentions GLiNER's
whole-document pass misses); GLiNER supplies precision (it rejects the
common-word false positives the cue alone cannot).

Why a context window, not the whole body
-----------------------------------------
GLiNER over the full body is exactly the pass that misses these mentions in the
first place (its score on an unremarkable name in a long document is low). Run
instead over a tight window centred on the candidate span: the cue has already
told us *where* to look, so the model only has to answer "is the thing at this
span a real software/dataset?" — a much easier call that clears the confidence
floor far more often.

Eval-only at v1: this module performs no DB writes and changes no provenance.
The production pass over the OA subset stays a separate operator-gated bead; when
it lands it reuses :func:`confirm_mentions` ahead of ``lafia.insert_mentions``.

ZFC note: the type-compatibility policy is a fixed mechanical mapping and the
surface match is structural string comparison — the *semantic* judgement (is
this span a real entity?) is delegated to the trained model, not hand-coded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from scix.extract.lafia import InformalMention
from scix.extract.ner_pass import GlinerExtractor, Mention, PaperInput, canonicalize

logger = logging.getLogger(__name__)

# Chars of body context handed to GLiNER on each side of the candidate span.
# Wide enough to give the model the disambiguating sentence, tight enough that a
# different entity elsewhere in the paragraph does not bleed in.
DEFAULT_CONTEXT_CHARS: int = 160

# Which GLiNER entity types confirm a Lafia candidate of a given type. "software"
# stays strict; a Lafia "dataset" candidate ("the LAMOST survey", "SDSS DR16") is
# routinely tagged by GLiNER as a mission or instrument rather than the literal
# "dataset" label, so those count as confirmation. Candidate types absent from
# this map fall back to requiring an exact GLiNER type match.
DEFAULT_TYPE_POLICY: Mapping[str, frozenset[str]] = {
    "software": frozenset({"software"}),
    "dataset": frozenset({"dataset", "mission", "instrument"}),
}


@dataclass(frozen=True)
class ConfirmedMention:
    """A Lafia candidate that GLiNER independently confirmed.

    ``mention`` is the original :class:`~scix.extract.lafia.InformalMention`;
    the ``gliner_*`` fields record which GLiNER mention confirmed it (the
    highest-confidence compatible overlap in the candidate's context window).
    """

    mention: InformalMention
    gliner_type: str
    gliner_confidence: float
    gliner_surface: str


def _context_window(body: str, mention: InformalMention, context_chars: int) -> str:
    """Slice the body around a candidate span for GLiNER re-scoring."""
    lo = max(0, mention.start_char - context_chars)
    hi = min(len(body), mention.end_char + context_chars)
    return body[lo:hi]


def _surfaces_overlap(candidate: str, gliner: str) -> bool:
    """True if two canonical surfaces denote the same span.

    Exact match, or one is a token-subset of the other so that a candidate
    "LAMOST" confirms against a GLiNER span "LAMOST survey" (and vice versa).
    Purely mechanical — no semantic comparison.
    """
    if candidate == gliner:
        return True
    cand_tokens = set(candidate.split())
    glin_tokens = set(gliner.split())
    if not cand_tokens or not glin_tokens:
        return False
    return cand_tokens <= glin_tokens or glin_tokens <= cand_tokens


def _best_confirming(
    mention: InformalMention,
    gliner_mentions: list[Mention],
    type_policy: Mapping[str, frozenset[str]],
) -> Mention | None:
    """Return the highest-confidence GLiNER mention that confirms ``mention``."""
    accept = type_policy.get(mention.entity_type, frozenset({mention.entity_type}))
    candidate_canon = canonicalize(mention.surface)
    best: Mention | None = None
    for g in gliner_mentions:
        if g.entity_type not in accept:
            continue
        if not _surfaces_overlap(candidate_canon, g.canonical_name):
            continue
        if best is None or g.confidence > best.confidence:
            best = g
    return best


def confirm_mentions(
    extractor: GlinerExtractor,
    body: str,
    mentions: list[InformalMention],
    *,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
    type_policy: Mapping[str, frozenset[str]] = DEFAULT_TYPE_POLICY,
) -> list[ConfirmedMention]:
    """Keep only Lafia candidates GLiNER confirms as a compatible named entity.

    One batched GLiNER call per body: each candidate's context window is scored,
    and the candidate survives iff GLiNER emits a mention in that window whose
    type is compatible (``type_policy``) and whose surface overlaps the
    candidate. The model's own confidence floor (``extractor.confidence``)
    applies inside ``predict`` — no extra threshold here.
    """
    if not mentions:
        return []

    windows = [
        PaperInput(bibcode="", text=_context_window(body, m, context_chars))
        for m in mentions
    ]
    per_window = extractor.predict(windows)

    confirmed: list[ConfirmedMention] = []
    for mention, gliner_mentions in zip(mentions, per_window, strict=True):
        best = _best_confirming(mention, gliner_mentions, type_policy)
        if best is not None:
            confirmed.append(
                ConfirmedMention(
                    mention=mention,
                    gliner_type=best.entity_type,
                    gliner_confidence=best.confidence,
                    gliner_surface=best.surface_text,
                )
            )
    return confirmed
