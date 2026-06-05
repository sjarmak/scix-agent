"""Tests for the GLiNER type-confirmation pass (src/scix/extract/lafia_confirm.py).

GLiNER is stubbed: a fake model implementing ``batch_predict_entities`` returns
scripted entities per text, so these tests run on CPU with no model download.
The stub lets us assert the confirmation logic — type compatibility, surface
overlap, the context window, and recall/precision behaviour — deterministically.
"""

from __future__ import annotations

from scix.extract.lafia import InformalMention, detect_informal_references
from scix.extract.lafia_confirm import (
    DEFAULT_TYPE_POLICY,
    ConfirmedMention,
    _context_window,
    _surfaces_overlap,
    confirm_mentions,
)
from scix.extract.ner_pass import GlinerExtractor


class _FakeGliner:
    """Stub GLiNER model.

    ``scripted`` maps a substring -> (surface, label, score). For each input
    text, every substring found in it contributes its scripted entity. This lets
    a test say "when GLiNER sees a window containing 'PyTorch', it returns a
    software entity 'PyTorch'@0.93".
    """

    def __init__(self, scripted: list[tuple[str, str, str, float]]) -> None:
        # (trigger_substring, surface, label, score)
        self.scripted = scripted

    def batch_predict_entities(self, texts, labels, threshold, batch_size):
        out = []
        for text in texts:
            ents = []
            for trigger, surface, label, score in self.scripted:
                if trigger in text and score >= threshold:
                    ents.append({"text": surface, "label": label, "score": score})
            out.append(ents)
        return out


def _extractor(scripted: list[tuple[str, str, str, float]]) -> GlinerExtractor:
    return GlinerExtractor(model=_FakeGliner(scripted), confidence=0.7)


def _mention(surface: str, etype: str, start: int, end: int) -> InformalMention:
    return InformalMention(
        surface=surface,
        canonical_name=surface,
        entity_type=etype,
        cue_id="test",
        confidence=0.9,
        start_char=start,
        end_char=end,
        evidence_span="",
    )


# ---------------------------------------------------------------------------
# Surface overlap
# ---------------------------------------------------------------------------


def test_surfaces_overlap_exact():
    assert _surfaces_overlap("pytorch", "pytorch")


def test_surfaces_overlap_token_subset_both_directions():
    assert _surfaces_overlap("lamost", "lamost survey")
    assert _surfaces_overlap("lamost survey", "lamost")


def test_surfaces_overlap_rejects_disjoint():
    assert not _surfaces_overlap("pytorch", "tensorflow")


def test_surfaces_overlap_rejects_empty():
    assert not _surfaces_overlap("", "pytorch")


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


def test_context_window_centres_on_span_and_clamps():
    body = "abcdefghij KEYWORD klmnopqrst"
    m = _mention("KEYWORD", "software", 11, 18)
    win = _context_window(body, m, context_chars=5)
    assert "KEYWORD" in win
    # 5 chars each side, clamped to body bounds.
    assert win == body[6:23]


# ---------------------------------------------------------------------------
# confirm_mentions — the core gate
# ---------------------------------------------------------------------------


def test_confirms_matching_type_and_surface():
    body = "The model was implemented in PyTorch and trained on a GPU."
    mentions = detect_informal_references(body)
    assert any(m.surface == "PyTorch" for m in mentions)

    extractor = _extractor([("PyTorch", "PyTorch", "software", 0.93)])
    confirmed = confirm_mentions(extractor, body, mentions)

    assert [c.mention.surface for c in confirmed] == ["PyTorch"]
    assert confirmed[0].gliner_type == "software"
    assert confirmed[0].gliner_confidence == 0.93
    assert isinstance(confirmed[0], ConfirmedMention)


def test_rejects_when_gliner_finds_nothing():
    """A heuristic false positive GLiNER does not recognise is dropped."""
    body = "We used Standard reduction techniques throughout the analysis."
    mentions = [_mention("Standard", "software", 8, 16)]
    extractor = _extractor([])  # GLiNER tags nothing
    assert confirm_mentions(extractor, body, mentions) == []


def test_rejects_incompatible_type():
    """GLiNER sees an entity at the span but of an incompatible type."""
    body = "Observations from the Hubble telescope were analysed."
    mentions = [_mention("Hubble", "software", 22, 28)]
    # GLiNER calls it a mission, which does NOT confirm a *software* candidate.
    extractor = _extractor([("Hubble", "Hubble", "mission", 0.95)])
    assert confirm_mentions(extractor, body, mentions) == []


def test_dataset_confirmed_by_mission_or_instrument():
    """A dataset candidate is confirmed by a compatible mission/instrument tag."""
    body = "Spectra were drawn from the LAMOST survey data release."
    mentions = [_mention("LAMOST", "dataset", 28, 34)]
    extractor = _extractor([("LAMOST", "LAMOST survey", "mission", 0.88)])
    confirmed = confirm_mentions(extractor, body, mentions)
    assert len(confirmed) == 1
    assert confirmed[0].gliner_type == "mission"


def test_rejects_entity_at_wrong_surface():
    """GLiNER finds a different entity in the window — candidate not confirmed."""
    body = "We used emcee alongside the well-known TensorFlow framework here."
    mentions = [_mention("emcee", "software", 8, 13)]
    # GLiNER only recognises TensorFlow, not the candidate emcee.
    extractor = _extractor([("TensorFlow", "TensorFlow", "software", 0.91)])
    assert confirm_mentions(extractor, body, mentions) == []


def test_keeps_highest_confidence_confirmation():
    body = "implemented in PyTorch, the PyTorch backend was tuned."
    mentions = [_mention("PyTorch", "software", 15, 22)]
    extractor = _extractor(
        [
            ("PyTorch", "PyTorch", "software", 0.75),
            ("PyTorch backend", "PyTorch", "software", 0.95),
        ]
    )
    confirmed = confirm_mentions(extractor, body, mentions)
    assert len(confirmed) == 1
    assert confirmed[0].gliner_confidence == 0.95


def test_empty_mentions_short_circuits():
    extractor = _extractor([("x", "x", "software", 0.9)])
    assert confirm_mentions(extractor, "any body text", []) == []


def test_custom_type_policy_strict_exact():
    """A strict policy (exact type only) rejects the mission-tagged dataset."""
    body = "data from the LAMOST survey."
    mentions = [_mention("LAMOST", "dataset", 14, 20)]
    extractor = _extractor([("LAMOST", "LAMOST", "mission", 0.9)])
    strict = {"dataset": frozenset({"dataset"})}
    assert confirm_mentions(extractor, body, mentions, type_policy=strict) == []
    # …but the default policy accepts it.
    assert len(confirm_mentions(extractor, body, mentions)) == 1


def test_default_policy_shape():
    assert DEFAULT_TYPE_POLICY["software"] == frozenset({"software"})
    assert "mission" in DEFAULT_TYPE_POLICY["dataset"]


def test_unknown_candidate_type_requires_exact_type_match():
    """A candidate type absent from the policy falls back to exact-type match."""
    body = "Computed with the QuTiP method in the simulation."
    mentions = [_mention("QuTiP", "method", 18, 23)]
    # GLiNER tags it "software" — not the candidate's "method" type — so the
    # fallback (require exact "method") rejects it…
    extractor = _extractor([("QuTiP", "QuTiP", "software", 0.95)])
    assert confirm_mentions(extractor, body, mentions) == []
    # …and an exact "method" tag confirms it.
    extractor = _extractor([("QuTiP", "QuTiP", "method", 0.95)])
    assert len(confirm_mentions(extractor, body, mentions)) == 1
