"""Unit tests for the SOMD software-mention detector (bead dbl.22).

Model-free: the span-merge logic and the confirmation routing are exercised with
synthetic token predictions / a stub detector, so the suite needs neither the GPU
nor the trained checkpoint.
"""

from __future__ import annotations

import pytest

from scix.extract.lafia import InformalMention
from scix.extract.somd_detect import (
    SoftwareSpan,
    _overlaps,
    _spans_from_tokens,
    confirm_software_mentions,
)


def _mention(surface: str, entity_type: str, start: int, end: int) -> InformalMention:
    return InformalMention(
        surface=surface,
        canonical_name=surface,
        entity_type=entity_type,
        cue_id="t",
        confidence=0.9,
        start_char=start,
        end_char=end,
        evidence_span="",
    )


class TestSpansFromTokens:
    def test_single_begin_token_becomes_span(self):
        text = "we used MATLAB here"
        # tokens: [CLS] we used MAT ##LAB here [SEP] — offsets for MAT/##LAB cover MATLAB
        offsets = [(0, 0), (0, 2), (3, 7), (8, 11), (11, 14), (15, 19), (0, 0)]
        special = [1, 0, 0, 0, 0, 0, 1]
        labels = ["O", "O", "O", "B-Application_Usage", "I-Application_Usage", "O", "O"]
        probs = [0.9] * 7
        spans = _spans_from_tokens(text, offsets, special, labels, probs)
        assert len(spans) == 1
        assert spans[0].surface == "MATLAB"
        assert (spans[0].start_char, spans[0].end_char) == (8, 14)
        assert spans[0].label == "Application_Usage"

    def test_o_token_closes_span(self):
        text = "Unity and Blender"
        offsets = [(0, 0), (0, 5), (6, 9), (10, 17), (0, 0)]
        special = [1, 0, 0, 0, 1]
        labels = ["O", "B-Application_Usage", "O", "B-Application_Usage", "O"]
        spans = _spans_from_tokens(text, offsets, special, labels, [0.8] * 5)
        assert [s.surface for s in spans] == ["Unity", "Blender"]

    def test_new_begin_splits_adjacent_entities(self):
        text = "PyTorchTensorFlow"  # two B- tags back to back -> two spans
        offsets = [(0, 0), (0, 7), (7, 17), (0, 0)]
        special = [1, 0, 0, 1]
        labels = ["O", "B-Application_Usage", "B-Application_Usage", "O"]
        spans = _spans_from_tokens(text, offsets, special, labels, [0.7] * 4)
        assert len(spans) == 2

    def test_no_software_tokens_yields_no_spans(self):
        text = "the quick brown fox"
        offsets = [(0, 0), (0, 3), (4, 9), (10, 15), (16, 19), (0, 0)]
        special = [1, 0, 0, 0, 0, 1]
        labels = ["O"] * 6
        assert _spans_from_tokens(text, offsets, special, labels, [0.5] * 6) == []

    def test_score_is_mean_of_entity_token_probs(self):
        text = "scikit learn"
        offsets = [(0, 0), (0, 6), (7, 12), (0, 0)]
        special = [1, 0, 0, 1]
        labels = ["O", "B-Application_Usage", "I-Application_Usage", "O"]
        spans = _spans_from_tokens(text, offsets, special, labels, [0.1, 0.6, 0.8, 0.1])
        assert spans[0].score == pytest.approx(0.7)


class TestOverlaps:
    def test_overlapping_intervals(self):
        assert _overlaps(0, 5, 3, 8)

    def test_touching_intervals_do_not_overlap(self):
        # half-open: [0,5) and [5,10) share no position
        assert not _overlaps(0, 5, 5, 10)

    def test_disjoint_intervals(self):
        assert not _overlaps(0, 5, 6, 9)


class _StubDetector:
    """Returns preset software spans per window, in call order."""

    def __init__(self, per_window: list[list[SoftwareSpan]]):
        self._per_window = per_window

    def detect_batch(self, windows):
        assert len(windows) == len(self._per_window)
        return self._per_window


class TestConfirmSoftwareMentions:
    def test_dataset_candidate_passes_through_without_somd(self):
        body = "x" * 50
        m = _mention("CIFAR10", "dataset", 20, 27)
        # detector finds nothing, but a dataset must still survive (out of domain)
        kept = confirm_software_mentions(_StubDetector([[]]), body, [m])
        assert kept == [m]

    def test_software_candidate_dropped_when_no_overlap(self):
        body = "we used the Adam solver in this work " + "x" * 40
        m = _mention("Adam", "software", body.index("Adam"), body.index("Adam") + 4)
        # SOMD span is elsewhere in the window, not overlapping "Adam"
        far = SoftwareSpan("MATLAB", 0, 6, "Application_Usage", 0.9)
        kept = confirm_software_mentions(_StubDetector([[far]]), body, [m])
        assert kept == []

    def test_software_candidate_kept_when_overlap(self):
        body = "implemented in MATLAB for analysis " + "y" * 40
        start = body.index("MATLAB")
        m = _mention("MATLAB", "software", start, start + 6)
        # window is body[start-160 : end+160] clamped to [0, len); compute the
        # candidate's offset within that window the same way the detector sees it.
        win_lo = max(0, start - 160)
        span = SoftwareSpan("MATLAB", start - win_lo, start - win_lo + 6, "Application_Usage", 0.95)
        kept = confirm_software_mentions(_StubDetector([[span]]), body, [m])
        assert kept == [m]

    def test_empty_mentions_returns_empty(self):
        assert confirm_software_mentions(_StubDetector([]), "body", []) == []

    def test_mixed_batch_keeps_dataset_drops_unconfirmed_software(self):
        body = "the CIFAR10 dataset and the Adam solver were " + "z" * 60
        ds = _mention("CIFAR10", "dataset", body.index("CIFAR10"), body.index("CIFAR10") + 7)
        sw = _mention("Adam", "software", body.index("Adam"), body.index("Adam") + 4)
        kept = confirm_software_mentions(_StubDetector([[], []]), body, [ds, sw])
        assert ds in kept and sw not in kept
