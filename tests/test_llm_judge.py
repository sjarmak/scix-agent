"""Unit tests for the M9 LLM-judge stub and Cohen's kappa."""

from __future__ import annotations

import pytest

from scix.eval.llm_judge import (
    ALLOWED_LABELS,
    JudgeLabel,
    LinkRow,
    cohens_kappa,
    judge,
    labels_from_judge,
)

# ---------------------------------------------------------------------------
# judge() — stub path
# ---------------------------------------------------------------------------


class TestJudgeStub:
    def test_stub_returns_one_label_per_link(self) -> None:
        links = [
            LinkRow(tier=1, bibcode="2099BIB..001A", entity_id=1),
            LinkRow(tier=2, bibcode="2099BIB..002A", entity_id=2),
            LinkRow(tier=4, bibcode="2099BIB..003A", entity_id=3),
        ]
        out = judge(links)
        assert len(out) == 3
        assert all(isinstance(x, JudgeLabel) for x in out)

    def test_stub_labels_in_allowed_set(self) -> None:
        links = [LinkRow(tier=1, bibcode=f"BIB{i}", entity_id=i) for i in range(10)]
        out = judge(links)
        for lab in out:
            assert lab.label in ALLOWED_LABELS

    def test_stub_is_deterministic_and_passes_through_identity(self) -> None:
        links = [LinkRow(tier=1, bibcode="BIB_X", entity_id=7)]
        a = judge(links)
        b = judge(links)
        assert a[0].label == b[0].label
        assert a[0].bibcode == "BIB_X"
        assert a[0].entity_id == 7

    def test_use_real_without_api_key_falls_back_to_stub(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        links = [LinkRow(tier=1, bibcode="BIB_Y", entity_id=42)]
        out = judge(links, use_real=True)
        assert len(out) == 1
        assert out[0].label in ALLOWED_LABELS

    def test_empty_input_returns_empty(self) -> None:
        assert judge([]) == []


# ---------------------------------------------------------------------------
# cohens_kappa()
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement_returns_one(self) -> None:
        h = ["correct", "incorrect", "ambiguous", "correct"]
        j = ["correct", "incorrect", "ambiguous", "correct"]
        assert cohens_kappa(h, j) == pytest.approx(1.0, abs=1e-9)

    def test_total_disagreement_binary_returns_minus_one(self) -> None:
        h = ["correct", "incorrect", "correct", "incorrect"]
        j = ["incorrect", "correct", "incorrect", "correct"]
        # Balanced classes: p_o = 0, p_e = 0.5 → κ = -1
        assert cohens_kappa(h, j) == pytest.approx(-1.0, abs=1e-9)

    def test_classic_textbook_example(self) -> None:
        # Classic 2x2 from Wikipedia (Landis & Koch):
        #     j=yes  j=no
        # h=yes  20    5
        # h=no  10    15
        # n=50. p_o = (20+15)/50 = 0.70
        # p_yes_h = 25/50 = 0.5, p_yes_j = 30/50 = 0.6
        # p_no_h  = 25/50 = 0.5, p_no_j  = 20/50 = 0.4
        # p_e = 0.5*0.6 + 0.5*0.4 = 0.5
        # κ = (0.7 - 0.5) / (1 - 0.5) = 0.4
        h = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
        j = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
        assert cohens_kappa(h, j) == pytest.approx(0.4, abs=1e-9)

    def test_all_same_label_on_both_sides_returns_one(self) -> None:
        h = ["correct"] * 5
        j = ["correct"] * 5
        assert cohens_kappa(h, j) == 1.0

    def test_all_same_label_disagree_returns_zero(self) -> None:
        # Both raters collapsed to a single (but different) label each:
        # p_e = 1*0 + 0*1 = 0 → formula reduces to p_o itself (0).
        h = ["correct"] * 5
        j = ["incorrect"] * 5
        assert cohens_kappa(h, j) == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns_zero(self) -> None:
        assert cohens_kappa([], []) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cohens_kappa(["correct"], ["correct", "incorrect"])

    def test_labels_from_judge_helper(self) -> None:
        labels = [
            JudgeLabel(bibcode="B1", entity_id=1, label="correct"),
            JudgeLabel(bibcode="B2", entity_id=2, label="incorrect"),
        ]
        assert labels_from_judge(labels) == ["correct", "incorrect"]
