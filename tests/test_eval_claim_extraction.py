"""Tests for ``scripts/eval_claim_extraction.py``.

Covers:
  - exact-span matching (claim_text reworded but span+type identical -> TP)
  - Jaccard fallback (shifted span, same type, Jaccard >= 0.6 -> TP)
  - end-to-end stub run yielding P=R=F1=1.0 with all documented JSON keys
  - end-to-end partial-overlap case with hand-computed metrics
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_claim_extraction.py"

# Make src/ importable for the script's own ``from scix.claims.extract import ...``.
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_module():
    """Import scripts/eval_claim_extraction.py as a module under a stable name."""
    spec = importlib.util.spec_from_file_location(
        "eval_claim_extraction_mod", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def evalmod():
    return _load_module()


# ---------------------------------------------------------------------------
# Unit: matching primitives
# ---------------------------------------------------------------------------


def test_jaccard_basic(evalmod):
    """Sanity: identical strings -> 1.0; disjoint -> 0.0; overlap -> in (0, 1)."""
    assert evalmod.jaccard("a b c", "a b c") == 1.0
    assert evalmod.jaccard("a b", "c d") == 0.0
    # tokens {a,b,c} vs {a,b,d} -> intersection 2, union 4 -> 0.5
    assert evalmod.jaccard("a b c", "a b d") == pytest.approx(0.5)


def test_exact_span_match_counts_as_tp_regardless_of_text(evalmod):
    """Predicted text differs from gold text, but span+type match -> TP."""
    gold = [
        {
            "claim_text": "Hubble constant measured at 73.04 km/s/Mpc",
            "claim_type": "factual",
            "char_span_start": 100,
            "char_span_end": 150,
        }
    ]
    predicted = [
        {
            # Completely different wording — would fail Jaccard alone.
            "claim_text": "completely unrelated wording xyzzy",
            "claim_type": "factual",
            "char_span_start": 100,
            "char_span_end": 150,
        }
    ]
    tp, fp, fn = evalmod.match_claims(predicted, gold)
    assert (tp, fp, fn) == (1, 0, 0)


def test_exact_span_but_wrong_type_is_not_tp(evalmod):
    """Same span, different claim_type -> still FP/FN."""
    gold = [
        {
            "claim_text": "x",
            "claim_type": "factual",
            "char_span_start": 0,
            "char_span_end": 5,
        }
    ]
    predicted = [
        {
            "claim_text": "x",
            "claim_type": "speculative",  # wrong type
            "char_span_start": 0,
            "char_span_end": 5,
        }
    ]
    tp, fp, fn = evalmod.match_claims(predicted, gold)
    assert (tp, fp, fn) == (0, 1, 1)


def test_jaccard_fallback_with_shifted_span(evalmod):
    """Predicted has different span but Jaccard >= 0.6 + same type -> TP."""
    gold_text = "the cnn classifier achieves 94.2 percent accuracy on galaxy zoo"
    pred_text = "the cnn classifier achieves 94.2 percent accuracy on galaxy zoo test set"
    # token sets:
    #   gold  = {the, cnn, classifier, achieves, 94, 2, percent, accuracy, on,
    #            galaxy, zoo}                                            -> 11 tokens
    #   pred  = gold_tokens | {test, set}                                 -> 13 tokens
    # intersection = 11, union = 13 -> 11/13 ~= 0.846 >= 0.6 -> match.
    gold = [
        {
            "claim_text": gold_text,
            "claim_type": "factual",
            "char_span_start": 50,
            "char_span_end": 100,
        }
    ]
    predicted = [
        {
            "claim_text": pred_text,
            "claim_type": "factual",
            # span shifted (off by ~10 chars) so tier-1 fails
            "char_span_start": 40,
            "char_span_end": 105,
        }
    ]
    tp, fp, fn = evalmod.match_claims(predicted, gold)
    assert (tp, fp, fn) == (1, 0, 0)


def test_each_predicted_satisfies_at_most_one_gold(evalmod):
    """Greedy first-match: a single predicted cannot be reused for two golds."""
    g1 = {
        "claim_text": "alpha beta gamma",
        "claim_type": "factual",
        "char_span_start": 0,
        "char_span_end": 10,
    }
    g2 = {
        "claim_text": "alpha beta gamma",
        "claim_type": "factual",
        "char_span_start": 0,
        "char_span_end": 10,
    }
    p1 = {
        "claim_text": "alpha beta gamma",
        "claim_type": "factual",
        "char_span_start": 0,
        "char_span_end": 10,
    }
    tp, fp, fn = evalmod.match_claims([p1], [g1, g2])
    assert tp == 1
    assert fp == 0
    assert fn == 1


def test_compute_metrics_zero_denominators(evalmod):
    """0/0 must yield 0.0 (no divide-by-zero)."""
    out = evalmod.compute_metrics(0, 0, 0)
    assert out == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---------------------------------------------------------------------------
# End-to-end: stub yielding gold gives P=R=F1=1.0 with all keys present
# ---------------------------------------------------------------------------


def test_end_to_end_stub_returns_gold_yields_perfect_score(tmp_path, evalmod):
    output_path = tmp_path / "eval_out.json"
    rc = evalmod.main(
        [
            "--gold-path",
            str(REPO_ROOT / "eval" / "claim_extraction_gold_standard.jsonl"),
            "--output",
            str(output_path),
            "--llm",
            "stub",
            "--model-name",
            "stub-test",
            "--prompt-version",
            "claim_extraction_v1",
        ]
    )
    assert rc == 0
    assert output_path.is_file()

    data = json.loads(output_path.read_text())

    # All documented top-level keys present.
    expected_keys = {
        "model",
        "prompt_version",
        "n_paragraphs",
        "n_gold_claims",
        "n_predicted",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "per_discipline",
    }
    assert expected_keys.issubset(data.keys())

    # Sanity-check global metrics.
    assert data["precision"] == pytest.approx(1.0)
    assert data["recall"] == pytest.approx(1.0)
    assert data["f1"] == pytest.approx(1.0)
    assert data["fp"] == 0
    assert data["fn"] == 0
    assert data["tp"] == data["n_gold_claims"]
    assert data["model"] == "stub-test"
    assert data["prompt_version"] == "claim_extraction_v1"

    # Per-discipline coverage and shape.
    per_disc = data["per_discipline"]
    assert isinstance(per_disc, dict) and per_disc, "per_discipline must be a non-empty dict"
    for disc, metrics in per_disc.items():
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            assert key in metrics, f"per_discipline[{disc!r}] missing {key!r}"
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end: partial overlap with hand-computed metrics
# ---------------------------------------------------------------------------


def test_end_to_end_partial_overlap_metrics(tmp_path, evalmod):
    """3 gold, 2 predicted, 1 TP, 1 FP, 2 FN -> P=0.5, R=1/3, F1=0.4."""
    paragraph = "alpha beta gamma delta epsilon zeta eta theta iota kappa"

    # Three gold claims; only one will be matchable by the stub's prediction.
    gold_entry = {
        "bibcode": "GOLDTEST...001",
        "section_index": 0,
        "paragraph_index": 0,
        "paragraph_text": paragraph,
        "expected_claims": [
            {
                "claim_text": "alpha beta gamma",
                "claim_type": "factual",
                "subject": "a",
                "predicate": "b",
                "object": "c",
                "char_span_start": 0,
                "char_span_end": 16,  # 'alpha beta gamma'
            },
            {
                "claim_text": "delta epsilon zeta",
                "claim_type": "methodological",
                "subject": "d",
                "predicate": "e",
                "object": "f",
                "char_span_start": 17,
                "char_span_end": 35,
            },
            {
                "claim_text": "eta theta iota",
                "claim_type": "speculative",
                "subject": "g",
                "predicate": "h",
                "object": "i",
                "char_span_start": 36,
                "char_span_end": 50,
            },
        ],
        "discipline": "astrophysics",
    }

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps(gold_entry) + "\n")

    # Predictions: one matches gold[0] exactly (TP); one is a totally
    # unrelated claim with a non-overlapping span and disjoint tokens (FP).
    predicted = [
        {
            "claim_text": "alpha beta gamma",
            "claim_type": "factual",
            "char_span_start": 0,
            "char_span_end": 16,
        },
        {
            # Neither span match nor Jaccard match for any gold.
            "claim_text": "completely unrelated noise xyzzy plugh",
            "claim_type": "factual",
            "char_span_start": 200,
            "char_span_end": 220,
        },
    ]

    # Build a stub LLM that always returns these predictions.
    class _Stub:
        def __init__(self, claims):
            self._claims = claims
            self.calls = []

        def extract(self, prompt, paragraph):
            self.calls.append((prompt, paragraph))
            return [dict(c) for c in self._claims]

    stub = _Stub(predicted)

    gold_entries = evalmod.load_gold(gold_path)
    result = evalmod.run_eval(
        gold_entries,
        stub,
        model_name="hand-tuned",
        prompt_version="claim_extraction_v1",
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 2
    assert result["n_gold_claims"] == 3
    assert result["n_predicted"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1 / 3)
    # F1 = 2 * 0.5 * (1/3) / (0.5 + 1/3) = (1/3) / (5/6) = 2/5 = 0.4
    assert result["f1"] == pytest.approx(0.4)

    # Per-discipline carries the same numbers since there's only one entry.
    assert "astrophysics" in result["per_discipline"]
    disc = result["per_discipline"]["astrophysics"]
    assert disc["tp"] == 1
    assert disc["fp"] == 1
    assert disc["fn"] == 2
    assert disc["precision"] == pytest.approx(0.5)
    assert disc["recall"] == pytest.approx(1 / 3)
    assert disc["f1"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Smoke: --help works (exits 0, prints usage)
# ---------------------------------------------------------------------------


def test_help_exits_zero(evalmod, capsys):
    with pytest.raises(SystemExit) as exc_info:
        evalmod._parse_args(["--help"])
    # argparse exits 0 on --help
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()
