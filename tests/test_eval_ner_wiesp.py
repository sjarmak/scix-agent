"""Unit tests for scripts/eval_ner_wiesp.py.

All tests are offline — HF model + dataset loaders are monkeypatched,
so no network access is required. F1 computation is exercised on tiny
synthetic BIO-tagged examples with hand-computed expected values.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Dynamically load scripts/eval_ner_wiesp.py as a module (scripts/ is not a package)
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "eval_ner_wiesp.py"
_spec = importlib.util.spec_from_file_location("eval_ner_wiesp", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
eval_ner_wiesp = importlib.util.module_from_spec(_spec)
sys.modules["eval_ner_wiesp"] = eval_ner_wiesp
_spec.loader.exec_module(eval_ner_wiesp)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestPinnedConstants:
    def test_model_revision_is_full_sha(self) -> None:
        """Acceptance criterion 3: SHA visible as module-level constant, not 'main'."""
        assert eval_ner_wiesp.MODEL_REVISION != "main"
        # 40-char hex SHA
        assert len(eval_ner_wiesp.MODEL_REVISION) == 40
        assert all(c in "0123456789abcdef" for c in eval_ner_wiesp.MODEL_REVISION.lower())

    def test_model_name_is_pinned_repo(self) -> None:
        assert eval_ner_wiesp.MODEL_NAME == "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"

    def test_dataset_is_wiesp(self) -> None:
        assert eval_ner_wiesp.DATASET_NAME == "adsabs/WIESP2022-NER"
        assert eval_ner_wiesp.DATASET_SPLIT == "test"


# ---------------------------------------------------------------------------
# BIO span extraction
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_simple_span(self) -> None:
        tags = ["B-Mission", "I-Mission", "O", "B-Instrument"]
        spans = eval_ner_wiesp._extract_entities(tags)
        assert spans == [("Mission", 0, 2), ("Instrument", 3, 4)]

    def test_all_outside(self) -> None:
        assert eval_ner_wiesp._extract_entities(["O", "O", "O"]) == []

    def test_stray_inside_treated_as_begin(self) -> None:
        tags = ["I-Mission", "I-Mission", "O"]
        assert eval_ner_wiesp._extract_entities(tags) == [("Mission", 0, 2)]

    def test_adjacent_different_types(self) -> None:
        tags = ["B-Mission", "B-Instrument"]
        assert eval_ner_wiesp._extract_entities(tags) == [
            ("Mission", 0, 1),
            ("Instrument", 1, 2),
        ]


# ---------------------------------------------------------------------------
# compute_metrics — F1 correctness on synthetic examples
# ---------------------------------------------------------------------------


class TestComputeMetricsPerfect:
    def test_identical_tags_f1_is_one(self) -> None:
        golds = [["B-Mission", "I-Mission", "O"], ["B-Instrument", "O", "B-Mission"]]
        preds = [["B-Mission", "I-Mission", "O"], ["B-Instrument", "O", "B-Mission"]]
        report = eval_ner_wiesp.compute_metrics(golds, preds)
        assert report.micro_f1 == 1.0
        assert report.macro_f1 == 1.0
        for s in report.per_entity:
            assert s.precision == 1.0
            assert s.recall == 1.0
            assert s.f1 == 1.0


class TestComputeMetricsPartial:
    def test_one_missed_one_spurious(self) -> None:
        # gold entities: Mission[0:2], Instrument[3:4], Mission[5:6]
        # pred entities: Mission[0:2]         (TP),
        #                Instrument[3:4] missed (FN),
        #                Instrument[5:6] spurious (FP) — pred says Instrument, gold says Mission
        # Mission gold[5:6] is not recovered either (FN for Mission).
        golds = [["B-Mission", "I-Mission", "O", "B-Instrument", "O", "B-Mission"]]
        preds = [["B-Mission", "I-Mission", "O", "O", "O", "B-Instrument"]]
        report = eval_ner_wiesp.compute_metrics(golds, preds)

        by_type = {s.entity_type: s for s in report.per_entity}
        # Mission: tp=1, fp=0, fn=1 -> p=1.0 r=0.5 f=0.6667
        assert by_type["Mission"].precision == pytest.approx(1.0)
        assert by_type["Mission"].recall == pytest.approx(0.5)
        assert by_type["Mission"].f1 == pytest.approx(2 / 3)
        assert by_type["Mission"].support == 2
        # Instrument: tp=0, fp=1, fn=1 -> p=0.0 r=0.0 f=0.0
        assert by_type["Instrument"].precision == pytest.approx(0.0)
        assert by_type["Instrument"].recall == pytest.approx(0.0)
        assert by_type["Instrument"].f1 == pytest.approx(0.0)
        assert by_type["Instrument"].support == 1

        # Micro: tp=1, fp=1, fn=2 -> p=0.5, r=1/3, f=0.4
        assert report.micro_precision == pytest.approx(0.5)
        assert report.micro_recall == pytest.approx(1 / 3)
        assert report.micro_f1 == pytest.approx(0.4)

        # Macro F1: (2/3 + 0.0) / 2 = 1/3
        assert report.macro_f1 == pytest.approx(1 / 3)

        # total_support: tp(1) + fn(2) = 3
        assert report.total_support == 3


class TestComputeMetricsEmptyPredictions:
    def test_all_outside_predictions(self) -> None:
        golds = [["B-Mission", "I-Mission", "O"]]
        preds = [["O", "O", "O"]]
        report = eval_ner_wiesp.compute_metrics(golds, preds)
        assert report.micro_precision == 0.0
        assert report.micro_recall == 0.0
        assert report.micro_f1 == 0.0
        # Schema still well-formed
        d = report.to_dict()
        assert "per_entity" in d
        assert "summary" in d
        assert "meta" in d
        assert d["summary"]["total_support"] == 1

    def test_both_empty(self) -> None:
        golds: list[list[str]] = [[], []]
        preds: list[list[str]] = [[], []]
        report = eval_ner_wiesp.compute_metrics(golds, preds)
        assert report.per_entity == ()
        assert report.micro_f1 == 0.0
        assert report.n_examples == 2


class TestComputeMetricsValidation:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            eval_ner_wiesp.compute_metrics([["O"]], [["O"], ["O"]])

    def test_per_example_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            eval_ner_wiesp.compute_metrics([["B-X", "O"]], [["B-X"]])


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class TestReportSchema:
    def test_has_required_top_level_keys(self) -> None:
        report = eval_ner_wiesp.compute_metrics(
            [["B-Mission", "O"]],
            [["B-Mission", "O"]],
            meta={"model_name": "x", "dataset": "y"},
        )
        d = report.to_dict()
        assert set(d.keys()) == {"per_entity", "summary", "meta"}
        assert d["meta"] == {"model_name": "x", "dataset": "y"}

    def test_summary_has_micro_and_macro(self) -> None:
        report = eval_ner_wiesp.compute_metrics([["B-Mission", "O"]], [["B-Mission", "O"]])
        summary = report.to_dict()["summary"]
        required = {
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "total_support",
            "n_examples",
        }
        assert required.issubset(summary.keys())

    def test_per_entity_entries_have_prf_support(self) -> None:
        report = eval_ner_wiesp.compute_metrics([["B-Mission", "O"]], [["B-Mission", "O"]])
        d = report.to_dict()
        for etype, scores in d["per_entity"].items():
            assert set(scores.keys()) == {"precision", "recall", "f1", "support"}
            assert isinstance(etype, str)


# ---------------------------------------------------------------------------
# Dataset loading from fixture
# ---------------------------------------------------------------------------


class TestLoadDatasetFixture:
    def test_loads_fixture_json(self, tmp_path: Path) -> None:
        fixture = tmp_path / "wiesp.json"
        fixture.write_text(
            json.dumps(
                [
                    {"tokens": ["M87"], "tags": ["B-CelestialObject"]},
                    {"tokens": ["Chandra", "observations"], "tags": ["B-Mission", "O"]},
                ]
            ),
            encoding="utf-8",
        )
        examples = eval_ner_wiesp.load_dataset(fixture_path=fixture)
        assert len(examples) == 2
        assert examples[0].tokens == ("M87",)
        assert examples[0].tags == ("B-CelestialObject",)
        assert examples[0].pred is None

    def test_fixture_with_pred_field(self, tmp_path: Path) -> None:
        fixture = tmp_path / "wiesp.json"
        fixture.write_text(
            json.dumps([{"tokens": ["M87"], "tags": ["B-CelestialObject"], "pred": ["O"]}]),
            encoding="utf-8",
        )
        examples = eval_ner_wiesp.load_dataset(fixture_path=fixture)
        assert examples[0].pred == ("O",)

    def test_sample_truncates(self, tmp_path: Path) -> None:
        fixture = tmp_path / "wiesp.json"
        fixture.write_text(
            json.dumps(
                [
                    {"tokens": ["a"], "tags": ["O"]},
                    {"tokens": ["b"], "tags": ["O"]},
                    {"tokens": ["c"], "tags": ["O"]},
                ]
            ),
            encoding="utf-8",
        )
        examples = eval_ner_wiesp.load_dataset(sample=2, fixture_path=fixture)
        assert len(examples) == 2

    def test_malformed_fixture_rejected(self, tmp_path: Path) -> None:
        fixture = tmp_path / "bad.json"
        fixture.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with pytest.raises(ValueError):
            eval_ner_wiesp.load_dataset(fixture_path=fixture)

    def test_token_tag_length_mismatch_rejected(self, tmp_path: Path) -> None:
        fixture = tmp_path / "bad.json"
        fixture.write_text(
            json.dumps([{"tokens": ["a", "b"], "tags": ["O"]}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            eval_ner_wiesp.load_dataset(fixture_path=fixture)


# ---------------------------------------------------------------------------
# Mocked inference: simulate transformers model behavior
# ---------------------------------------------------------------------------


class _FakeEncoding(dict):
    def __init__(self, word_ids: list[int | None]) -> None:
        super().__init__()
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return self._word_ids


class _FakeTokenizer:
    def __call__(
        self,
        words: list[str],
        is_split_into_words: bool = False,
        return_tensors: str = "pt",
        truncation: bool = False,
    ) -> _FakeEncoding:
        return _FakeEncoding(word_ids=list(range(len(words))))


class _FakeLogits:
    def __init__(self, ids_per_token: list[int]) -> None:
        self._ids = ids_per_token

    def argmax(self, dim: int = -1) -> "_FakeTensor":
        return _FakeTensor(self._ids)


class _FakeTensor:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[int]:
        return self._ids


class _FakeModelOutputs:
    def __init__(self, predicted_ids: list[int]) -> None:
        # outputs.logits is indexable [0] → per-token logits object that exposes argmax
        self.logits = [_FakeLogits(predicted_ids)]


class _FakeModel:
    def __init__(self, predicted_ids_by_example: list[list[int]]) -> None:
        self._preds = predicted_ids_by_example
        self._call_idx = 0
        self.config = SimpleNamespace(id2label={0: "O", 1: "B-Mission", 2: "I-Mission"})

    def eval(self) -> None:
        pass

    def __call__(self, **encoding: object) -> _FakeModelOutputs:
        ids = self._preds[self._call_idx]
        self._call_idx += 1
        return _FakeModelOutputs(ids)


class _FakeTorchNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _FakeTorchModule:
    @staticmethod
    def no_grad() -> _FakeTorchNoGrad:
        return _FakeTorchNoGrad()


class TestRunInferenceMocked:
    def test_returns_aligned_predicted_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock tokenizer + model to return fixed label ids; verify alignment."""
        fake_torch = _FakeTorchModule()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        tokenizer = _FakeTokenizer()
        # Example 1: "Chandra mission" → tags B-Mission, I-Mission
        # Example 2: "M87 observed"   → tags O, O
        model = _FakeModel(predicted_ids_by_example=[[1, 2], [0, 0]])
        id2label = {0: "O", 1: "B-Mission", 2: "I-Mission"}

        examples = [
            eval_ner_wiesp.Example(tokens=("Chandra", "mission"), tags=("O", "O")),
            eval_ner_wiesp.Example(tokens=("M87", "observed"), tags=("O", "O")),
        ]
        preds = eval_ner_wiesp.run_inference(examples, tokenizer, model, id2label)
        assert preds == [["B-Mission", "I-Mission"], ["O", "O"]]


class TestLoadModelAndTokenizerMocked:
    def test_pins_revision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify from_pretrained is called with the pinned SHA, not 'main'."""
        calls: list[tuple[str, str]] = []

        class _FakeTok:
            @classmethod
            def from_pretrained(cls, name: str, revision: str | None = None) -> "_FakeTok":
                calls.append(("tok", revision or ""))
                return cls()

        class _FakeAutoModel:
            @classmethod
            def from_pretrained(cls, name: str, revision: str | None = None) -> "_FakeAutoModel":
                calls.append(("model", revision or ""))
                instance = cls()
                instance.config = SimpleNamespace(id2label={0: "O"})
                return instance

        fake_transformers = SimpleNamespace(
            AutoTokenizer=_FakeTok,
            AutoModelForTokenClassification=_FakeAutoModel,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        tok, model, id2label = eval_ner_wiesp.load_model_and_tokenizer()

        assert len(calls) == 2
        assert all(c[1] == eval_ner_wiesp.MODEL_REVISION for c in calls)
        assert id2label == {0: "O"}


# ---------------------------------------------------------------------------
# End-to-end: main() writes a well-formed JSON report from a fixture
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    def test_main_writes_report_from_fixture(self, tmp_path: Path) -> None:
        fixture = tmp_path / "wiesp.json"
        fixture.write_text(
            json.dumps(
                [
                    {
                        "tokens": ["Chandra", "observed", "M87"],
                        "tags": ["B-Mission", "O", "B-CelestialObject"],
                        "pred": ["B-Mission", "O", "B-CelestialObject"],
                    },
                    {
                        "tokens": ["Hubble", "is", "a", "telescope"],
                        "tags": ["B-Mission", "O", "O", "B-Instrument"],
                        "pred": ["B-Mission", "O", "O", "O"],
                    },
                ]
            ),
            encoding="utf-8",
        )
        output = tmp_path / "out.json"
        rc = eval_ner_wiesp.main(["--fixture", str(fixture), "--output", str(output)])
        assert rc == 0
        assert output.exists()

        report = json.loads(output.read_text(encoding="utf-8"))
        assert "per_entity" in report
        assert "summary" in report
        assert "meta" in report
        assert report["meta"]["model_name"] == eval_ner_wiesp.MODEL_NAME
        assert report["meta"]["model_revision"] == eval_ner_wiesp.MODEL_REVISION
        assert report["meta"]["predictions_source"] == "fixture"
        assert report["summary"]["n_examples"] == 2
        # Mission: 2/2 tp -> f1=1.0; Instrument: 0/1 tp -> f1=0.0; CelestialObject 1/1 -> f1=1.0
        by_type = report["per_entity"]
        assert by_type["Mission"]["f1"] == pytest.approx(1.0)
        assert by_type["CelestialObject"]["f1"] == pytest.approx(1.0)
        assert by_type["Instrument"]["f1"] == pytest.approx(0.0)

    def test_main_respects_env_fixture_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture = tmp_path / "wiesp.json"
        fixture.write_text(
            json.dumps([{"tokens": ["M87"], "tags": ["O"], "pred": ["O"]}]),
            encoding="utf-8",
        )
        output = tmp_path / "out.json"
        monkeypatch.setenv("WIESP_TEST_FIXTURE", str(fixture))
        rc = eval_ner_wiesp.main(["--output", str(output)])
        assert rc == 0
        assert output.exists()
