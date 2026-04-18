#!/usr/bin/env python3
"""Evaluate the pinned nasa-smd-ibm-v0.1_NER_DEAL checkpoint on WIESP2022-NER.

Loads the HuggingFace model at a pinned commit SHA (no ``main`` drift),
runs token-classification inference over the ``adsabs/WIESP2022-NER``
``test`` split, and reports per-entity-type precision / recall / F1 plus
micro / macro summaries. Output is written to ``results/ner_wiesp_eval.json``
by default.

The script is offline-safe for tests: when a fixture JSON path is supplied
via ``--fixture`` or the ``WIESP_TEST_FIXTURE`` env var, the HuggingFace
model and dataset loaders are not called at all. Every example in the
fixture may optionally include a ``pred`` field, in which case inference
is skipped entirely and the fixture tags drive the metrics computation.

To refresh the pinned commit, visit
https://huggingface.co/adsabs/nasa-smd-ibm-v0.1_NER_DEAL/commits/main
and paste the latest full SHA into ``MODEL_REVISION`` below.

Usage::

    python scripts/eval_ner_wiesp.py
    python scripts/eval_ner_wiesp.py --sample 100 --output /tmp/ner.json
    WIESP_TEST_FIXTURE=tests/data/wiesp_tiny.json python scripts/eval_ner_wiesp.py

Acceptance criteria are documented in the PRD work unit
``ner-wiesp-eval``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Ensure src/ is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned model + dataset identifiers
# ---------------------------------------------------------------------------

#: HuggingFace repo id for the NASA SMD IBM NER_DEAL checkpoint.
MODEL_NAME: str = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"

#: Full commit SHA on ``main`` at the time this script was pinned.
#: Never use "main" — drift silently breaks reproducibility.
MODEL_REVISION: str = "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"

#: HuggingFace dataset identifier for the evaluation corpus.
DATASET_NAME: str = "adsabs/WIESP2022-NER"

#: Split of the dataset used for evaluation.
DATASET_SPLIT: str = "test"

#: Default output path for the JSON report.
DEFAULT_OUTPUT: Path = Path(__file__).resolve().parent.parent / "results" / "ner_wiesp_eval.json"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """Single NER example: token list, gold BIO tags, optional predicted tags."""

    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    pred: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PerEntityScore:
    """Precision / recall / F1 / support for a single entity type."""

    entity_type: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MetricsReport:
    """Top-level metrics report written to JSON."""

    per_entity: tuple[PerEntityScore, ...]
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_support: int
    n_examples: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_entity": {
                s.entity_type: {
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "support": s.support,
                }
                for s in self.per_entity
            },
            "summary": {
                "micro_precision": self.micro_precision,
                "micro_recall": self.micro_recall,
                "micro_f1": self.micro_f1,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall,
                "macro_f1": self.macro_f1,
                "total_support": self.total_support,
                "n_examples": self.n_examples,
            },
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# BIO entity extraction (pure Python, no seqeval dependency required)
# ---------------------------------------------------------------------------


def _extract_entities(tags: Sequence[str]) -> list[tuple[str, int, int]]:
    """Extract (entity_type, start, end) spans from a BIO-tagged sequence.

    Treats any tag not starting with ``B-`` or ``I-`` as outside (``O``).
    Spans are half-open: ``[start, end)``.
    """
    entities: list[tuple[str, int, int]] = []
    cur_type: str | None = None
    cur_start: int | None = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur_type is not None and cur_start is not None:
                entities.append((cur_type, cur_start, i))
            cur_type = tag[2:]
            cur_start = i
        elif tag.startswith("I-"):
            etype = tag[2:]
            if cur_type == etype and cur_start is not None:
                continue
            # Treat stray I- as the start of a new span (robust to malformed tags)
            if cur_type is not None and cur_start is not None:
                entities.append((cur_type, cur_start, i))
            cur_type = etype
            cur_start = i
        else:
            if cur_type is not None and cur_start is not None:
                entities.append((cur_type, cur_start, i))
            cur_type = None
            cur_start = None

    if cur_type is not None and cur_start is not None:
        entities.append((cur_type, cur_start, len(tags)))
    return entities


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 with safe divide-by-zero handling."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Public: metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    golds: Sequence[Sequence[str]],
    preds: Sequence[Sequence[str]],
    *,
    n_examples: int | None = None,
    meta: dict[str, Any] | None = None,
) -> MetricsReport:
    """Compute per-entity-type and summary metrics from BIO-tagged sequences.

    Args:
        golds: Sequence of gold BIO tag lists, one per example.
        preds: Sequence of predicted BIO tag lists, one per example.
        n_examples: Optional override for ``n_examples`` in the report (defaults to ``len(golds)``).
        meta: Extra metadata attached to the report.

    Returns:
        A :class:`MetricsReport` with per-entity and summary P/R/F1.

    Raises:
        ValueError: If ``golds`` and ``preds`` have different lengths or if any
            per-example tag lists differ in length.
    """
    if len(golds) != len(preds):
        raise ValueError(f"golds ({len(golds)}) / preds ({len(preds)}) length mismatch")

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    entity_types: set[str] = set()

    for gold_tags, pred_tags in zip(golds, preds):
        if len(gold_tags) != len(pred_tags):
            raise ValueError(
                f"per-example length mismatch: gold={len(gold_tags)} pred={len(pred_tags)}"
            )
        gold_spans = set(_extract_entities(gold_tags))
        pred_spans = set(_extract_entities(pred_tags))

        for span in gold_spans:
            entity_types.add(span[0])
        for span in pred_spans:
            entity_types.add(span[0])

        for span in gold_spans & pred_spans:
            tp[span[0]] += 1
        for span in pred_spans - gold_spans:
            fp[span[0]] += 1
        for span in gold_spans - pred_spans:
            fn[span[0]] += 1

    per_entity: list[PerEntityScore] = []
    macro_p_sum = 0.0
    macro_r_sum = 0.0
    macro_f_sum = 0.0
    total_tp = total_fp = total_fn = 0

    for etype in sorted(entity_types):
        p, r, f = _prf(tp[etype], fp[etype], fn[etype])
        support = tp[etype] + fn[etype]
        per_entity.append(
            PerEntityScore(entity_type=etype, precision=p, recall=r, f1=f, support=support)
        )
        macro_p_sum += p
        macro_r_sum += r
        macro_f_sum += f
        total_tp += tp[etype]
        total_fp += fp[etype]
        total_fn += fn[etype]

    micro_p, micro_r, micro_f = _prf(total_tp, total_fp, total_fn)
    n = len(per_entity) if per_entity else 1
    macro_p = macro_p_sum / n if per_entity else 0.0
    macro_r = macro_r_sum / n if per_entity else 0.0
    macro_f = macro_f_sum / n if per_entity else 0.0

    return MetricsReport(
        per_entity=tuple(per_entity),
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        total_support=total_tp + total_fn,
        n_examples=n_examples if n_examples is not None else len(golds),
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# Model + dataset loaders (lazy, import guarded)
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(revision: str = MODEL_REVISION) -> tuple[Any, Any, dict[int, str]]:
    """Load tokenizer + token-classification model at a pinned commit.

    Import of ``transformers`` is deferred so this module can be imported
    in offline test environments that don't ship torch/transformers.

    Args:
        revision: Commit SHA to pin. Default is :data:`MODEL_REVISION`.

    Returns:
        Tuple of ``(tokenizer, model, id2label)``.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: WPS433

    logger.info("Loading %s @ %s", MODEL_NAME, revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, revision=revision)
    id2label = getattr(model.config, "id2label", {}) or {}
    return tokenizer, model, dict(id2label)


def load_dataset(
    sample: int = 0,
    fixture_path: str | os.PathLike[str] | None = None,
) -> list[Example]:
    """Load the evaluation dataset, either from HuggingFace or from a local fixture.

    The fixture path takes precedence. Fixture format is a JSON array of
    objects with ``tokens``, ``tags``, and optional ``pred`` fields.

    Args:
        sample: If >0, truncate to this many examples.
        fixture_path: Local JSON fixture to read instead of HF datasets.

    Returns:
        List of :class:`Example`.
    """
    if fixture_path is not None:
        path = Path(fixture_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Fixture {path} must be a JSON array")
        examples: list[Example] = []
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError("Each fixture row must be an object")
            tokens = tuple(row["tokens"])
            tags = tuple(row["tags"])
            if len(tokens) != len(tags):
                raise ValueError(f"tokens/tags length mismatch in fixture row: {row}")
            pred_raw = row.get("pred")
            pred = tuple(pred_raw) if pred_raw is not None else None
            if pred is not None and len(pred) != len(tokens):
                raise ValueError(f"tokens/pred length mismatch in fixture row: {row}")
            examples.append(Example(tokens=tokens, tags=tags, pred=pred))
        if sample and sample > 0:
            examples = examples[:sample]
        logger.info("Loaded %d examples from fixture %s", len(examples), path)
        return examples

    from datasets import load_dataset as hf_load_dataset  # noqa: WPS433

    logger.info("Loading %s[%s] from HuggingFace", DATASET_NAME, DATASET_SPLIT)
    ds = hf_load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    tag_names = _resolve_tag_names(ds)

    examples = []
    for i, row in enumerate(ds):
        if sample and i >= sample:
            break
        tokens = tuple(row["tokens"])
        raw_tags = row.get("ner_tags", row.get("tags"))
        if raw_tags is None:
            raise KeyError("WIESP row must have 'ner_tags' or 'tags'")
        tags = tuple(tag_names[t] if isinstance(t, int) else str(t) for t in raw_tags)
        examples.append(Example(tokens=tokens, tags=tags))
    return examples


def _resolve_tag_names(ds: Any) -> list[str]:
    """Extract the BIO label names from a HuggingFace dataset feature."""
    features = getattr(ds, "features", {}) or {}
    for key in ("ner_tags", "tags"):
        feat = features.get(key)
        if feat is None:
            continue
        seq_feature = getattr(feat, "feature", None)
        names = getattr(seq_feature, "names", None)
        if names:
            return list(names)
    return []


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    examples: Sequence[Example],
    tokenizer: Any,
    model: Any,
    id2label: dict[int, str],
) -> list[list[str]]:
    """Run token-classification inference and return per-token predicted BIO tags.

    Uses word-level alignment: for each word, the label of its first
    subword token is taken. Handles ``tokenizer(..., is_split_into_words=True)``
    interfaces that expose ``word_ids()``.

    Args:
        examples: The examples to predict on.
        tokenizer: A HuggingFace tokenizer instance.
        model: A HuggingFace token-classification model.
        id2label: Mapping from label id to string label.

    Returns:
        List of predicted tag lists, one per example, aligned to input tokens.
    """
    import torch  # noqa: WPS433

    predictions: list[list[str]] = []
    model.eval()

    for ex in examples:
        encoding = tokenizer(
            list(ex.tokens),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits[0]
        predicted_ids = logits.argmax(dim=-1).tolist()

        word_ids = encoding.word_ids(batch_index=0)
        per_word: list[str] = ["O"] * len(ex.tokens)
        seen: set[int] = set()
        for subword_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if wid < len(per_word):
                per_word[wid] = id2label.get(predicted_ids[subword_idx], "O")
        predictions.append(per_word)

    return predictions


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(report: MetricsReport, output_path: str | os.PathLike[str]) -> Path:
    """Write a :class:`MetricsReport` to JSON. Returns the path written."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote report to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument("--sample", type=int, default=0, help="Limit dataset rows (0 = all)")
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Local fixture JSON path; overrides HF dataset download",
    )
    p.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model loading (fixture must provide 'pred' for every example)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    fixture = args.fixture
    env_fixture = os.environ.get("WIESP_TEST_FIXTURE")
    if fixture is None and env_fixture:
        fixture = Path(env_fixture)

    examples = load_dataset(sample=args.sample, fixture_path=fixture)
    if not examples:
        logger.error("No examples loaded; refusing to write empty report")
        return 1

    preds_from_fixture = all(ex.pred is not None for ex in examples)

    if preds_from_fixture:
        logger.info("All fixture rows supply 'pred'; skipping model inference")
        predictions = [list(ex.pred) for ex in examples]  # type: ignore[arg-type]
    elif args.no_model:
        raise RuntimeError("--no-model requires every example to provide 'pred'")
    else:
        tokenizer, model, id2label = load_model_and_tokenizer()
        predictions = run_inference(examples, tokenizer, model, id2label)

    golds = [list(ex.tags) for ex in examples]
    meta = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "source": "fixture" if fixture else "huggingface",
        "predictions_source": "fixture" if preds_from_fixture else "model",
    }
    if fixture is not None or preds_from_fixture:
        meta["_note"] = (
            "Fixture-based placeholder report. Regenerate on a host with GPU "
            "and network access by running: "
            "`python scripts/eval_ner_wiesp.py --output results/ner_wiesp_eval.json` "
            "(no --fixture flag) to evaluate the pinned model on the real "
            "WIESP2022-NER test split."
        )
    report = compute_metrics(golds, predictions, n_examples=len(examples), meta=meta)
    write_report(report, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
