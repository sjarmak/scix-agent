#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Nightly NER canary — drift detection for nasa-smd-ibm-v0.1_NER_DEAL.
#
# Scheduling (comment only; not auto-installed):
#
# Cron:
#   0 3 * * * cd /home/ds/projects/scix_experiments && \
#     .venv/bin/python scripts/canary_ner.py >> logs/canary_ner/cron.log 2>&1
#
# systemd user timer (~/.config/systemd/user/canary-ner.timer):
#   [Unit]
#   Description=Nightly NER canary drift check
#
#   [Timer]
#   OnCalendar=*-*-* 03:00:00
#   Persistent=true
#
#   [Install]
#   WantedBy=timers.target
#
# Companion service unit (~/.config/systemd/user/canary-ner.service):
#   [Unit]
#   Description=NER canary drift check
#
#   [Service]
#   Type=oneshot
#   WorkingDirectory=/home/ds/projects/scix_experiments
#   ExecStart=/home/ds/projects/scix_experiments/.venv/bin/python \
#     scripts/canary_ner.py
#
# Enable with:  systemctl --user enable --now canary-ner.timer
#
# The script exits 0 when per-entity F1 drift stays within DRIFT_THRESHOLD
# (0.05) and 1 when any entity type exceeds it, making it cron-friendly:
# an alerting wrapper can trigger on non-zero exit.
# ---------------------------------------------------------------------------
"""Run the pinned NER model on a fixed reference set and detect F1 drift.

This script is the daily smoke test for the
``adsabs/nasa-smd-ibm-v0.1_NER_DEAL`` model used by the entity-extraction
pipeline. It loads the baseline per-entity F1 produced by
``scripts/eval_ner_wiesp.py`` (the ``ner-wiesp-eval`` work unit), runs
inference on 10 fixed reference snippets, computes per-entity F1, and
writes a timestamped log under ``logs/canary_ner/``. Non-zero exit
signals drift exceeding the configured threshold so cron/systemd can
alert.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL_NAME = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"
# TODO: must match MODEL_REVISION in scripts/eval_ner_wiesp.py. Refresh
# from the HF model page (commit list) when the eval script is repinned.
MODEL_REVISION = "87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d"

DRIFT_THRESHOLD = 0.05  # max allowed |current_f1 - baseline_f1| per entity
REFERENCE_FIXTURE = Path("tests/fixtures/canary_ner_reference.json")
BASELINE_PATH = Path("results/ner_wiesp_eval.json")
LOG_DIR = Path("logs/canary_ner")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FileFormatError(ValueError):
    """Raised when a JSON file exists but does not match the expected schema."""


# ---------------------------------------------------------------------------
# Data containers (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySpan:
    type: str
    span: tuple[int, int]
    text: str


@dataclass(frozen=True)
class ReferencePaper:
    id: str
    text: str
    gold_entities: tuple[EntitySpan, ...]


# ---------------------------------------------------------------------------
# Baseline + fixture loading
# ---------------------------------------------------------------------------


REQUIRED_BASELINE_TOP_KEYS = ("per_entity", "summary", "model_revision")
REQUIRED_PER_ENTITY_KEYS = ("precision", "recall", "f1")


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and validate the baseline F1 report.

    Raises:
        FileFormatError: if the file is absent or missing required keys.
    """
    if not path.exists():
        raise FileFormatError(
            f"Baseline report not found at {path}. Run scripts/eval_ner_wiesp.py "
            "(work unit ner-wiesp-eval) to produce it."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FileFormatError(
            f"Baseline at {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise FileFormatError(
            f"Baseline at {path} must be a JSON object, got {type(payload).__name__}"
        )

    missing = [k for k in REQUIRED_BASELINE_TOP_KEYS if k not in payload]
    if missing:
        raise FileFormatError(
            f"Baseline at {path} missing required keys: {missing}. "
            "Expected schema: {per_entity, summary, model_revision, generated_at}."
        )

    per_entity = payload.get("per_entity")
    if not isinstance(per_entity, dict) or not per_entity:
        raise FileFormatError(
            f"Baseline at {path} 'per_entity' must be a non-empty object."
        )

    for entity_type, metrics in per_entity.items():
        if not isinstance(metrics, dict):
            raise FileFormatError(
                f"Baseline at {path} per_entity[{entity_type!r}] must be an object."
            )
        missing_metric = [k for k in REQUIRED_PER_ENTITY_KEYS if k not in metrics]
        if missing_metric:
            raise FileFormatError(
                f"Baseline at {path} per_entity[{entity_type!r}] missing "
                f"{missing_metric}. Expected precision/recall/f1."
            )

    return payload


def load_reference(path: Path) -> list[ReferencePaper]:
    """Load the fixed reference fixture."""
    if not path.exists():
        raise FileFormatError(f"Reference fixture not found at {path}.")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "papers" not in payload:
        raise FileFormatError(
            f"Reference fixture at {path} must have a top-level 'papers' list."
        )

    papers_raw = payload["papers"]
    if not isinstance(papers_raw, list) or not papers_raw:
        raise FileFormatError(
            f"Reference fixture at {path} 'papers' must be a non-empty list."
        )

    papers: list[ReferencePaper] = []
    for row in papers_raw:
        entities = tuple(
            EntitySpan(
                type=e["type"],
                span=tuple(e["span"]),  # type: ignore[arg-type]
                text=e["text"],
            )
            for e in row.get("gold_entities", [])
        )
        papers.append(
            ReferencePaper(
                id=row["id"], text=row["text"], gold_entities=entities
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Model loading + inference (mockable)
# ---------------------------------------------------------------------------


def load_model(revision: str) -> Any:
    """Load the pinned HuggingFace NER pipeline.

    Kept minimal so tests can monkeypatch it. Returns an opaque bundle
    consumed by run_inference; the real implementation uses
    ``transformers.pipeline`` so upstream changes stay local.
    """
    # Deferred import — heavy dependency, only needed for the real run.
    from transformers import (  # type: ignore
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, revision=revision
    )
    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    return ner


def run_inference(
    model_bundle: Any | None,
    papers: Iterable[ReferencePaper],
    *,
    mock: bool = False,
) -> dict[str, tuple[EntitySpan, ...]]:
    """Run NER and return predicted entity spans keyed by paper id.

    When ``mock=True`` the model_bundle is ignored and gold entities are
    returned as predictions (i.e. perfect F1) — this is the default path
    used in unit tests and by the ``--mock-model`` CLI flag.
    """
    predictions: dict[str, tuple[EntitySpan, ...]] = {}
    for paper in papers:
        if mock or model_bundle is None:
            predictions[paper.id] = paper.gold_entities
            continue

        raw = model_bundle(paper.text)
        entities: list[EntitySpan] = []
        for item in raw:
            entity_type = item.get("entity_group") or item.get("entity") or ""
            start = int(item["start"])
            end = int(item["end"])
            entities.append(
                EntitySpan(
                    type=str(entity_type),
                    span=(start, end),
                    text=paper.text[start:end],
                )
            )
        predictions[paper.id] = tuple(entities)
    return predictions


# ---------------------------------------------------------------------------
# F1 + drift
# ---------------------------------------------------------------------------


def compute_per_entity_f1(
    predictions: dict[str, tuple[EntitySpan, ...]],
    papers: Iterable[ReferencePaper],
) -> dict[str, dict[str, float]]:
    """Span-level micro-F1 per entity type (exact (type, span) match)."""
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}

    def _key(span: EntitySpan) -> tuple[str, int, int]:
        return (span.type, span.span[0], span.span[1])

    for paper in papers:
        gold = {_key(s): s for s in paper.gold_entities}
        pred = {_key(s): s for s in predictions.get(paper.id, ())}

        gold_keys = set(gold.keys())
        pred_keys = set(pred.keys())

        for key in gold_keys & pred_keys:
            tp[key[0]] = tp.get(key[0], 0) + 1
        for key in pred_keys - gold_keys:
            fp[key[0]] = fp.get(key[0], 0) + 1
        for key in gold_keys - pred_keys:
            fn[key[0]] = fn.get(key[0], 0) + 1

    entity_types = set(tp) | set(fp) | set(fn)
    out: dict[str, dict[str, float]] = {}
    for etype in sorted(entity_types):
        t = tp.get(etype, 0)
        f_p = fp.get(etype, 0)
        f_n = fn.get(etype, 0)
        precision = t / (t + f_p) if (t + f_p) else 0.0
        recall = t / (t + f_n) if (t + f_n) else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        out[etype] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def compute_drift(
    current: dict[str, dict[str, float]],
    baseline_per_entity: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Per-entity-type drift relative to baseline.

    Only entity types present in the baseline are compared — new types
    in ``current`` are reported with baseline_f1=None and drift=None so
    they don't trip the threshold check.
    """
    drift: dict[str, dict[str, float]] = {}
    seen = set(current.keys()) | set(baseline_per_entity.keys())
    for etype in sorted(seen):
        cur = current.get(etype, {}).get("f1")
        base = baseline_per_entity.get(etype, {}).get("f1")
        if cur is not None and base is not None:
            drift[etype] = {
                "current_f1": float(cur),
                "baseline_f1": float(base),
                "drift": abs(float(cur) - float(base)),
            }
        else:
            drift[etype] = {
                "current_f1": float(cur) if cur is not None else None,  # type: ignore[dict-item]
                "baseline_f1": float(base) if base is not None else None,  # type: ignore[dict-item]
                "drift": None,  # type: ignore[dict-item]
            }
    return drift


def evaluate_drift(
    drift_map: dict[str, dict[str, float]], threshold: float
) -> tuple[float, bool]:
    """Return (max_drift, exceeded)."""
    comparable = [
        float(v["drift"])
        for v in drift_map.values()
        if v.get("drift") is not None
    ]
    if not comparable:
        return 0.0, False
    max_drift = max(comparable)
    return max_drift, max_drift > threshold


# ---------------------------------------------------------------------------
# Log writing
# ---------------------------------------------------------------------------


def write_log(
    log_dir: Path,
    *,
    current: dict[str, dict[str, float]],
    baseline: dict[str, Any],
    drift_map: dict[str, dict[str, float]],
    max_drift: float,
    exceeded: bool,
    threshold: float,
    paper_count: int,
    now: datetime | None = None,
) -> Path:
    """Write <YYYY-MM-DD>.json under log_dir; return the path."""
    stamp = now or datetime.now(timezone.utc)
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"{stamp.strftime('%Y-%m-%d')}.json"

    payload = {
        "generated_at": stamp.isoformat().replace("+00:00", "Z"),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "baseline_revision": baseline.get("model_revision"),
        "baseline_generated_at": baseline.get("generated_at"),
        "current": current,
        "per_entity": drift_map,
        "max_drift": max_drift,
        "threshold": threshold,
        "exceeded": exceeded,
        "paper_count": paper_count,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned NER model on the fixed reference set and detect "
            "F1 drift against the baseline produced by eval_ner_wiesp.py."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help=f"Path to baseline JSON (default: {BASELINE_PATH}).",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=REFERENCE_FIXTURE,
        help=f"Path to reference fixture (default: {REFERENCE_FIXTURE}).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help=f"Directory for daily log files (default: {LOG_DIR}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DRIFT_THRESHOLD,
        help=f"Max allowed per-entity F1 drift (default: {DRIFT_THRESHOLD}).",
    )
    parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Skip HF model load; use gold labels as predictions (for testing).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    baseline = load_baseline(args.baseline)
    papers = load_reference(args.fixture)
    logger.info(
        "Loaded baseline (%d entity types) + %d reference papers",
        len(baseline["per_entity"]),
        len(papers),
    )

    if args.mock_model:
        model_bundle: Any | None = None
    else:
        logger.info("Loading model %s@%s", MODEL_NAME, MODEL_REVISION)
        model_bundle = load_model(MODEL_REVISION)

    predictions = run_inference(model_bundle, papers, mock=args.mock_model)
    current = compute_per_entity_f1(predictions, papers)
    drift_map = compute_drift(current, baseline["per_entity"])
    max_drift, exceeded = evaluate_drift(drift_map, args.threshold)

    log_path = write_log(
        args.log_dir,
        current=current,
        baseline=baseline,
        drift_map=drift_map,
        max_drift=max_drift,
        exceeded=exceeded,
        threshold=args.threshold,
        paper_count=len(papers),
    )
    logger.info(
        "Wrote canary log to %s (max_drift=%.4f, exceeded=%s)",
        log_path,
        max_drift,
        exceeded,
    )
    return 1 if exceeded else 0


if __name__ == "__main__":
    sys.exit(main())
