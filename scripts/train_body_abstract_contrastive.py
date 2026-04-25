#!/usr/bin/env python3
"""Body↔abstract contrastive fine-tune (S1 of prd_full_text_applications_v2.md).

Pilot fine-tune of a sentence-encoder using
(abstract, random body paragraph from the same paper) as positive pairs.
MultipleNegativesRankingLoss treats the other abstracts in the batch as
implicit negatives, so negatives are not constructed explicitly.

Operational deferral
--------------------
The full 100K pilot is **GPU-window deferred** — see
``docs/runbooks/train_body_abstract_contrastive.md``. This script ships
production-ready and CI-tested via a smoke path that uses 32 synthetic
pairs and a tiny model (~30s on CPU). The real run is wrapped with
``scix-batch`` to stay friendly with the gascity OOM rules.

Usage (smoke / CPU)
-------------------
.. code-block:: bash

    .venv/bin/python scripts/train_body_abstract_contrastive.py \\
        --base-model minilm-test \\
        --cohort-strategy random \\
        --cohort-size 32 \\
        --max-steps 5 \\
        --synthetic-pairs 32 \\
        --output-model-dir /tmp/body_contrastive_smoke

Usage (real pilot, GPU window)
------------------------------
.. code-block:: bash

    scix-batch --mem-high 25G --mem-max 40G \\
        .venv/bin/python scripts/train_body_abstract_contrastive.py \\
            --base-model indus \\
            --cohort-strategy stratified-by-arxiv-class \\
            --cohort-size 100000 \\
            --output-model-dir models/body_contrastive_indus_$(date +%Y%m%d) \\
            --batch-size 64 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("body_contrastive")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COVERAGE_BIAS_JSON = REPO_ROOT / "results" / "full_text_coverage_bias.json"

BASE_MODELS: dict[str, str] = {
    "indus": "nasa-impact/nasa-smd-ibm-st-v2",
    "specter2": "allenai/specter2_base",
    # Tiny model used by the smoke test. Not a production option.
    "minilm-test": "sentence-transformers/all-MiniLM-L6-v2",
}

COHORT_STRATEGIES = ("stratified-by-arxiv-class", "random")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortAllocation:
    """Per-stratum target count from the cohort planner."""

    label: str
    n: int
    weight: float


# ---------------------------------------------------------------------------
# Coverage-bias I/O (M1 dependency)
# ---------------------------------------------------------------------------


def load_coverage_bias(path: str | Path) -> dict[str, Any]:
    """Load M1's ``full_text_coverage_bias.json`` payload.

    Raises FileNotFoundError if the file is missing — the caller is
    responsible for deciding whether that is fatal (it is for
    ``stratified-by-arxiv-class``, optional for ``random``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Coverage-bias JSON not found at {path}. "
            "Run scripts/report_full_text_coverage_bias.py first "
            "(or pass --cohort-strategy random)."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Cohort planning
# ---------------------------------------------------------------------------


def build_cohort_plan(
    coverage: dict[str, Any] | None,
    cohort_size: int,
    strategy: str,
    seed: int = 13,
) -> list[CohortAllocation]:
    """Translate ``--cohort-strategy`` into per-stratum allocations.

    Stratified mode samples proportional to the *corpus prior*
    (``q_corpus``) — not the full-text-skewed ``p_fulltext`` — so the
    trained model does not over-fit the body-coverage bias documented
    in M1.
    """
    if cohort_size <= 0:
        raise ValueError(f"cohort_size must be positive, got {cohort_size}")
    if strategy not in COHORT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of {COHORT_STRATEGIES}"
        )

    if strategy == "random":
        return [CohortAllocation(label="random", n=cohort_size, weight=1.0)]

    # stratified-by-arxiv-class
    if coverage is None:
        raise ValueError(
            "stratified-by-arxiv-class strategy requires a coverage payload"
        )
    rows = (
        coverage.get("facets", {})
        .get("arxiv_class", {})
        .get("rows", [])
    )
    if not rows:
        raise ValueError(
            "Coverage payload missing facets.arxiv_class.rows; cannot stratify"
        )

    weights = [float(r.get("q_corpus", 0.0)) for r in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("All q_corpus weights are zero; cannot stratify")

    raw = [w / total_weight for w in weights]
    counts = [int(round(cohort_size * p)) for p in raw]

    drift = cohort_size - sum(counts)
    if drift != 0 and counts:
        # Apply rounding drift to the largest bucket so totals match exactly.
        idx = max(range(len(counts)), key=lambda i: counts[i])
        counts[idx] += drift

    rng = random.Random(seed)  # reserved for future use (e.g. shuffling)
    _ = rng.random()

    return [
        CohortAllocation(label=str(row["label"]), n=int(n), weight=p)
        for row, n, p in zip(rows, counts, raw, strict=True)
        if n > 0
    ]


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def make_synthetic_pairs(n: int, seed: int = 13) -> list[Any]:
    """Build N synthetic (abstract, body-paragraph) pairs for the smoke test.

    Each pair shares a topic-keyword (``topic_<i>``) so the encoder can
    learn the alignment with very few steps. Returns
    ``sentence_transformers.InputExample`` objects but the import is
    deferred so callers without the optional dep can still import this
    module.
    """
    from sentence_transformers import InputExample

    rng = random.Random(seed)
    # Pool of distinguishable keyword tokens. We pair each example to a
    # unique token so the contrastive signal is unambiguous: an abstract
    # mentioning ``alpha`` should be closer to the body that mentions
    # ``alpha`` than to bodies for ``beta`` / ``gamma`` / ... in the same
    # batch. This is the recipe MNRL is designed for.
    keywords = [f"keyword_{i:04d}" for i in range(max(n, 1))]
    rng.shuffle(keywords)
    examples: list[Any] = []
    for i in range(n):
        kw = keywords[i]
        abstract = (
            f"This abstract studies {kw}. We measure {kw} across multiple "
            f"datasets and report results on the {kw} benchmark."
        )
        body = (
            f"In this body paragraph we describe the {kw} method in detail. "
            f"The {kw} pipeline operates on the {kw} corpus and produces "
            f"{kw}-shaped outputs."
        )
        examples.append(InputExample(texts=[abstract, body]))
    return examples


# ---------------------------------------------------------------------------
# Loss capture (lets tests assert that the loss actually decreases)
# ---------------------------------------------------------------------------


def make_loss_capture(model: Any) -> Any:
    """Return a ``MultipleNegativesRankingLoss`` that records each step's loss.

    Defined as a factory (not a top-level class) so importing this
    module does not require sentence_transformers / torch — only the
    code paths that actually train do.
    """
    import torch
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    history: list[float] = []

    class LossCapture(MultipleNegativesRankingLoss):
        def __init__(self, m: Any) -> None:
            super().__init__(model=m)
            self.history = history

        def forward(
            self,
            sentence_features: Any,
            labels: Any,
        ) -> "torch.Tensor":
            loss = super().forward(sentence_features, labels)
            try:
                self.history.append(float(loss.detach().cpu().item()))
            except Exception:  # pragma: no cover - defensive
                self.history.append(float("nan"))
            return loss

    return LossCapture(model)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(base_model: str) -> Any:
    """Resolve the ``--base-model`` flag to a SentenceTransformer instance."""
    from sentence_transformers import SentenceTransformer

    hf_id = BASE_MODELS.get(base_model, base_model)
    logger.info("Loading base model %s (-> %s)", base_model, hf_id)
    return SentenceTransformer(hf_id)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    model: Any,
    examples: list[Any],
    output_dir: Path,
    batch_size: int = 32,
    epochs: int = 1,
    max_steps: int | None = None,
) -> list[float]:
    """Run the contrastive fine-tune loop and return the per-step loss."""
    from torch.utils.data import DataLoader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss = make_loss_capture(model)
    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)

    fit_kwargs: dict[str, Any] = {
        "train_objectives": [(dataloader, loss)],
        "epochs": epochs,
        "output_path": str(output_dir),
        "show_progress_bar": False,
    }
    if max_steps is not None:
        fit_kwargs["steps_per_epoch"] = max_steps

    logger.info(
        "Training: examples=%d batch_size=%d epochs=%d max_steps=%s output=%s",
        len(examples),
        batch_size,
        epochs,
        max_steps,
        output_dir,
    )
    model.fit(**fit_kwargs)
    return list(loss.history)


def run_smoke_training(
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_pairs: int = 32,
    max_steps: int | None = None,
    batch_size: int = 4,
    epochs: int = 5,
    output_dir: Path | str | None = None,
    seed: int = 13,
) -> tuple[float, float]:
    """Smoke entry point used by tests.

    Seeds Python's ``random`` and ``torch`` so the loss curve is
    deterministic across runs (per-step MNRL loss on tiny synthetic
    batches is otherwise too noisy to make a useful regression signal).

    Returns ``(avg_loss_first_epoch, avg_loss_last_epoch)`` so the test
    can assert end-of-training loss is below start-of-training loss
    using stable averages, not noisy per-step values.
    """
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if output_dir is None:
        import tempfile

        output_dir = Path(tempfile.mkdtemp(prefix="body_contrastive_smoke_"))
    else:
        output_dir = Path(output_dir)

    model = SentenceTransformer(model_id)
    examples = make_synthetic_pairs(n_pairs, seed=seed)
    history = train(
        model,
        examples,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        max_steps=max_steps,
    )
    if not history:
        raise RuntimeError("Training produced no loss history; nothing to assert on")

    # Compare first-epoch average to last-epoch average — averages
    # smooth out per-batch stochasticity, and an epoch is the natural
    # unit since DataLoader shuffles between epochs.
    epoch_size = max(1, len(history) // max(1, epochs))
    avg_first = sum(history[:epoch_size]) / epoch_size
    avg_last = sum(history[-epoch_size:]) / epoch_size
    return avg_first, avg_last


# ---------------------------------------------------------------------------
# Run-log writer
# ---------------------------------------------------------------------------


def write_run_log(
    output_dir: Path,
    args: argparse.Namespace,
    plan: list[CohortAllocation],
    loss_history: list[float],
) -> Path:
    """Persist a small JSON run.log next to the model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.json"
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cohort_plan": [asdict(a) for a in plan],
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "loss_steps": len(loss_history),
    }
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a sentence encoder on (abstract, body-paragraph) "
            "positive pairs (S1 of prd_full_text_applications_v2.md). "
            "The full 100K pilot is GPU-window deferred — see "
            "docs/runbooks/train_body_abstract_contrastive.md."
        )
    )
    parser.add_argument(
        "--base-model",
        default="indus",
        choices=sorted(BASE_MODELS.keys()),
        help="Base sentence encoder to fine-tune (default: indus).",
    )
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=100_000,
        help="Number of (abstract, body-paragraph) training pairs (default: 100000).",
    )
    parser.add_argument(
        "--cohort-strategy",
        default="stratified-by-arxiv-class",
        choices=list(COHORT_STRATEGIES),
        help="How to draw the cohort (default: stratified-by-arxiv-class).",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the fine-tuned model + run.json "
            "(default: models/body_contrastive_<ts>)."
        ),
    )
    parser.add_argument(
        "--coverage-bias-json",
        type=Path,
        default=DEFAULT_COVERAGE_BIAS_JSON,
        help="Path to M1 coverage-bias JSON (default: results/full_text_coverage_bias.json).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on training steps per epoch (default: full epoch).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--synthetic-pairs",
        type=int,
        default=0,
        help="If >0, train on N synthetic pairs instead of pulling bodies from DB (smoke).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for cohort sampling and synthetic pair construction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cohort plan and exit without loading the model or training.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_model_dir or (
        REPO_ROOT
        / "models"
        / f"body_contrastive_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    coverage: dict[str, Any] | None = None
    if args.cohort_strategy == "stratified-by-arxiv-class":
        coverage = load_coverage_bias(args.coverage_bias_json)

    plan = build_cohort_plan(
        coverage=coverage,
        cohort_size=args.cohort_size,
        strategy=args.cohort_strategy,
        seed=args.seed,
    )
    logger.info("Cohort plan: %d strata, total=%d", len(plan), sum(a.n for a in plan))
    for alloc in plan:
        logger.info("  %-20s n=%d weight=%.4f", alloc.label, alloc.n, alloc.weight)

    if args.dry_run:
        logger.info("--dry-run set; skipping model load and training.")
        write_run_log(output_dir, args, plan, loss_history=[])
        return 0

    if args.synthetic_pairs > 0:
        logger.info(
            "Training on %d synthetic pairs (smoke path; --base-model=%s).",
            args.synthetic_pairs,
            args.base_model,
        )
        model = load_model(args.base_model)
        examples = make_synthetic_pairs(args.synthetic_pairs, seed=args.seed)
    else:
        # Real DB-backed pair construction is documented but not executed
        # in this PR — see runbook §4. Fail loudly so a future run that
        # forgets --synthetic-pairs cannot silently no-op.
        raise NotImplementedError(
            "Real-data pair construction is GPU-window deferred. "
            "Pass --synthetic-pairs N for the smoke loop, or implement the "
            "DB-backed pair builder per docs/runbooks/train_body_abstract_contrastive.md."
        )

    history = train(
        model,
        examples,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    log_path = write_run_log(output_dir, args, plan, history)
    logger.info(
        "Done. Steps=%d loss[0]=%.4f loss[-1]=%.4f run_log=%s",
        len(history),
        history[0] if history else float("nan"),
        history[-1] if history else float("nan"),
        log_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
