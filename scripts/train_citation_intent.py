#!/usr/bin/env python3
"""Train a SciBERT-based citation intent classifier on the SciCite dataset.

Downloads the SciCite dataset from HuggingFace, maps labels to our schema
(background, method, result_comparison), fine-tunes SciBERT, and saves the
model to disk.

Usage:
    python scripts/train_citation_intent.py --output-dir models/citation_intent
    python scripts/train_citation_intent.py --epochs 5 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SciBERT on SciCite for citation intent classification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/citation_intent",
        help="Directory to save the fine-tuned model (default: models/citation_intent)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Base model to fine-tune (default: allenai/scibert_scivocab_uncased)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training and eval batch size (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length (default: 512)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import evaluate
        import numpy as np
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        logger.error(
            "Required packages not installed. Install with:\n"
            "  pip install transformers datasets evaluate torch scikit-learn numpy"
        )
        raise SystemExit(1) from exc

    # Label mapping: SciCite uses 0=background, 1=method, 2=result
    # We keep the same integer IDs but rename "result" -> "result_comparison"
    id2label = {0: "background", 1: "method", 2: "result_comparison"}
    label2id = {v: k for k, v in id2label.items()}

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading SciCite dataset from HuggingFace...")
    dataset = load_dataset("allenai/scicite")

    # SciCite has train/validation/test splits with columns:
    #   string, sectionName, label, citingPaperId, citedPaperId, ...
    # The "string" column is the citation context text.
    # The "label" column is 0/1/2.

    logger.info(
        "Dataset loaded: train=%d, validation=%d, test=%d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples["string"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True)

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    logger.info("Loading base model: %s", args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred: tuple) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {**acc, **f1}

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training for %d epochs...", args.epochs)
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    logger.info("Test results: %s", test_results)

    # ------------------------------------------------------------------
    # 7. Save model and tokenizer
    # ------------------------------------------------------------------
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model to %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    logger.info("Training complete. Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
