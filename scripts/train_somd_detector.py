#!/usr/bin/env python3
"""Fine-tune SciBERT on the SOMD-2024 corpus as a software-mention detector (bead dbl.22).

Why this exists
---------------
The cheap Lafia + GLiNER type-confirmation pass (bead dbl.20) plateaus at ~79%
precision on real arXiv: GLiNER legitimately type-confirms generic software-typed
noun phrases the cue mis-anchors ("Adam solver", "IBM framework", "VR Toolkit"),
so the confirmation stage structurally cannot reject them (see
``results/dbl21_lafia_span_fix_eval.md``). dbl.22 is the fallback the dbl.18
references name: a *heavier* ML detector trained specifically on software-mention
recognition. This trains the local, open-weight realisation — a SciBERT token
classifier on the SOMD-2024 subtask-1 data (the SoMeSci corpus, repackaged for
the NSLP-2024 shared task) — which is the modern superset of the Bi-LSTM-CRF
reference (arXiv:2405.13135) and the winning SOMD-2024 architecture family
(BERTology three-stage framework, arXiv:2405.01575).

Local open-weight only: the SciBERT backbone (``allenai/scibert_scivocab_uncased``)
is already in the HF cache; no paid API, no Falcon-7b download. EVAL ONLY — the
detector feeds ``scix.extract.somd_detect`` for the dbl.22 precision eval; no DB
writes and no production pass here.

Data layout (downloaded into ``data/somd2024/``, gitignored)
------------------------------------------------------------
``subtask1_train.data.txt``   one whitespace-tokenised sentence per line
``subtask1_train.labels.txt`` parallel BIO labels (27-tag SOMD scheme), aligned
                              token-for-token with the data file.

Run under scix-batch (GPU + memory isolation; see CLAUDE.md):

    scix-batch .venv/bin/python scripts/train_somd_detector.py \
        --out models/somd_scibert --epochs 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger("train_somd_detector")

DEFAULT_BACKBONE = "allenai/scibert_scivocab_uncased"
DEFAULT_DATA_DIR = Path("data/somd2024")
DEFAULT_OUT_DIR = Path("models/somd_scibert")
# Label assigned to subword continuations and special tokens so the loss ignores
# them (standard HF token-classification alignment).
IGNORE_INDEX = -100


def read_conll(data_path: Path, labels_path: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Read parallel token / BIO-label files into per-sentence lists.

    Each line is one sentence; tokens and labels are whitespace-separated and
    must align one-for-one. A mismatch is a corpus error, not something to paper
    over — raise rather than silently truncate.
    """
    tokens: list[list[str]] = []
    labels: list[list[str]] = []
    with data_path.open() as df, labels_path.open() as lf:
        for lineno, (dline, lline) in enumerate(zip(df, lf, strict=True), start=1):
            toks = dline.split()
            labs = lline.split()
            if len(toks) != len(labs):
                raise ValueError(f"{data_path}:{lineno}: {len(toks)} tokens vs {len(labs)} labels")
            if not toks:
                continue
            tokens.append(toks)
            labels.append(labs)
    return tokens, labels


def build_label_maps(labels: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    """Sorted, deterministic label<->id maps with O pinned to id 0."""
    vocab = sorted({lab for seq in labels for lab in seq})
    vocab.remove("O")
    ordered = ["O", *vocab]
    label2id = {lab: i for i, lab in enumerate(ordered)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def encode(tokens, labels, tokenizer, label2id, max_length):
    """Tokenise into subwords and project word labels onto the first subword."""
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    aligned: list[list[int]] = []
    for i, word_labels in enumerate(labels):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        row: list[int] = []
        for wid in word_ids:
            if wid is None:
                row.append(IGNORE_INDEX)
            elif wid != prev:
                row.append(label2id[word_labels[wid]])
            else:
                row.append(IGNORE_INDEX)
            prev = wid
        aligned.append(row)
    enc["labels"] = aligned
    return enc


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


def entity_f1(predictions, label_ids, id2label) -> dict[str, float]:
    """Token-level software-presence P/R/F1 (any non-O tag = software token).

    Binary collapse, because the downstream confirmation use only asks "is this
    span a software mention" — the fine-grained 27-tag accuracy is not the gate.
    """
    preds = np.argmax(predictions, axis=-1)
    tp = fp = fn = 0
    for p_row, l_row in zip(preds, label_ids):
        for p, lbl in zip(p_row, l_row):
            if lbl == IGNORE_INDEX:
                continue
            gold_sw = id2label[lbl] != "O"
            pred_sw = id2label[p] != "O"
            tp += gold_sw and pred_sw
            fp += pred_sw and not gold_sw
            fn += gold_sw and not pred_sw
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"sw_precision": prec, "sw_recall": rec, "sw_f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokens, labels = read_conll(
        args.data_dir / "subtask1_train.data.txt",
        args.data_dir / "subtask1_train.labels.txt",
    )
    label2id, id2label = build_label_maps(labels)
    logger.info("loaded %d sentences, %d labels", len(tokens), len(label2id))

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(tokens))
    n_dev = int(len(tokens) * args.dev_fraction)
    dev_idx, train_idx = set(idx[:n_dev]), idx[n_dev:]
    tr_tok = [tokens[i] for i in train_idx]
    tr_lab = [labels[i] for i in train_idx]
    dv_tok = [tokens[i] for i in sorted(dev_idx)]
    dv_lab = [labels[i] for i in sorted(dev_idx)]

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForTokenClassification.from_pretrained(
        args.backbone,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = _Dataset(encode(tr_tok, tr_lab, tokenizer, label2id, args.max_length))
    dev_ds = _Dataset(encode(dv_tok, dv_lab, tokenizer, label2id, args.max_length))

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=str(args.out / "_trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        bf16=bf16,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda ep: entity_f1(ep.predictions, ep.label_ids, id2label),
    )
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("final dev metrics: %s", metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    (args.out / "dev_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("saved detector to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
