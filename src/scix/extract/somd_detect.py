"""SOMD software-mention detector — the heavier ML fallback for bead dbl.22.

The Lafia + GLiNER type-confirmation pass (bead dbl.20) lifts informal-reference
precision from ~38% to ~79% but cannot clear the >=85% gate: GLiNER legitimately
type-confirms generic software-typed noun phrases the cue mis-anchors ("Adam
solver", "IBM framework", "VR Toolkit"), so a type-confirmation stage structurally
cannot reject them (``results/dbl21_lafia_span_fix_eval.md``). dbl.22 is the
fallback the dbl.18 references name: a detector trained specifically on
*software-mention recognition*, which knows "Adam solver" is not a named software
artefact even though it is software-typed.

This module loads the local SciBERT token classifier fine-tuned on the SOMD-2024
subtask-1 corpus (``scripts/train_somd_detector.py``) and exposes it two ways:

* :meth:`SomdDetector.detect` — tag software-mention spans in arbitrary text.
* :func:`confirm_software_mentions` — a second confirmation stage that drops
  *software*-typed Lafia candidates no SOMD software span overlaps. Dataset-typed
  candidates are out of SOMD's domain (the SoMeSci corpus annotates software, not
  datasets) and pass through untouched — the caller keeps GLiNER's verdict for
  those. This is a deliberate, documented scope limit, not a silent fallback.

Local open-weight only (SciBERT backbone, no paid API, no Falcon-7b download).
EVAL ONLY: no DB writes, no provenance changes. When a production pass lands it
reuses :func:`confirm_software_mentions` after ``lafia_confirm.confirm_mentions``.

ZFC note: the software/dataset routing and the span-overlap test are mechanical;
the semantic judgement ("is the span at this position a real software mention?")
is delegated to the trained model, not hand-coded.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from scix.extract.lafia import InformalMention

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models/somd_scibert")
# Chars of context on each side of a candidate span handed to the detector. Matches
# lafia_confirm.DEFAULT_CONTEXT_CHARS: the cue already localised the mention, so the
# model only re-scores a tight window, not the whole body.
DEFAULT_CONTEXT_CHARS: int = 160
DEFAULT_MAX_LENGTH: int = 256
# Lafia entity_type values the SOMD detector is competent to adjudicate. SOMD/SoMeSci
# annotates software only, so dataset candidates are out of domain.
SOFTWARE_TYPE: str = "software"


@dataclass(frozen=True)
class SoftwareSpan:
    """A software mention the SOMD detector tagged in a text.

    Offsets are character positions into the text passed to :meth:`detect`.
    ``label`` is the dominant SOMD tag of the span (e.g. ``Application_Usage``);
    ``score`` is the mean softmax probability of its entity tokens.
    """

    surface: str
    start_char: int
    end_char: int
    label: str
    score: float


def _spans_from_tokens(
    text: str,
    offsets: Sequence[tuple[int, int]],
    special: Sequence[int],
    pred_labels: Sequence[str],
    probs: Sequence[float],
) -> list[SoftwareSpan]:
    """Merge runs of consecutive non-``O`` subword tokens into software spans.

    A ``B-`` tag opens a new span; an ``I-``/continuation extends the open one; an
    ``O`` (or a special token) closes it. Type collapse is intentional — the
    downstream question is "is there software here", not which fine SOMD subtype.
    """
    spans: list[SoftwareSpan] = []
    cur_lo: int | None = None
    cur_hi = 0
    cur_label = ""
    cur_probs: list[float] = []

    def _flush() -> None:
        nonlocal cur_lo, cur_hi, cur_label, cur_probs
        if cur_lo is not None:
            spans.append(
                SoftwareSpan(
                    surface=text[cur_lo:cur_hi],
                    start_char=cur_lo,
                    end_char=cur_hi,
                    label=cur_label,
                    score=sum(cur_probs) / len(cur_probs),
                )
            )
        cur_lo, cur_hi, cur_label, cur_probs = None, 0, "", []

    for (lo, hi), is_special, lab, prob in zip(offsets, special, pred_labels, probs, strict=True):
        if is_special or lo == hi:
            continue
        if lab == "O":
            _flush()
            continue
        is_begin = lab.startswith("B-")
        if cur_lo is None or is_begin:
            _flush()
            cur_lo, cur_hi, cur_label, cur_probs = lo, hi, lab[2:] if "-" in lab else lab, [prob]
        else:
            cur_hi = hi
            cur_probs.append(prob)
    _flush()
    return spans


class SomdDetector:
    """SciBERT token classifier fine-tuned on SOMD-2024 software mentions."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        *,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 16,
    ) -> None:
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"SOMD detector not found at {model_dir}. Train it first: "
                "scix-batch .venv/bin/python scripts/train_somd_detector.py"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def detect_batch(self, texts: Sequence[str]) -> list[list[SoftwareSpan]]:
        """Tag software spans in each text. One forward pass per ``batch_size`` block."""
        results: list[list[SoftwareSpan]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            enc = self.tokenizer(
                chunk,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            special = enc.pop("special_tokens_mask")
            model_inputs = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**model_inputs).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            for i, text in enumerate(chunk):
                labels = [self.id2label[int(p)] for p in pred[i]]
                results.append(
                    _spans_from_tokens(
                        text,
                        offsets[i].tolist(),
                        special[i].tolist(),
                        labels,
                        conf[i].tolist(),
                    )
                )
        return results

    def detect(self, text: str) -> list[SoftwareSpan]:
        """Tag software spans in a single text."""
        return self.detect_batch([text])[0]


def _overlaps(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> bool:
    """True if the two half-open char intervals share at least one position."""
    return a_lo < b_hi and b_lo < a_hi


def confirm_software_mentions(
    detector: SomdDetector,
    body: str,
    mentions: Iterable[InformalMention],
    *,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[InformalMention]:
    """Drop software-typed Lafia candidates no SOMD software span overlaps.

    Mirrors :func:`scix.extract.lafia_confirm.confirm_mentions`: each candidate is
    re-scored inside a tight context window (the cue already localised it). A
    *software*-typed candidate survives iff the detector emits a software span
    overlapping the candidate's offset inside that window. Dataset-typed
    candidates are out of SOMD's domain and pass through unchanged.
    """
    mentions = list(mentions)
    if not mentions:
        return []

    windows: list[str] = []
    win_offsets: list[tuple[int, int]] = []  # candidate offset relative to its window
    for m in mentions:
        lo = max(0, m.start_char - context_chars)
        hi = min(len(body), m.end_char + context_chars)
        windows.append(body[lo:hi])
        win_offsets.append((m.start_char - lo, m.end_char - lo))

    per_window = detector.detect_batch(windows)

    kept: list[InformalMention] = []
    for m, spans, (clo, chi) in zip(mentions, per_window, win_offsets, strict=True):
        if m.entity_type != SOFTWARE_TYPE:
            kept.append(m)  # out of SOMD domain — caller keeps its prior verdict
            continue
        if any(_overlaps(clo, chi, s.start_char, s.end_char) for s in spans):
            kept.append(m)
    return kept
