# dbl.22 — heavier ML software-mention detector: build + negative result

Bead dbl.22 is the fallback from dbl.20: *if* the cheap Lafia + GLiNER
type-confirmation pass plus the span-boundary fixes (dbl.21/23/23w) still miss the
≥85% precision gate on real arXiv, escalate to the heavier ML detector the dbl.18
references name (Falcon-7b SOMD, arXiv:2405.08514 / Bi-LSTM-CRF, arXiv:2405.13135).

## Precondition (confirmed met)

`results/dbl21_lafia_span_fix_eval.md`: after all span fixes the GLiNER-confirm
default tops out at **78.9% precision / 90.9% recall** on the 85-mention dbl.21
labelled sample. The 8 surviving false positives are not span-boundary errors —
they are generic software-typed noun phrases GLiNER legitimately type-confirms
("Adam solver", "IBM framework", "VR Toolkit") plus dataset-typed cue mis-anchors
("Dogs" → "Kaggle Cats vs. Dogs Dataset"). A type-confirmation stage structurally
cannot reject the software ones, so the escalation is warranted.

## What was built (local open-weight, no paid API, no Falcon download)

Falcon-7b SOMD has no released checkpoint — it would require a 14 GB download plus
full fine-tuning. The lighter, fully-local realisation is the modern superset of
the Bi-LSTM-CRF reference and the winning SOMD-2024 architecture family
(BERTology three-stage framework, arXiv:2405.01575): a **SciBERT token classifier
fine-tuned on the SOMD-2024 subtask-1 corpus** (the SoMeSci corpus, repackaged for
NSLP-2024; obtained from the three-stage-framework repo, gitignored under
`data/somd2024/`). The SciBERT backbone was already in the HF cache.

- `scripts/train_somd_detector.py` — fine-tune (4 epochs, dev software-presence
  P/R/F1 = **90.1 / 88.7 / 89.4**). Checkpoint → `models/somd_scibert/` (gitignored).
- `src/scix/extract/somd_detect.py` — `SomdDetector.detect` tags software spans;
  `confirm_software_mentions` is a second confirmation stage that drops
  *software*-typed Lafia candidates no SOMD span overlaps. Dataset candidates are
  out of SoMeSci's domain and pass through on GLiNER's verdict (documented limit).
- `scripts/eval_somd_confirm.py` — replays the detector over the **same** dbl.21
  labelled sample (apples-to-apples, no re-labelling).

## Result — does NOT clear the gate, at a recall cost

| pipeline | precision | recall retained |
|---|---|---|
| GLiNER confirm (dbl.20/21) | 78.9% | 90.9% |
| + SOMD detector (dbl.22) | **83.3%** | **75.8%** |

SOMD rejected 3 of the 5 software FPs (IBM, LDA→Gensim, LWp) but **also dropped 5
genuine software references** (YCSB, PatchMatch, HuggingFace, Unity time.time, LSL
Markers) and **kept 2 FPs** (Adam, VR). Net: +4.4 pts precision for −15.1 pts
recall, still short of 85%.

## Why a heavier model does not fix this (the actionable finding)

Max software-class probability the detector assigns each GLiNER-confirmed software
candidate, sorted desc (FP = gold-false):

```
TP LSL 1.00   TP HistCite 1.00   FP VR 1.00   TP MATLAB 0.99   TP PythonImaging 0.99
TP OpenStack 0.93   FP Adam 0.90   TP Unity-LSL 0.90   FP LWp 0.84
TP LSL-Markers 0.44   TP YCSB 0.22   TP HuggingFace 0.06   FP IBM 0.03
TP PatchMatch 0.02   FP LDA 0.00   TP Unity-time.time 0.00
```

(Probability is the max softmax weight for any software-class label at the
candidate's tokens; the detector emits a span only when the argmax label is
non-`O`, so a candidate at p≈0.84 can still produce no overlapping span if `O`
wins the argmax at its exact offsets — which is why the kept/dropped column does
not track this column monotonically.)

The FPs are **interleaved with the TPs across the whole range** — `VR` (0.998) and
`Adam` (0.904) outscore genuine references `YCSB` (0.215), `HuggingFace` (0.061),
`PatchMatch` (0.021). **No threshold separates them**, so a confidence sweep
cannot beat the argmax operating point. The detector genuinely considers "VR" and
"Adam" software (they *are* software-domain terms) and misses novel named tools the
SoMeSci corpus underrepresents (HuggingFace, YCSB, PatchMatch).

The residual error is therefore **not model capacity** — it is two things a heavier
software-mention model cannot resolve:

1. **Labeling-boundary ambiguity.** The disagreement on "VR Toolkit" / "Adam
   solver" is whether a generic `<term> <generic-head>` phrase counts as a *named
   software reference*. That is a gold-label/task-definition call, not a detection
   error. Falcon-7b would face the same boundary.
2. **Dataset coverage.** 3 of the 8 surviving FPs are dataset-typed ("Dogs",
   "Gaussians", "PN"). SoMeSci annotates software only, so **no** SOMD-family model
   can touch them; the dataset precision ceiling needs a dataset-mention detector,
   not a software one.

## Recommendation

- **Do not adopt** the SOMD confirmation stage for the production informal-reference
  pass: it trades 15 pts of recall for 4.4 pts of precision and still misses the
  gate. The cheap GLiNER path (78.9% / 90.9%) remains the better operating point.
- Reaching ≥85% on this sample is **not** a model-swap problem. The two real levers
  are (a) a precise labeling-policy definition for generic `<term> <head>` phrases,
  and (b) a dataset-mention detector for the dataset-typed residual — both new beads.
- The detector, training, and eval are retained as reproducible artifacts so the
  conclusion can be re-checked if the labeling policy changes.
