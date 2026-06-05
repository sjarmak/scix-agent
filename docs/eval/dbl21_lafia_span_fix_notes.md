# Lafia span-boundary fix — eval result (bead dbl.21)

Follow-on to dbl.20. dbl.20 found the GLiNER type-confirmation pass lifts Lafia
informal-reference precision but leaves residual false positives dominated by
**span-boundary errors**: the cue captures a leading generic descriptive
adjective adjacent to a head noun (`high-quality dataset`, `open-source code`,
`high-resolution dataset`), and GLiNER legitimately confirms the full noun
phrase, so the confirmation pass structurally cannot reject it.

## The fix

`src/scix/extract/lafia.py` now trims a leading generic descriptive adjective
off the **span** of a cue-extracted candidate before it is emitted
(`_LEADING_ADJECTIVES` + `_is_leading_adjective`, applied in `_finalise_run`).
This is a span-boundary adjustment, **not** a name-token stopword:

- `high-resolution LAMOST survey` → candidate becomes `LAMOST` (start offset
  shifts to the real name; offsets still slice back to the surface).
- `high-quality dataset` → the head noun is the cue, so trimming the adjective
  leaves an empty run and **no candidate is emitted**.

The adjective set is restricted to generic quality / resolution / scale /
availability / processing descriptors that never lead a resource name. Domain
compounds that can be part of a real name (`high-mass`, `X-ray`) are excluded,
and real hyphenated names (`Pan-STARRS`) are untouched.

The name-span extraction was refactored to carry per-token offsets so the
trimmed span reports an accurate `start_char`/`end_char`. A side effect: the
existing author-citation guard (`data from Cohen et al.`) now sees an accurate
end offset and correctly drops one malformed candidate whose surface had
captured an author surname (`Dancing2Music Lee`); that candidate was never
GLiNER-confirmed, so it does not affect the gate.

## Re-measurement (eval-only, no DB writes)

Re-derived from the same labelled sample the dbl.20 eval was built on
(`tests/fixtures/lafia_confirm_validation.jsonl`) by re-running the **fixed**
detector over each row's evidence snippet — methodology validated by first
confirming the pre-fix detector reproduces all 92 stored surfaces exactly. The
surviving rows (`results/dbl21_lafia_confirm_postfix.jsonl`, 85 of 92) were
scored with the real harness:

```
python scripts/eval_lafia_gliner_confirm.py score \
    --labeled results/dbl21_lafia_confirm_postfix.jsonl \
    --report  results/dbl21_lafia_span_fix_eval.md
```

(The auto-generated `results/dbl21_lafia_span_fix_eval.md` carries the dbl.20
report template — its **table is authoritative**, its prose recommendation is
not, since this run *is* the span fix it proposes.)

### Before → after

| pipeline                         | precision (pre → post) | recall (pre → post) | gate |
|----------------------------------|------------------------|---------------------|------|
| Lafia heuristic (baseline)       | 37.0% → 38.8%          | 100% → 100%         | —    |
| + GLiNER confirm — default (≥0.7) | 71.4% → **78.9%**     | 88.2% → 90.9%       | FAIL |
| + GLiNER confirm — conf ≥ 0.95   | 82.8% → **88.9%**      | 70.6% → 72.7%       | PASS |
| + GLiNER confirm — exact surface | 100% → 100%            | 17.6% → 18.2%       | PASS |

The span fix removes exactly the 4 confirmed leading-adjective FPs
(`high-quality`, `high-resolution`, `open-source`, `High-Quality Standardized`)
plus 3 non-confirmed baseline FPs.

## Verdict: default-policy gate NOT met (78.9% < 85%)

The fix does what it was scoped to do — it eliminates the leading-adjective FP
class — but the **default** confirmation policy stays at 78.9%, below the 85%
gate, so the production pass remains blocked. The 8 residual confirmed FPs are
all **out of scope** for a span-boundary fix:

- partial-name capture: `Dogs` (← *Kaggle Cats vs. Dogs Dataset*), `Gaussians`
  (← *25 Gaussians dataset*)
- method / algorithm acronyms mistaken for software: `LDA` (← *Gensim's LDA
  library*), `Adam` (← *Adam solver*), `PN`
- company / generic acronyms: `IBM` (*IBM framework*), `VR` (*VR Toolkit*),
  `LWp`

These are exactly the cases the dbl.18 references target with a learned model,
i.e. **bead dbl.22** (Falcon-7b SOMD / Bi-LSTM-CRF, local open-weight only).

### Secondary finding

Span fix **+ GLiNER conf floor 0.95** clears the gate at 88.9% precision, but at
72.7% recall (24/33 true refs retained) — the span fix is what pushes this
variant over 85% (it was 82.8% before). Whether trading recall for a tighter
threshold is acceptable for production, versus escalating to dbl.22 for
default-policy precision at high recall, is an architecture call, not decided
here.
