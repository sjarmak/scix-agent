# Lafia + GLiNER type-confirmation precision eval (bead dbl.20)

Source sample: `tests/fixtures/lafia_confirm_validation.jsonl` (92 labelled candidate mentions, 34 true references).

| pipeline | candidates kept | true | false | precision | recall retained |
|---|---|---|---|---|---|
| Lafia heuristic (baseline) | 92 | 34 | 58 | **37.0%** | 100.0% |
| + GLiNER confirm — default (subset overlap, conf>=0.7) | 42 | 30 | 12 | **71.4%** | 88.2% |
| + GLiNER confirm — conf>=0.95 | 29 | 24 | 5 | **82.8%** | 70.6% |
| + GLiNER confirm — exact surface match | 6 | 6 | 0 | **100.0%** | 17.6% |

- **Baseline precision**: 37.0% on this sample. dbl.18 measured 60-67% on its two samples; absolute precision is sample-dependent (this slice is modern, cs-heavy arXiv with figure/table OCR noise), so the transferable signal is the *relative* lift, not the absolute baseline.
- **Default-confirmed precision**: 71.4% — acceptance gate (>=85%): **FAIL**.
- **Recall retained**: 88.2% of true references survive confirmation (30/34).
- **False positives removed**: 46 of 58 (79.3%).
- **Knob sweep**: raising the GLiNER confidence floor or demanding an exact surface match trades recall for precision but does not clear 85% at usable recall — exact match reaches 100% precision only by discarding most true references (GLiNER's span includes the head noun, so it rarely equals the cue-extracted surface).

## Recommendation

GLiNER confirmation lifts precision from 37.0% to 71.4%, removes 79% of false positives, and retains 88% recall, but does NOT reach the >=85% gate on this sample, so the production pass stays blocked. The residual false positives are dominated by Lafia *span-boundary* errors (the cue captures an adjective adjacent to a head noun, e.g. "high-quality dataset"); GLiNER legitimately confirms the full noun phrase, so the confirmation pass structurally cannot reject them. Two non-exclusive next steps, both separate beads: (a) fix Lafia name-span extraction so the candidate excludes leading generic adjectives, then re-measure; (b) escalate to the heavier ML detector the dbl.18 refs use (Falcon-7b SOMD / Bi-LSTM-CRF), which was out of scope here.
