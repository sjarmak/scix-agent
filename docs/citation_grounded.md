# Citation-Grounded Gate

Mechanical post-draft check that every assertion in a SciX Deep Search
persona's draft answer traces to a tool-result quote. Implements PRD
**MH-6** with amendment **A6** (softened UX).

- Module: `src/scix/citation_grounded.py`
- Threshold tuner: `src/scix/citation_grounded_tune.py`
- Tests: `tests/test_citation_grounded.py`,
  `tests/test_citation_grounded_adversarial.py`

## Public API

### `grounded_check(draft, tool_results, threshold=0.82) -> GroundingReport`

Runs the gate once. Splits `draft` into sentence-like segments, filters
to assertions, and scores each against the quotes extracted from
`tool_results`. Returns:

```python
@dataclass(frozen=True)
class GroundingReport:
    assertions: tuple[str, ...]      # sentences flagged as claims
    unmatched: tuple[str, ...]       # subset whose score < threshold
    grounded: bool                   # True iff unmatched is empty
    threshold_used: float
```

### `revise_with_gate(draft, tool_results, dispatcher, max_revisions=2, rigor_mode=False, threshold=0.82) -> RevisedDraft`

Bounded revision loop. Calls `dispatcher.revise(draft, unmatched)` up to
`max_revisions` times. After the budget is spent, residual unmatched
sentences are auto-stripped per amendment A6.

```python
@dataclass(frozen=True)
class RevisedDraft:
    answer: str                      # final text (with replacements)
    stripped: tuple[str, ...]        # sentences removed
    revision_count: int              # 0..max_revisions
    grounded: bool
    threshold_used: float
```

`RevisionDispatcher` is a single-method protocol:

```python
class RevisionDispatcher(Protocol):
    def revise(self, draft: str, unmatched: list[str]) -> str: ...
```

Production wraps a Claude Code OAuth subagent; tests inject a fake.

### `shadow_tune(gold_paraphrases, target_fp_rate=0.15) -> float`

Offline threshold tuner (called once after a 48-hour shadow window).
Returns the largest threshold whose false-positive rate on legitimate
paraphrases is at or below `target_fp_rate`.

## Assertion parsing (heuristic)

A sentence is treated as an assertion when it ends in `.`, `!`, or `?`
**and** contains at least one of:

- A content verb (`find`, `show`, `demonstrate`, `conclude`, `report`,
  `observe`, `measure`, `detect`, `derive`, `predict`, `confirm`,
  `refute`, `suggest`, `indicate`, `present`, `propose`, `claim`,
  `argue`, `establish`, `identify`, `infer`, `support`, `reveal`, plus
  inflections);
- A digit (numeric content);
- A mid-sentence uppercase token (named-entity hint);
- A parenthetical citation (`(Author 2011)`, `(2011)`, or `[1]`).

Sentences without claim content (questions, transitions, headers) are
not subject to the gate. The parser is intentionally permissive: a
mis-flagged sentence costs at most one revision turn from the bounded
budget.

## Substring short-circuit

If an assertion is a substring of any tool quote (or vice versa, after
normalization to lowercase + collapsed whitespace), the score
short-circuits to 0.95 — above the default 0.82 threshold — without
touching the embedder. Exact-quote citations are therefore free of
GPU round-trips.

## Embedding path

When the substring path doesn't fire, the assertion and all tool quotes
are embedded with **INDUS** (`nasa-impact/nasa-smd-ibm-st-v2`, 768d, mean
pooling) via `scix.embed`. The score is the maximum cosine similarity
across quotes.

The embedder is injectable through `set_embedder(fn)` for unit testing.
The provided count-vector test embedder is deterministic and avoids the
INDUS model load.

## A6 UX modes

| Mode               | Residual replacement                | Trailer block                                |
| ------------------ | ----------------------------------- | -------------------------------------------- |
| Default            | Superscript marker (`¹`, `²`, …)    | `Footnotes` block listing stripped sentences |
| `--rigor` / `True` | Literal `[ungrounded claim removed]`| (none — annotation is inline)                |

Per PRD: default mode reads "as a colleague" (synthesized confident
output with footnoted gate annotations); rigor mode reveals the full
gate trace.

## Grounding != Truth

The gate is **mechanical, not epistemic**. Per PRD MH-6 and Tension 2:

> The output never claims "evidence verified" or "facts checked"; it
> claims "citations grounded in retrieved sources" — a mechanical not
> epistemic property.

What `citation_grounded` *does*:

- Verifies that every claim-bearing sentence in the draft has a
  cosine-similar (or substring-matching) quote in the tool-result
  history.
- Flags or strips sentences that don't.

What `citation_grounded` *does not*:

- Verify that the matched quote is true.
- Verify that the cited paper made the claim correctly.
- Verify that the corpus contains all relevant counter-evidence.
- Detect citation laundering (the claim and its source agree
  syntactically but the source is itself unsupported by primary
  literature).
- Compensate for retraction blindness, methodological supersession, or
  paradigm shifts (those are MH-3 / amendment A3 concerns —
  `correction_events JSONB`).

A draft can be `grounded=True` while being *wrong* in any of these ways.
The retraction-overlay (MH-3), correction-events ingestion (A3), and
red-team eval set (SH-3) are the layers that address truth concerns.
This gate addresses only the much narrower question "did the persona
make up sentences that aren't in the retrieval results?"

A green `citation_grounded: true` badge is therefore a **lower bound on
honesty about retrieval**, not a truth claim. UI surfaces consuming this
flag must not present it as "verified."

## Threshold tuning workflow

1. **Build a 50-pair gold set** of `(legitimate_paraphrase,
   source_quote)` pairs drawn from MH-1 hand-validation. Both sides
   should be ones you would accept as a match.
2. **Run shadow mode** for ~48 hours: log gate decisions but do not
   block emission. Keep the default threshold at 0.82 during this
   window.
3. **Call `shadow_tune(gold_pairs, target_fp_rate=0.15)`** to pick the
   enforcement threshold. The function returns the largest threshold
   whose false-positive rate on legitimate paraphrases stays at or
   below 15%.
4. **Lock the returned threshold** into the production
   `grounded_check` call site and enable enforcement.

## Limitations

- The sentence parser is a regex heuristic. It mis-handles abbreviations
  (Mr., e.g.), numerals embedded in citations (Smith 2011), and very
  long compound sentences. The bounded-revision budget caps the cost.
- The embedder defaults to INDUS — chosen for scientific-text alignment.
  Quotes from non-science contexts (UI strings, error messages) score
  poorly. Tool-result extraction prefers `quote`, `text`, `snippet`,
  `context_snippet`, `body`, `content`, `result`, `answer`, `abstract`,
  and `title` keys; nested `results: [...]` arrays are flattened.
- The 20-case adversarial fixture uses a count-vector test embedder, not
  INDUS. The fixture asserts gate **logic** (substring path, threshold
  enforcement, max-revision bounding); INDUS-vs-paraphrase calibration
  is the responsibility of `shadow_tune` against a real gold set.
