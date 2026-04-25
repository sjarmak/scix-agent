# Plan — negative-results-extractor (M3)

## Pattern catalog (with confidence tiers)

### High confidence (tier=3, label='high')

These phrases are unambiguous null-result statements in scientific prose.

- `no significant` (e.g. "no significant detection", "no significant excess")
- `null result(s)?`
- `failed to detect`
- `do(es)? not detect`
- `did not detect`
- `no evidence (for|of)`
- `non-detection`
- `not detected`
- `we exclude` (with sigma/limit context)
- `excluded? at \d+ ?(σ|sigma)`
- `rejected? at \d+ ?(σ|sigma)`
- `inconsistent with` (only when in conclusion-like sections)
- `retracted`
- `(was|has been) refuted`

### Medium confidence (tier=2, label='medium')

Hedged or contextual null statements; need section-of-interest guard.

- `cannot rule out`
- `consistent with no` (e.g. "consistent with no signal")
- `no statistically significant`
- `upper limit(s)? of`
- `lower limit(s)? of`
- `no correlation`
- `not statistically (significant|distinguishable)`
- `no clear (evidence|signal|trend)`
- `not consistent with`

### Low confidence (tier=1, label='low')

Ambiguous; only fire if section is in scope.

- `(if|whether) (real|present|correct)` (typical hedging in conclusions)
- `marginal(ly)? (significant|detected)?`
- `tentative(ly)? detected`

## Section guard

Only emit a span if its containing section name is in
`{'results', 'discussion', 'conclusions', 'summary'}`.
If the body has no recognizable headers (`'full'`/`'preamble'` only), allow
emission only for tier=3 patterns (high-confidence, section-agnostic).

## Data flow

```
body, sections (from parse_sections)
  -> for each (section_name, start, end, text):
       skip if section_name not in WINDOW
       for each pattern in catalog:
            for each match:
                 evidence = pad_to_250(body, match.start_in_body)
                 emit NegativeResultSpan(
                     section=section_name,
                     pattern_id=...,
                     confidence_tier_int=...,
                     confidence_tier_label=...,
                     match_text=match.group(0),
                     start_char=...,
                     end_char=...,
                     evidence_span=evidence,  # exactly 250 chars
                 )
  -> dedup overlapping matches (keep highest tier; tiebreak by earliest start)
```

`pad_to_250` rule: take `body[max(0, mid-125):mid+125]`, then if length < 250
pad on the side that has room until length == 250 (or end of text). When the
body itself is shorter than 250 chars, return the whole body padded with
spaces to 250 chars.

## NegativeResultSpan dataclass

```python
@dataclass(frozen=True)
class NegativeResultSpan:
    section: str
    pattern_id: str
    confidence_tier: int          # 1/2/3
    confidence_label: str         # 'low'/'medium'/'high'
    match_text: str
    start_char: int               # absolute offset into body
    end_char: int
    evidence_span: str            # exactly 250 chars
```

## Detector entry point

```python
def detect_negative_results(
    body: str,
    sections: list[tuple[str, int, int, str]] | None = None,
) -> list[NegativeResultSpan]:
    ...
```

If `sections` is None, calls `parse_sections(body)` itself.

## DB writer

```python
SOURCE = "neg_results_v1"
EXTRACTION_VERSION = "neg_results_v1"
EXTRACTION_TYPE = "negative_result"

def insert_extractions(
    conn: psycopg.Connection,
    bibcode: str,
    spans: list[NegativeResultSpan],
) -> int:
    payload = {
        "spans": [asdict(s) for s in spans],
        "n_spans": len(spans),
        "tier_counts": {"high": ..., "medium": ..., "low": ...},
    }
    max_tier = max((s.confidence_tier for s in spans), default=0) or None
    cur.execute(
        "INSERT INTO staging.extractions "
        "(bibcode, extraction_type, extraction_version, payload, source, confidence_tier) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (bibcode, extraction_type, extraction_version) "
        "DO UPDATE SET payload = EXCLUDED.payload, "
        "              source = EXCLUDED.source, "
        "              confidence_tier = EXCLUDED.confidence_tier",
        (bibcode, EXTRACTION_TYPE, EXTRACTION_VERSION, Jsonb(payload), SOURCE, max_tier),
    )
```

## Script: `scripts/run_negative_results.py`

argparse:
- `--max-papers INT`
- `--since-bibcode STR`
- `--dry-run`
- `--allow-prod`
- `--dsn STR`
- `--batch-size INT` (default 500)
- `-v/--verbose`

Guard:
```python
if is_production_dsn(dsn) and not args.allow_prod:
    logger.error("Refusing to run against production DSN %s — pass --allow-prod",
                 redact_dsn(dsn))
    return 2
```

Iteration: read `(bibcode, body)` pairs from `papers` where `body IS NOT NULL`,
ordered by bibcode, after the watermark. Detect, collect, batch-insert.

## Fixture: `tests/fixtures/negative_results_gold_100.jsonl`

100 lines. Distribution:
- 40 true (positives): hand-crafted hedging/null phrases drawn from common
  astrophysics language. Mix of high/medium/low tier examples.
- 60 false (negatives): plausible astronomy prose that contains words like
  "significant", "evidence", "detect", "limit", "ruled out" used in POSITIVE
  contexts (so the detector cannot cheat with single keyword matching).

Each line: `{"text": str, "label": true|false, "tier": "high"/"medium"/"low"|null, "note": str}`.

## Tests: `tests/test_negative_results.py`

1. `test_pattern_catalog_high_examples` — unit tests on raw text matches.
2. `test_section_guard_blocks_intro_match` — high-tier match in 'introduction'
   should NOT be emitted.
3. `test_evidence_span_is_exactly_250_chars` — invariant on output.
4. `test_negative_result_dataclass_is_frozen`.
5. `test_full_text_no_section_headers_only_high_tier_emitted`.
6. `test_db_insert_uses_staging_extractions_with_correct_columns` — mock
   `psycopg.Connection.cursor().execute()` and assert SQL + params.
7. `test_precision_recall_on_gold_fixture` — load 100-line jsonl, run
   detect on each `text` (treated as a single "results" section), compute
   precision/recall, assert >= 0.70 / >= 0.60.

## Phase ordering

1. Write detector module (`src/scix/negative_results.py`).
2. Write fixture (100 hand-curated lines).
3. Write tests; iterate detector pattern catalog until P>=0.70 R>=0.60.
4. Write CLI script.
5. Commit.

## Out of scope

- No new migration (schema already accepts the type).
- No promotion-script changes (mig 015 doesn't carry source/tier; that's a
  known issue tracked elsewhere).
- No MCP `entity` tool wiring (separate work unit).
