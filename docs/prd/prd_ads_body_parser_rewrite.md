# PRD: ADS-Body Parser Rewrite — Inline Keyword-Anchor Section Extraction

## Status (2026-04-21)

Build PRD — ready to execute. Unblocks every downstream consumer of
`papers_fulltext.sections` (claim extraction, section embeddings
[`wqr.9`], section-level MCP tools). Without this fix, 14.64M of the
14.94M Tier-1 rows produced by the structural-fulltext run are
effectively empty (section array = `[]`).

## Goal

Replace `src/scix/sources/ads_body_parser.py` — a regex family bank
with `re.MULTILINE` `^heading$` anchors — with an inline-keyword-anchor
parser that finds section boundaries *within* flat single-line
PDF-extracted ADS body text. Then re-parse the 14,941,487 rows already
in `papers_fulltext` under the new `parser_version`, raising
multi-section yield from 0.016 % → ≥60 %.

## Context (what we learned)

### The data shape

All ADS body text is stored as flat single-line PDF-extracted text:

| bibstem | sampled | zero-newline | avg body (chars) |
| ------- | ------- | ------------ | ---------------- |
| arXiv   | 6,837   | 6,832 (99.9 %) | 55,718 |
| PhRvD   | 600     | 600 (100 %)   | 35,772 |
| ApJ     | 505     | 505 (100 %)   | 42,272 |
| MNRAS   | 481     | 481 (100 %)   | 52,161 |
| A&A     | 451     | 451 (100 %)   | 45,002 |

### Why the old parser failed

`re.MULTILINE` with `^heading$` requires the heading to sit on its own
line. With zero newlines, `^` only matches at position 0, so the
parser produces at most one section per paper:

| n_sections | rows | pct |
| ---------- | ---- | --- |
| 0 | 14,644,354 | 98.01 % |
| 1 |    294,742 |  1.97 % |
| ≥2 |     2,391 |  0.016 % |

### Why inline-keyword matching will work

Section keywords *do* exist inline. 200 random modern (post-2020)
MNRAS papers:

| Keyword                | Hit rate |
| ---------------------- | -------- |
| `INTRODUCTION`         | 99.5 %   |
| `DISCUSSION/CONCLUSIONS` | 95.0 % |
| `RESULTS`              | 55.5 %   |
| `METHODS/OBSERVATIONS` | 52.0 %   |
| `REFERENCES`           | 35.5 %   |
| `INTRODUCTION` + `REFERENCES` together | 35.5 % |

Scanning for a fixed vocabulary of canonical section markers recovers
meaningful structure on >95 % of modern papers without LLM calls.

## Non-Goals

- **LLM-based structural tagging** — too expensive at 15M scale;
  inline keyword matching is the cheap, deterministic fix. Revisit
  later for the residual 30 % that still produce < 2 sections.
- **Tier 2 (ar5iv) wiring** — separate PRD (to be drafted). Ar5iv
  gives higher-fidelity LaTeX-derived structure but requires
  backfilling `papers_external_ids` first; orthogonal to this fix.
- **Sub-section / heading-level detection** — emit `level=1` for all
  sections in v2; hierarchical section tree is a follow-up.
- **Paragraph-boundary recovery** — PDF-extracted bodies have no
  paragraph breaks to begin with. Claim-extraction will have to
  paragraph-split at its own layer, or inherit whole-section chunks.
- **Inline citation extraction** — keep `inline_cites=[]` for
  `source='ads_body'` rows; that belongs to a separate bibliographic-
  parsing pass.

## Deliverables

### D1. Rewritten parser

`src/scix/sources/ads_body_parser.py` — in-place rewrite, no new
module. Contract-compatible surface: same `parse_ads_body(body,
bibstem) -> (sections, meta)` signature, same `Section` dataclass,
same `compute_confidence` shape.

- `PARSER_VERSION = "ads_body_inline_v2"` (bumped from
  `ads_body_regex@v1`)
- Section-marker vocabulary (case-insensitive match, canonical
  uppercase on emit, word-boundary anchored):
  `ABSTRACT`, `INTRODUCTION`, `BACKGROUND`, `RELATED WORK`,
  `OBSERVATIONS`, `DATA`, `METHODS`, `METHODOLOGY`, `MODEL`,
  `THEORY`, `ANALYSIS`, `RESULTS`, `DISCUSSION`, `SUMMARY`,
  `CONCLUSION`, `CONCLUSIONS`, `REFERENCES`, `BIBLIOGRAPHY`,
  `ACKNOWLEDGMENTS`, `ACKNOWLEDGEMENTS`, `APPENDIX`,
  `APPENDIX A`–`APPENDIX Z`
- Handle numbered prefixes before the keyword: `1 INTRODUCTION`,
  `1. Introduction`, `1.1 Background`. Strip numeric prefix,
  preserve canonical heading.
- Minimum-section threshold: only emit sections if ≥ 2 distinct
  canonical markers hit; otherwise return `[]` (avoids single-marker
  false positives).
- `Section` fields: `heading` (canonical casing), `level=1`,
  `text` (body slice from end-of-heading to start-of-next-heading or
  EOF), `offset` (start char position of the heading match).
- Confidence: keep existing formula; recomputed on new metrics.

### D2. Re-parse workflow in the driver

`scripts/populate_papers_fulltext.py` — add an explicit reparse mode
rather than ad-hoc SQL gymnastics.

- New CLI flag: `--reparse-from-version <OLD_VERSION>` (mutually
  exclusive with the default new-row mode).
- When set, `iter_candidate_papers` selects
  `papers_fulltext.bibcode` where `parser_version = OLD_VERSION AND
  source = 'ads_body'`, joining back to `papers` for the body text.
- The write path switches from `COPY` to an `INSERT ... ON CONFLICT
  (bibcode) DO UPDATE SET sections = EXCLUDED.sections, inline_cites
  = EXCLUDED.inline_cites, parser_version = EXCLUDED.parser_version,
  parsed_at = now()`. Accepts a per-batch throughput hit (~1000 rps)
  in exchange for in-place updates and transactional safety.
- `DriverStats` gains a `reparsed: int` counter.
- Resume semantics unchanged: re-run with the same flag picks up
  bibcodes that still have the old version.

### D3. Parser unit + integration tests

- Unit (`tests/test_ads_body_parser.py` updates): synthetic flat bodies
  with and without canonical markers; numbered-prefix handling;
  minimum-threshold behaviour; word-boundary edge cases (e.g.,
  `METHODOLOGY` must not match `METHODS`); case-insensitive match with
  canonical uppercase emit; overlapping candidates (`CONCLUSION` vs
  `CONCLUSIONS` — pick the longer).
- Integration (`tests/test_populate_papers_fulltext.py` updates): run
  the driver with `--reparse-from-version` against a seeded 1K-paper
  slice in `scix_test`, assert section_empty count drops below 30 %.

### D4. Pre-prod canary + full re-parse

- Stratified-sample canary: pull 10K papers from prod into `scix_test`
  (stratify by `substring(bibcode,1,4)` decade × top-10 bibstems),
  run reparse, inspect section distribution + spot-check 50 papers.
- Full production reparse via `scix-batch`, same pattern as the
  initial run. Log to `logs/reparse_papers_fulltext_<date>.log`.

### D5. Runbook update

`docs/runbook_populate_papers_fulltext.md` — add a **Re-parse** section
documenting `--reparse-from-version`, when to use it (parser_version
bump), and the expected duration/throughput delta vs fresh runs.

## Acceptance

- **Functional**
  - `PARSER_VERSION == "ads_body_inline_v2"` in the shipped parser.
  - `populate_papers_fulltext.py --reparse-from-version
    ads_body_regex@v1` updates existing `papers_fulltext` rows in
    place; no new rows created.
  - Existing `pytest tests/test_ads_body_parser.py` and
    `tests/test_populate_papers_fulltext.py` pass.
- **Quality (measured on full prod post-reparse)**
  - `section_empty` rate across all `papers_fulltext` rows ≤ 30 %.
  - Median `jsonb_array_length(sections)` for rows where
    `substring(bibcode,1,4) >= '2010'` is ≥ 4.
  - New `papers_fulltext_failures` rows from the reparse < 0.1 % of
    reparsed bibcodes.
- **Operational**
  - Reparse runtime ≤ 2 × initial-run time (≤ 4 h hard cap).
  - Zero OOMs, zero batch-level aborts; driver remains resumable.
- **Validation**
  - Manual spot-check of 50 randomly-sampled reparsed papers: ≥ 45
    have correct-looking headings (noted in a test-report file).

## Dependencies

- **Upstream:** none — all code lives in this repo.
- **Downstream consumers unblocked:**
  - `prd_nanopub_claim_extraction.md`
  - `wqr.9` section embeddings
  - Future section-aware MCP tools

## Parallel execution notes (for dispatched agent)

- Branch off `main` at the post-structural-fulltext tip (≥ `25a95f8`
  + the `INVOCATION_ID` and SQL-filter follow-up fixes).
- Parser rewrite (D1) and driver reparse-flag (D2) can proceed in
  parallel; they merge cleanly (parser is a pure function, driver
  change is isolated to `iter_candidate_papers` + write path).
- Heavy DB writes during D4 — use `scix-batch` per CLAUDE.md.
  `--mem-high 6G --mem-max 10G` is a reasonable default; reparse is
  lighter on RSS than initial because no stats-on-body COPY.
- `SCIX_TEST_DSN=dbname=scix_test` for all development. Migrations
  041 + 047 must be applied to `scix_test`.

## Open questions (flag, do not block)

- **Heading casing** — emit canonical uppercase (`INTRODUCTION`), or
  preserve the body's casing (`Introduction` / `INTRODUCTION`)?
  Default: canonical uppercase for consistency; downstream can alias.
- **Section-name drift over time** — older papers may use `SUMMARY`
  instead of `CONCLUSIONS`, or have no explicit `RESULTS` heading.
  Audit the post-reparse section-name histogram and expand vocabulary
  if a clear gap shows up.
- **Multiple appendices** — `APPENDIX A`, `APPENDIX B`, etc. should
  each become their own section. Vocabulary entry `APPENDIX [A-Z]` as
  a single regex catches this; verify on spot-check.
- **False positive on inline prose** — e.g. "as discussed in the
  Discussion section". The ≥ 2-markers threshold and word-boundary
  anchor mitigate, but verify on spot-check.

## Risks

- **Parser v2 regresses on v1-section papers.** The 295K papers that
  got exactly 1 section under v1 may go to 0 sections under v2 if
  they only have one canonical marker. Mitigate: add a fixture-based
  CI test comparing v1-vs-v2 section counts on a 1K stratified sample;
  v2 section-count must dominate pairwise.
- **`ON CONFLICT DO UPDATE` is slower than COPY.** Per-row UPDATE
  vs per-batch COPY roughly halves throughput — realistic budget
  500–1000 rps instead of 2000. Acceptable; < 8 h wall clock.
- **Mid-run crash mixes v1 and v2 rows.** Expected and handled —
  re-running with `--reparse-from-version ads_body_regex@v1`
  naturally picks up only the remaining v1 rows.
- **Establishing the re-parse workflow.** The structural-fulltext
  PRD explicitly deferred "version-bump rollout". This PRD makes it
  real. Future bumps reuse the same `--reparse-from-version` path —
  no new scaffolding needed.
