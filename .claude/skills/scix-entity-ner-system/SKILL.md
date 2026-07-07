---
name: scix-entity-ner-system
description: >
  Maintainer guide to the SciX cross-discipline entity graph and NER system:
  the entities / document_entities schema (57.7M paper-entity links), the
  three extraction lanes (LLM batch extraction, lexical linking, the GLiNER
  dbl.3 zero-shot pass), the INDUS post-classifier, the per-bucket precision
  profile and lower-bound honesty discipline, the eq95 denylist, the JIT
  resolver lane (bulkhead / canary / stubs), and the ZFC rule (classify via
  the entity graph, never keyword heuristics). Load when working on
  src/scix/extract/, src/scix/jit/, entity_resolver, resolve_entities,
  document_entities, NER precision, GLiNER, ner_quality_profile, or the
  entity MCP tool's internals. NOT for using the MCP tools as a literature
  researcher — use scix-mcp. NOT for the MCP tool cap / contract regen — use
  scix-mcp-tool-surface. NOT for citation communities / PageRank — use
  scix-citation-graph. NOT for embedding ingest — use scix-embedding-pipeline.
---

# SciX Entity & NER System

Audience: an engineer or agent with zero prior context who must debug, extend,
or evaluate the entity graph. Everything below was verified by reading source
at the provenance pin (bottom of this file) on 2026-07-07. Heavy commands are
shown with their guards and marked **do not run casually** — this host
co-hosts production services and an agent supervisor.

## When NOT to use this skill

| You want to…                                                | Use instead                        |
| ----------------------------------------------------------- | ---------------------------------- |
| Query entities as a research agent (MCP tool usage)         | `scix-mcp` (query-side skill)      |
| Add/rename an MCP tool, regen the contract, the 15-tool cap | `scix-mcp-tool-surface`            |
| Understand RRF fusion / dense lane / alias lexical lanes    | `scix-retrieval-architecture`      |
| Run any multi-GB job safely on this host                    | `scix-memory-and-batch-discipline` |
| DSN guards, prod-DB protection, query_log telemetry         | `scix-db-safety-and-telemetry`     |
| Communities, PageRank, claim_blame provenance               | `scix-citation-graph`              |
| What counts as evidence, gold sets, judging                 | `scix-eval-and-evidence`           |

## 1. The entity graph in one page

Jargon, defined once:

- **Entity**: a canonical scientific thing (a telescope, a software package, a
  gene) with one row in `entities`.
- **Mention**: a surface string in a paper that (maybe) refers to an entity.
- **Link**: a `document_entities` row tying a bibcode to an entity, with
  provenance (`match_method`, `tier`, `confidence`, `evidence` JSONB).
- **NER**: named-entity recognition — extracting typed mentions from text.
- **GLiNER**: a zero-shot NER model family; you give it label strings at
  inference time, no per-label training.
- **INDUS**: `nasa-impact/nasa-smd-ibm-st-v2`, the local 768d sentence
  encoder used across SciX (also the dense retrieval lane).
- **dbl**: the cross-discipline entity expansion epic; **dbl.3** is the
  GLiNER backfill + quality-eval bead this system's discipline comes from.

Tables (created in `migrations/021_entity_graph.sql`, hardened in `028`):

| Table                         | Role                              | Key facts                                                                                                                                                       |
| ----------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entities`                    | canonical records                 | `UNIQUE (canonical_name, entity_type, source)`; `ambiguity_class` + `link_policy` enums (028); ~9M rows (README, 2026-06)                                       |
| `entity_aliases`              | alternate names                   | `PRIMARY KEY (entity_id, alias)`; lower(alias) index                                                                                                            |
| `entity_identifiers`          | external IDs (Wikidata QID, DOI…) | `PRIMARY KEY (id_scheme, external_id)`                                                                                                                          |
| `entity_relationships`        | entity↔entity edges               | `UNIQUE (subject, predicate, object)`                                                                                                                           |
| `document_entities`           | paper↔entity bridge               | PK `(bibcode, entity_id, link_type, tier)` after 028; **57.7M rows** (README + `docs/mcp_tool_audit_2026-04.md:181`, dated 2026-04); no FK on bibcode by design |
| `document_entities_jit_cache` | JIT lane cache                    | migration 034; write-restricted (see §6)                                                                                                                        |

`document_entities.tier` semantics: migration 028's comment defines 0–3
(0=legacy, 1=high-precision, 2=alias+context, 3=LLM-adjudicated); the GLiNER
pass writes **tier 4** ("ML/NER signal", `NER_TIER = 4` in
`src/scix/extract/ner_pass.py`). Treat the code constant as operational truth;
the 028 comment predates dbl.3.

### Two type vocabularies — do not conflate them

This is the most common newcomer confusion:

1. **Extraction payload categories (plural)** — keys inside
   `extractions.payload` JSONB. v1: `methods, datasets, instruments,
materials` (4). v3: + `observables, software` (6). The MCP `entity`
   tool's `action='search'` accepts exactly the plural four
   (`_VALID_ENTITY_TYPES` in `src/scix/mcp_handlers/entity.py`).
2. **`entities.entity_type` values (singular)** — what the graph and GLiNER
   write: `software, dataset, method, organism, chemical, gene, instrument,
mission, location` (the 9-label GLiNER map) plus harvest-sourced types
   (`target, observable, observatory, taxon`, …).

README's "13 vocabularies" (gene, software, mission, organism, target,
observable, chemical, location, taxon, plus methods/datasets/instruments/
materials) mixes both vocabularies into one marketing line. The live distinct
`entity_type` inventory is a DB fact; the one-line check is in Provenance.
UNVERIFIED at authoring time (read-only session, no DB connections).

## 2. The three extraction lanes

### Lane A — LLM batch extraction (`src/scix/extract/__init__.py`)

Anthropic **Messages Batches API** extraction over abstracts (v1: 4 plural
categories; v3: 6, adds body excerpt up to `_V3_MAX_BODY_CHARS = 6000`).
Tool-use schema forces structured output (`extract_entities` /
`extract_entities_v3`); few-shot examples are embedded in the module. Results
checkpoint to local JSONL **before** any DB write, then upsert into
`extractions` keyed on `(bibcode, extraction_type, extraction_version)`.

Cost discipline: `estimate_cost()` against a pinned `_MODEL_COSTS` table
(dated 2026-04 — re-check prices before any new run); pipeline refuses to
start if estimate + cumulative > budget (default $10) and halts at 80% of
budget (`BudgetExceededError`). Cumulative cost + processed bibcodes persist
in `data/extractions/.checkpoint_{version}.json`.

**File-layout trap:** `src/scix/extract.py` and `src/scix/extract/__init__.py`
are byte-identical committed duplicates (both 53,471 bytes at the pin). Python
imports the **package**, so an edit to `src/scix/extract.py` alone is a silent
no-op. Edit the package `__init__.py`; converging the duplicates deserves its
own bead (route via scix-change-control).

Run (needs `ANTHROPIC_API_KEY`; writes to the DB — **do not run casually**,
and never against prod without the DSN discipline in
scix-db-safety-and-telemetry):

```
# v3 pipeline is invoked from Python (no dedicated CLI script at the pin):
# scix.extract.run_extraction_pipeline(dsn=..., pilot_size=..., budget_usd=...)
```

### Lane B — Lexical linking (`src/scix/link_entities.py` and friends)

Dictionary-driven: resolves extracted mention strings against
`entities`/`entity_aliases` via a bulk-loaded caching resolver, writes
`document_entities` with lexical `match_method` values
(`keyword_exact_lower`, `aho_corasick_abstract`, `canonical_exact`,
`alias_exact` — see `src/scix/aho_corasick.py`, `scripts/link_tier1.py`,
`link_tier2.py`, `link_incremental.py`). Precision is high by construction
(matched a curated vocabulary): the profile assigns a blanket
`LEXICAL_PRECISION_DEFAULT = 0.95`. The vocabularies themselves come from the
`scripts/harvest_*.py` family (ASCL, AAS facilities, CMR/GCMD, PDS4, PhySH,
Papers-with-Code methods, SBDB, SPASE, VizieR, Wikidata enrichment, UAT).

Limitation that motivated Lane C: the dictionaries are astro-skewed, so
lexical-only coverage misses the long tail (software in cs.CL, datasets in
q-bio, organisms in biomed).

### Lane C — GLiNER zero-shot pass (dbl.3, `src/scix/extract/ner_pass.py`)

The architectural shift: zero-shot NER that needs no pre-existing dictionary.

Pinned facts (module constants — re-verify before trusting, commands in
Provenance):

| Constant                  | Value                                | Meaning                                                                                       |
| ------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------- |
| `DEFAULT_MODEL_NAME`      | `gliner-community/gliner_large-v2.5` | pinned for reproducible mentions                                                              |
| `NER_SOURCE_VERSION`      | `gliner_large-v2.5/v1`               | stamped to `entities.source_version`; **bump on any model or label change**                   |
| `NER_LABELS`              | 9 labels, frozen v1                  | software, dataset, method, organism, chemical, gene_or_protein, instrument, mission, location |
| `LABEL_TO_ENTITY_TYPE`    | 9→9 map                              | `gene_or_protein` collapses to `gene`; rest pass through                                      |
| `DEFAULT_CONFIDENCE`      | 0.7                                  | mention floor (bead spec)                                                                     |
| `NER_TIER`                | 4                                    | tier written to `document_entities`                                                           |
| `DEFAULT_INFERENCE_BATCH` | 8                                    | 5090 fp16 VRAM headroom                                                                       |
| `DEFAULT_MAX_TEXT_CHARS`  | 5000                                 | GLiNER-large truncates at 768 tokens anyway                                                   |

Pipeline: keyset-paginated bibcode-ordered batches → `batch_predict_entities`
→ canonicalize (`lower`, strip trailing parenthetical, collapse whitespace,
**punctuation kept** — "CRISPR-Cas9", "p53") → per-doc dedup (best confidence
per `(canon, type)`) → upsert `entities` (`source='gliner'`) → upsert
`document_entities` (`link_type='mentions'`, `match_method='gliner'`,
tier 4, confidence = `GREATEST(old, new)`).

Resumable + idempotent: each batch checkpoints in `ingest_log` under
`ner_pass:{target}:{first_bibcode}`; killed runs resume at the next
un-checkpointed batch. Re-running with a higher confidence floor refines,
never duplicates.

**OA gate:** `iter_paper_batches(oa_only=True)` is the default — it gates on
`papers_is_oa_or_preprint(papers)` (migration 068). Body-text AI on
closed-access papers is a publisher TDM-clause risk; `--include-closed` is an
explicit operator opt-in. Never remove this gate.

Battle-tested production invocation (from the CLI docstring, dated
2026-04-25; **heavy — GPU + prod DB writes — do not run casually**;
`scix-batch` is this installation's mandatory memory-scoped wrapper, see
scix-memory-and-batch-discipline):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    scix-batch --mem-high 16G --mem-max 24G \
    python scripts/run_ner_pass.py \
        --target abstract \
        --batch-size 1000 \
        --inference-batch 8 \
        --max-text-chars 3500 \
        --require-batch-scope
```

Why those flags: activation memory scales O(seq_len²), so `--inference-batch
8` (not 16) avoids single-long-abstract VRAM spikes at ~zero throughput cost
(~75 docs/s either way); `--max-text-chars 3500` stops paying attention
compute for text past the 768-token window; `expandable_segments` lets torch
reuse fragmented VRAM. `--require-batch-scope` makes the script refuse to run
unless `SYSTEMD_SCOPE` is set (i.e. it was launched via the wrapper).

Safe local sanity check (no DB writes, still loads the model — run under
scix-batch on this host):

```bash
scix-batch python scripts/run_ner_pass.py --target abstract --max-papers 200 --dry-run
```

Body-text variant: `scripts/run_ner_bodies.py` is section-aware — it parses
sections, keeps only roles `{'method','result'}` (drops ~60% of body tokens
and the references section, which otherwise floods the table with author
surnames mis-typed as location/organism), and writes to `staging.extractions`
with `extraction_type='ner_body'`. Same OA gate.

## 3. The dbl.3 quality discipline (the load-bearing part)

### Two-stage NER

GLiNER span detection is high-recall but its **typing** is noisy (~58%
modern / ~54% pre-1990 aggregate precision in the 2026-04-25 evals). Stage 2
(`src/scix/extract/ner_classifier.py`) re-types each mention by cosine
similarity to per-type **INDUS anchor centroids** (~150 hand-curated
examples per type with usage sentences, `data/ner_anchors/seed_v1.json`;
embedding input is `"MENTION | CONTEXT"`, mirrored exactly between anchors
and live mentions). Where the classifier disagrees with GLiNER, the mention
gets `agreement=false` and downstream tools default-filter it.

The post-pass driver (`src/scix/extract/ner_classify_pass.py`,
CLI `scripts/run_ner_classify_pass.py` — wrap in scix-batch) streams
`document_entities` rows where `match_method='gliner'` and
`NOT (evidence ? 'agreement')`, then merges
`{classifier_type, classifier_score, agreement}` into `evidence` JSONB.
Naturally idempotent (the filter skips judged rows); checkpoints under
`ner_classify_pass:{first_bibcode}`.

### The precision profile — source of truth

`src/scix/extract/ner_quality_profile.py`. Empirical numbers from the dbl.3
acceptance eval: **414 hand-judged mentions** (207 pre-1990 + 207 modern
stratified samples; the raw judgments are committed at
`docs/eval/dbl3_ner_precision_judgments_200{,_modern}.jsonl`).

**THE RULE:** the bead acceptance criterion "≥80% precision at conf≥0.7" is
**NOT met in aggregate**. It IS met for several `(entity_type, era,
agreement)` buckets. Never state an aggregate precision for GLiNER output;
always report per-bucket, and **when the bucket is unknown, use the LOWER
bound** (that is exactly what `precision_estimate()` implements: unknown
year → the pre-1990 bucket; unknown bucket → 0.5; classifier-rejected →
0.2 rough ceiling; non-gliner sources → 0.95 lexical default).

Classifier-filtered (`agreement=true`) precision, the numbers agents see
(small-n buckets are estimates only — `CLASSIFIER_FILTERED_N` carries n):

| type       | pre-1990        | modern (≥2010)  | verdict                |
| ---------- | --------------- | --------------- | ---------------------- |
| instrument | 0.67 (n=6)      | **1.00** (n=4)  | modern passes (tiny n) |
| method     | 0.64 (n=11)     | **0.86** (n=14) | modern passes          |
| location   | **0.86** (n=14) | **0.86** (n=7)  | passes both            |
| mission    | 0.47 (n=15)     | **0.83** (n=12) | modern passes          |
| software   | 0.20 (n=5)      | 0.71 (n=7)      | close, modern only     |
| chemical   | 0.76 (n=17)     | 0.46 (n=13)     | pre-1990 near-pass     |
| organism   | 0.62 (n=16)     | 0.50 (n=4)      | fails                  |
| dataset    | **0.00** (n=3)  | 0.40 (n=5)      | broken                 |
| gene       | 0.33 (n=6)      | **0.18** (n=11) | broken                 |

History lesson baked into the module docstring: the original partial-coverage
pre-1990 numbers (n=14) overstated buckets dramatically (mission 100%→47%,
organism 100%→62%) — the values above are the corrected full-coverage set.
That is the incident behind the lower-bound rule: **small-n optimism is how
this system lies to you.**

`precision_band(p)`: `high` ≥0.85, `medium` ≥0.70, `low` ≥0.50, else
`noisy`. These bands ride on every MCP `entity(action='papers')` result row
(`precision_estimate` + `precision_band`), so agents apply their own
min-precision filter instead of getting an opaque pass/fail.

**Doc-drift warning:** the profile's docstrings cite
`docs/eval/dbl3_ner_precision_eval_2026-04-25{,_modern}.md` and
`dbl3_ner_classifier_filtered_eval.md` — those markdown reports are **not in
the tree** at the pin (only the judgments/sample JSONLs are). The constants in
`ner_quality_profile.py` are the surviving source of truth; re-derivation is
`scripts/reeval_classifier_filtered.py` (reads the 414 judgments + live
classifier verdicts — needs DB).

### The eq95 denylist

`src/scix/extract/ner_denylist.py`: a small mechanical
`(canonical_name_lower, entity_type)` exclusion set applied **at the agent
surface** (entity resolve/papers paths), not at extraction time — the
`document_entities` rows still exist. Three entries alone account for 163K
spurious mentions verified against prod 2026-04-26 (`'experimental data'`/
dataset 80,540; `'data'`/dataset 55,423; `'method'`/method 26,976). Pair
granularity is deliberate: `'protein'` is denylisted as `gene` without hiding
a hypothetical `'protein dynamics'` method.

To extend: add the tuple to `_DENYLIST`, run
`pytest tests/test_ner_denylist.py`, document the discovery context in the
module docstring. An insert-time filter + backfill DELETE is explicitly a
separate, more invasive change (route through change control).

## 4. The ZFC rule: classify via the entity graph, never keywords

ZFC ("Zero Framework Cognition", this codebase's convention): orchestration
code may do IO, structural validation, and mechanical transforms — it may NOT
make semantic judgments with keyword tables or hardcoded heuristics.

The incident (bead `scix_experiments-aptn`, commit `1f377bf`, 2026-06-22):
`src/scix/viz/api.py`'s `demo_disambig` used `_guess_entity_type` + a
23-entry `_ENTITY_TYPE_HINTS` keyword→bucket table to pick a typed-search
entity_type. It misclassified multi-word queries ("magnetic field model",
"Mars survey", "JWST pipeline"). The fix deleted the table and function:
entity_type now comes from **the resolved candidates' own `entity_type`** —
the entity graph's classification — and the typed-search step is _skipped_
rather than fabricated when no candidate maps to a searchable bucket.

Operational rule for any new code: if you need an entity's type, resolve the
mention (`EntityResolver` cascade: exact canonical 1.0 → alias 0.9 →
identifier 0.85 → optional pg_trgm fuzzy at similarity score) and read
`entity_type` off the candidate. If you catch yourself writing
`if 'telescope' in query: type = 'instrument'`, stop — that pattern was
removed once already. Mechanical exceptions that ARE allowed (each carries an
in-code ZFC note): the denylist (exact-match exclusion), lafia/somd pattern
detectors (`src/scix/extract/lafia.py`, `somd_detect.py` — cue detection, not
semantic judging), and calibrated-threshold similarity matching.

## 5. Agent-facing surface (internals only — cap/contract lives elsewhere)

`src/scix/mcp_handlers/entity.py`, one `entity` tool, four actions:

| action    | input                | reads                                    | notes                                                                                                                                                                                       |
| --------- | -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolve` | free text            | entities/aliases/identifiers             | denylist-filtered; `fuzzy=true` opt-in                                                                                                                                                      |
| `papers`  | `entity_id` or query | `document_entities` join                 | attaches `precision_estimate`/`precision_band` per row from `(entity_type, source, evidence->>'agreement', year)`; profile lookup is best-effort (never breaks the response, logs at debug) |
| `profile` | `entity_id`          | full entity profile                      | folded-in `entity_context` (bead 9afa)                                                                                                                                                      |
| `search`  | plural type + query  | legacy `extractions` payload containment | plural 4 only; claim/finding types rejected at the front door with a structured error pointing to `claim_search`                                                                            |

Any change to this tool's _surface_ (params, actions, errors) must go through
the contract regen + 15-tool-cap discipline — `scix-mcp-tool-surface`.

## 6. The JIT lane (`src/scix/jit/`) — mostly stubs, know what's real

PRD §M11a/b/c + §M13. What is REAL at the pin: the bulkhead (async
concurrency limiter, `DEFAULT_BUDGET_MS = 400` hard wall including permit
acquisition), the router policy (5% canary share `CANARY_SHARE = 0.05`;
degrade chain live_jit → local_ner → `STATIC_CORE_FALLBACK` sentinel), the
partitioned TTL cache over `document_entities_jit_cache` (migration 034),
and the tests (`tests/test_jit_*.py`).

What is STUB: `local_ner.py` is a **deterministic CPU echo stub** (returns
every candidate at fixed confidence 0.75, `scibert-stub-v1`) and
`call_live_jit` echoes candidates at 0.95 — production Haiku/SciBERT
inference has NOT landed. Do not present JIT resolution quality numbers as
real; there are none. `docs/prd/ner_gpu_deployment.md` holds the deployment
plan (INDUS-NER-DEAL primary, with the explicit caveat it was trained on
DEAL/WIESP facilities+celestial-objects — 2 of the 6 v3 categories).

**Write-boundary invariant (M13):** `src/scix/resolve_entities.py` is the
ONLY module allowed to write `document_entities` /
`document_entities_jit_cache`, statically enforced by
`scripts/ast_lint_resolver.py` (libcst walk, fails CI). The tier producers
(`ner_pass.py`, `ner_classify_pass.py`, `jit/cache.py`) carry explicit
`# resolver-lint: bypass` markers because the resolver is currently
read-side-only; unit u10 is slated to route tier writes through it. If you
add a `document_entities` write anywhere else without a justified bypass
marker, CI fails — that is intended. Do not scatter new bypass markers.

## 7. Change checklists

**Adding or changing a GLiNER label** — treat as **HALT-branch-ready:
requires Stephanie's sign-off (PROVISIONAL pending Stephanie, discovery Q5)**:

1. `NER_LABELS` is frozen v1; a label change is a v2 change. Bump
   `NER_SOURCE_VERSION` so downstream can filter by producing schema.
2. Extend `LABEL_TO_ENTITY_TYPE`; decide whether the new type collapses
   (as gene_or_protein→gene did).
3. Add the type to the precision-profile eval plan BEFORE backfilling: a
   type with no judged bucket surfaces at the 0.5 fallback — worse than
   honest absence.
4. Tests ship with the change (`tests/test_ner_pass.py` +
   `test_ner_quality_profile.py`).

**Re-running the backfill** (model bump, higher floor): idempotent by
construction, but budget the run via scix-batch, keep `--since-bibcode` for
manual resume, and re-run the classifier post-pass after (new rows have no
`agreement` verdict → they surface at unfiltered precision until judged).

**Extending the denylist**: 4-step procedure in §3.

**Refreshing the precision numbers**: `scripts/reeval_classifier_filtered.py`
(414 judgments + live verdicts), `scripts/dbl6_label_diff.py`,
`scripts/eval_ner_wiesp.py` (WIESP benchmark), `scripts/canary_ner.py`
(nightly drift canary for `nasa-smd-ibm-v0.1_NER_DEAL`). All need DB access —
observe the DSN discipline. If numbers move, update
`ner_quality_profile.py` constants AND commit the regenerated report next to
the judgments in `docs/eval/` (fixing the missing-report drift noted in §3).

## 8. Known open edges (stated plainly, 2026-07-07)

- `dataset` and `gene` buckets are broken (0.00–0.40 / 0.18–0.33). No fix
  landed at the pin; candidate directions live in the dbl epic beads.
- Local NER and live-JIT lanes are stubs (§6); the JIT lane serves no real
  inference today.
- The cited dbl3 markdown eval reports are absent from the tree (§3).
- `src/scix/extract.py` / `extract/__init__.py` duplication (§2A).
- Later dbl passes (lafia dbl.20/21, somd dbl.22 — software/dataset mention
  confirmation) live in `src/scix/extract/lafia*.py`, `somd_detect.py` with
  eval notes in `docs/eval/dbl2*_*.md`; they are adjacent refinements, not
  covered in depth here.
- The live distinct `entity_type` inventory and current `document_entities`
  row count are DB facts this skill could not verify read-only; the 57.7M /
  ~9M / 13-type figures are dated 2026-04→06 from README + tool audit.

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit` @
`452ab86` (not `main`; at authoring time HEAD was an ancestor of
`origin/main`, 5 commits behind, all architecture-docs/CI — no entity code
diverged). Working tree carried uncommitted embedding-pipeline changes
(bead s7cy, in-flight); none touched the entity/NER files documented here.

One-line re-verification (all read-only):

```bash
git -C . branch --show-current && git rev-parse --short HEAD
# constants drifted?
grep -n "DEFAULT_MODEL_NAME\|NER_SOURCE_VERSION\|DEFAULT_CONFIDENCE\|NER_TIER\|NER_LABELS" src/scix/extract/ner_pass.py
grep -n "UNFILTERED_PRECISION\|CLASSIFIER_FILTERED_PRECISION\|LEXICAL_PRECISION_DEFAULT" src/scix/extract/ner_quality_profile.py
# extract.py duplication still present?
cmp -s src/scix/extract.py src/scix/extract/__init__.py && echo "still identical" || echo "diverged/converged — update §2A"
# write-boundary lint still enforced?
grep -rn "resolver-lint: bypass" src/scix/ | wc -l && ls scripts/ast_lint_resolver.py
# denylist size
grep -c '("' src/scix/extract/ner_denylist.py
# judgments still 414?
wc -l docs/eval/dbl3_ner_precision_judgments_200.jsonl docs/eval/dbl3_ner_precision_judgments_200_modern.jsonl
# entity tests still collected (safe, no DB):
pytest --collect-only -q tests/test_ner_pass.py tests/test_ner_classifier.py tests/test_ner_quality_profile.py tests/test_ner_denylist.py 2>/dev/null | tail -1
# live type inventory (prod read — needs DB + DSN discipline, see scix-db-safety-and-telemetry):
#   SELECT entity_type, count(*) FROM entities GROUP BY 1 ORDER BY 2 DESC;
#   SELECT count(*) FROM document_entities;
```

PROVISIONAL dependencies on the Phase-1 discovery answers: Q5 (NER-label
additions treated as HALT-branch-ready pending Stephanie's word — §7) and Q4
(this skill is written repo-portable; the `scix-batch` wrapper is this
installation's operational requirement, not a repo artifact).
