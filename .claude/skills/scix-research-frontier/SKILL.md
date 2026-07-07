---
name: scix-research-frontier
description: >
  The executable research campaign on SciX's hardest live problem — retrieval-quality
  integrity ("does the INDUS dense lane earn its place, and does hybrid survive a
  skeptic?") — plus the other open frontier (claim_blame has no real-corpus accuracy
  number) and the ADASS external-positioning bar (what may be claimed, reproducibility
  standards). Load when: planning or running a fusion-calibration sweep, interpreting
  dense-vs-BM25 nDCG numbers, evaluating a stronger dense lane, deciding what the ADASS
  paper can claim, building the claim_blame gold set, or asked "is hybrid retrieval
  actually better here?". NOT for how the RRF/lane code is wired — use
  scix-retrieval-architecture. NOT for eval-metric mechanics or the gold-set catalog —
  use scix-eval-and-evidence. NOT for how changes get gated/approved — use
  scix-change-control. NOT for running the embed/ingest pipeline — use
  scix-embedding-pipeline.
---

# SciX Research Frontier — the retrieval-integrity campaign and the publication bar

Date-stamped 2026-07-07. Authored from committed reality at branch
`bd/0yp5-external-copy-accuracy-audit`, HEAD `452ab86` (NOT `main`; this rig's work
lands on `bd/*` branches). All numbers below come from artifacts in `results/` and
bead prose, never from memory. Where a fact depends on a provisional Phase-1 answer
it is marked **PROVISIONAL pending Stephanie (Qn)**.

## When to use this skill / when not

| You are about to…                                                                    | Use                                |
| ------------------------------------------------------------------------------------ | ---------------------------------- |
| Run/interpret a fusion sweep, judge the dense lane, plan a stronger-dense experiment | **this skill**                     |
| Decide what the ADASS paper may claim; check a draft against the evidence            | **this skill**                     |
| Build the claim_blame real-corpus gold set (bead `6ajy`)                             | **this skill**                     |
| Understand how `hybrid_search`/RRF/lanes are implemented                             | `scix-retrieval-architecture`      |
| Look up metric definitions, gold-set formats, judge harnesses                        | `scix-eval-and-evidence`           |
| Get approval for an ADR-pinned change (new lane, new model, quantization)            | `scix-change-control`              |
| Fix the broken embed path / Qdrant sync                                              | `scix-embedding-pipeline`          |
| Size or schedule a heavy run on this host                                            | `scix-memory-and-batch-discipline` |

## Definitions (once)

| Term                            | Meaning here                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Lane**                        | One retrieval signal producing a ranked list: INDUS dense (Qdrant), title/abstract BM25 (PG tsvector), body BM25 (PG, ~46% coverage) |
| **RRF**                         | Reciprocal Rank Fusion: fuse lanes by `sum(1/(k+rank))`; production `RRF_K = 60` (`src/scix/search.py:34` at HEAD)                   |
| **INDUS**                       | `nasa-impact/nasa-smd-ibm-st-v2`, 768d local embedding model; the only full-corpus dense lane                                        |
| **nDCG@10 / MRR@10 / Recall@k** | Standard rank metrics; the sweep reports all three per fusion config                                                                 |
| **Gold set**                    | JSONL of `{query, bucket, gold_bibcodes}`; the instrument a number is measured on                                                    |
| **Dense-dominant instrument**   | A gold set on which `dense_only` beats `bm25_only`; whether one exists corpus-wide is the open question                              |
| **claim_blame**                 | MCP tool tracing a claim to its earliest non-retracted origin via the citation graph                                                 |

## 1. State of the evidence (2026-07-07) — the three instruments

The same retrieval stack has been measured on three gold sets. They tell three
different stories. **Never quote a number without naming its instrument.**

| Instrument                                                          | n     | dense_only nDCG@10 | bm25_only nDCG@10 | Best hybrid                                    | Artifact                                                                                                                                                                     |
| ------------------------------------------------------------------- | ----- | ------------------ | ----------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense-sensitive set (bead `dfba`, 2026-06-15)                       | 27q   | **0.864**          | 0.088             | naive RRF _hurt_ vs dense-alone                | bead prose; only the pilot backup (`results/dense_sensitive/_pilot_backup/ab_eval.json`, pilot overall 0.893) is in this worktree — full-run artifact **unverified in-tree** |
| 50q curated (`results/fusion_sweep_v1.md`, 2026-06)                 | 50q   | 0.0399             | 0.0756            | weighted_sum(0.5) 0.0727 — **diagnostic only** | `results/fusion_sweep_v1.{md,json}`                                                                                                                                          |
| 1200q recall gold (`results/fusion_sweep_1200q.md`, run 2026-06-20) | 1200q | **0.3803**         | **0.4291**        | **naive_rrf(k=60) 0.4786 (+0.0983 vs dense)**  | `results/fusion_sweep_1200q.{md,json,log}`                                                                                                                                   |

The honest headline — **PROVISIONAL pending Stephanie (Q3)**:

> Naive RRF (k=60) lifts nDCG@10 by +0.098 over dense-alone on the 1200q recall gold
> set, but the dense lane underperforms BM25 on that set (0.3803 vs 0.4291), so the
> lift partly reflects outrunning a weak dense lane, not a clean fusion gain. A dense
> lane that beats BM25 corpus-wide is an **open frontier**, never a precondition
> stated as met. INDUS/dense superiority is never stated as fact.

Three consequences, all load-bearing:

1. **The production config is already the sweep winner.** `hybrid_search` defaults to
   `rrf_k = RRF_K = 60` (`src/scix/search.py:34` and the `hybrid_search` signature at
   HEAD). The 1200q sweep validates the default; it does **not** warrant a config
   change. The k-sweep is flat (k=10..100 spans 0.4767–0.4786, ≤0.002): k is not the
   lever.
2. **The `dfba` 0.864 number is instrument-specific.** It came from a purpose-built
   dense-sensitive 27q set and was shipped for ADASS "characterized honestly as
   on-disk serving" (bead `dfba`, closed by Stephanie's reply banked in `4en9`).
   Quoting it as corpus-general is the single easiest way to fail a skeptic.
3. **The premise inversion is itself the finding.** `fusion_sweep.py` was built to fix
   "naive RRF hurts vs a dominant dense lane" (bead `isve`); on both broader sets the
   dense lane is _not_ dominant, so the original question dissolved and this campaign
   replaced it.

Re-verify the table (read-only, safe):

```bash
cat results/fusion_sweep_1200q.md results/fusion_sweep_v1.md
grep -n "RRF_K = " src/scix/search.py
wc -l eval/recall_gold_v1.jsonl eval/retrieval_50q.jsonl   # 1200 / 50
```

## 2. The campaign: does the INDUS dense lane earn its place, and does hybrid survive a skeptic?

**PROVISIONAL pending Stephanie (Q1):** this is the designated hardest-problem
campaign. The s7cy embed fire and the disk/RAM window are live _operational_
incidents, handled in `scix-embedding-pipeline` / `scix-index-and-storage-discipline`
/ `scix-change-control`, not here.

Harness: `scripts/fusion_sweep.py` + `eval/recall_gold_v1.jsonl`. The script is
**READ-ONLY against the corpus** (its own docstring: "it only issues SELECT/kNN
queries") but it loads INDUS on GPU and hits prod PG + Qdrant, so every live run is
heavy work: **`scix-batch` wrapper mandatory, do not run casually, coordinate an
operator window** (see `scix-memory-and-batch-discipline`). The reference 1200q run
took ~27 min under `scix-batch` (log timestamps 12:09→12:36, 2026-06-20).

### Phase 0 — Preconditions and confounds (read-only; run these checks, nothing else)

```bash
# 0a. Instrument present? (it is gitignored + untracked — a fresh clone LACKS it)
wc -l eval/recall_gold_v1.jsonl          # expect 1200
git check-ignore eval/recall_gold_v1.jsonl && echo "not in a clone — see Phase 4 repro gap"

# 0b. Harness present and unchanged?
git log --oneline -3 -- scripts/fusion_sweep.py
python -c "import ast; ast.parse(open('scripts/fusion_sweep.py').read()); print('parses')"

# 0c. Dense lane reachable? QDRANT_URL must be set for the dense lane to serve at all
#     (unset => hybrid degrades; see scix-vector-serving-qdrant). Do NOT hardcode a port.
echo "${QDRANT_URL:-UNSET}"
```

**Confound you must record before any run — PROVISIONAL pending Stephanie (Q2):**
committed HEAD's embed pipeline writes to `paper_embeddings`, a table dropped
out-of-process ~2026-06-29/30 (bead `s7cy`, OPEN). Verify the committed reality
yourself:

```bash
git show HEAD:src/scix/embed.py | grep -c "paper_embeddings"    # >0: writes to a dropped table
git show HEAD:scripts/daily_sync.sh | grep -n "set -euo"        # Step 5 failure aborts Steps 6-7
```

Consequence: new papers since ~2026-06-29 have **no dense vector** (~83K reported in
bead/audit prose — unverified here, needs a DB/Qdrant count you must not run
casually). The dense lane's effective recall decays over time while BM25's does not,
so **any dense-vs-BM25 comparison drifts anti-dense until ingest is fixed**. A
direct-to-Qdrant remediation exists but is **uncommitted, in-flight — teach and cite
committed reality; canonize nothing** until it lands through change control.

### Phase 1 — Reproduce the baseline (heavy; scix-batch; operator window)

```bash
# DO NOT RUN CASUALLY — GPU model load + prod PG/Qdrant reads, ~30 min.
scix-batch .venv/bin/python scripts/fusion_sweep.py \
  --queries eval/recall_gold_v1.jsonl \
  --output results/fusion_sweep_1200q_rerun_$(date +%Y%m%d).md \
  --json-output results/fusion_sweep_1200q_rerun_$(date +%Y%m%d).json
```

(`--dry-run` validates wiring with no DB and no model — the docstring-documented
safe smoke; use it first in any new environment.)

**Expected observations and branches:**

| Observation                                                                          | Meaning                                            | Branch                                                                                        |
| ------------------------------------------------------------------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| dense_only ≈ 0.380, bm25_only ≈ 0.429, naive_rrf(k=60) ≈ 0.479 (± a few thousandths) | Baseline reproduced                                | Proceed to Phase 2                                                                            |
| dense_only materially below 0.38, BM25 stable                                        | The un-vectored-papers gap grew (Phase 0 confound) | STOP interpreting; run the Phase 2c coverage audit first; report the gap size with the number |
| dense lane returns 0 rows / errors                                                   | `QDRANT_URL` unset or Qdrant down                  | `scix-vector-serving-qdrant`                                                                  |
| Everything shifted, both lanes                                                       | Corpus drift or gold-set edit                      | Diff the gold file; `git log` the harness; re-pin before comparing                            |

### Phase 2 — Skeptic audit of the instrument (the first three concrete steps in this repo)

The 1200q gold is citation-graph-constructed: per line, `gold = self + strongest
citation-graph neighbour(s)`, one bucket (`recall_decile`), deciles 0–9 × 120. A
skeptic's attack: _title-like queries whose gold is the paper itself are a BM25 home
game_ (line 1 of the file is literally "SGLF/ZTF Transient Discovery Report for
2021-09-07" with itself as gold). These three steps test that attack.

**2a. Per-decile breakdown (cheap, do first).** The sweep buckets by the gold file's
`bucket` field (`scripts/fusion_sweep.py` ~line 296–302), and all 1200 rows share one
bucket, so the shipped per-bucket table is degenerate (one row). Write a re-bucketed
copy and re-run:

```bash
python - <<'EOF'
import json
with open('eval/recall_gold_v1.jsonl') as f, \
     open('eval/recall_gold_v1_by_decile.jsonl', 'w') as out:
    for line in f:
        o = json.loads(line)
        o['bucket'] = f"decile_{o['decile']}"
        out.write(json.dumps(o) + "\n")
EOF
# then the Phase-1 command with --queries eval/recall_gold_v1_by_decile.jsonl  (heavy)
```

Expected: a 10-row per-bucket table. **Gate:** if the RRF lift concentrates in a few
deciles (or the dense deficit does), the headline must be restated per-decile — an
aggregate +0.098 hiding a bimodal story does not survive review.

**2b. Query-type audit (needs judgment, not keywords).** Classify the 1200 queries
(title-like / concept / method / author-ish). Per the ZFC rule, semantic
classification is model work — run a judge pass (see `scix-eval-and-evidence` for the
persona/judge harness), not a regex. Expected: a title-like share; **gate:** report
dense-vs-BM25 per class. If BM25's win lives in title-like self-retrieval and dense
wins on concept queries, the "weak dense lane" claim gets a precise shape — and the
dfba dense-sensitive result stops looking like an outlier.

**2c. Vector-coverage audit (live reads; do not run casually).** For every gold
bibcode, is a vector actually present in Qdrant collection `scix_indus_v2_papers_s1`?
Requires live Qdrant retrieval + a PG count — route through `scix-db-safety-and-telemetry`
(prod DSN discipline) and run under `scix-batch`. Expected: a coverage %. **Gate:** any
dense number published without this % alongside it is inadmissible while `s7cy` is open.

### Phase 3 — The dense-lane frontier (solution menu, ranked; each with obligations)

Ranked by information-per-cost. Every item is **open/candidate — nothing here is
adopted**.

1. **Separate ANN loss from model weakness (cheapest, most discriminating).**
   `SCIX_QDRANT_EXACT=1` forces exact (non-indexed) kNN — an eval-only control built
   into `vector_search` (`src/scix/search.py`, HEAD ~line 552). Re-run Phase 1 on a
   subsample with it set. Expected: if exact kNN recovers a meaningful fraction of the
   dense deficit → serving/ANN recall problem (Qdrant index params territory,
   `scix-vector-serving-qdrant`); if numbers barely move → the model itself is the
   ceiling. Obligation: exact kNN is drastically slower — subsample first, `scix-batch`
   always.
2. **Query–document asymmetry diagnostic.** INDUS embeds title+abstract documents;
   gold queries are short strings. The sweep JSON stores only per-config aggregates
   (verify: `python -c "import json; print(list(json.load(open('results/fusion_sweep_1200q.json'))['results'].keys()))"`),
   so a per-query-length analysis needs a small harness extension that persists
   per-query rows. That is eval-code change: tests ship with it, normal review, no ADR
   needed (it changes no serving behavior).
3. **A stronger local dense lane (expensive; heavily gated).** Candidates with their
   existing in-repo evidence:
   - _nomic-embed-text-v1.5_: beat INDUS 0.459 vs 0.427 nDCG@10 on the 50q/10K-sample
     pilot (`docs/paper_outline.md` §4.4; nomic-vs-INDUS gap not significance-tested);
     full-corpus embed cost/storage was the stated blocker.
   - _Contrastive fine-tune_: `scripts/train_body_abstract_contrastive.py` +
     `results/body_contrastive_pilot.md` exist as a pilot line.
   - _Post-fusion reranker_: already benchmarked — `results/indus_ranker_benchmark_m2.json`
     shows +0.0055 nDCG@10 over BM25 on nasa-smd-IR (marginal);
     `results/retrieval_eval_50q_rerank_*.md` cover local rerankers.

   Obligations for ANY candidate lane — treat all as HALT-branch-ready,
   **PROVISIONAL pending Stephanie (Q5)**: local-weight only (paid-API embedding
   lanes are banned outright — CLAUDE.md + `feedback_no_paid_apis`); binary
   quantization banned (>40% nDCG@10 loss); a full-corpus embed is a multi-day,
   disk/RAM-gated operation currently parked behind the operator maintenance window;
   new lane = ADR through `scix-change-control`; and the ADR-013 index rules apply
   (≤50k scratch build + forced-index-scan smoke before any long build; benchmark DDL
   byte-identical to prod).

**You have a result when:** on the 1200q set (or a successor instrument that ships
with the repo), a candidate dense lane's `dense_only` nDCG@10 ≥ `bm25_only`'s, with
per-decile/per-class breakdown, vector-coverage % reported, and the fusion lift
re-measured on top — or when Phase 2 demonstrates, with per-class numbers, that the
instrument (not the lane) explains the deficit. Either outcome is publishable; only
the unexamined aggregate is not.

## 3. Frontier 2 — claim_blame has no real-corpus accuracy number (bead `6ajy`, OPEN)

The provenance tools (`claim_blame`, `find_replications`) are the ADASS
differentiator, and `claim_blame`'s test coverage is entirely synthetic-fixture shape
tests — **there is no real-corpus accuracy number; the PRD MH-1/MH-8b recall gate was
never run** (bead `6ajy`, the 2026-06-22 audit's smartest-addition, still OPEN,
`gc.no_route: true` — it needs domain judgment, operator or judge pass; it is not
auto-slung).

The specified plan (from the bead — quote it, don't improvise):

- Build 30–50 known-origin cases (Riess+2011 H0, Perlmutter+1999 Λ, BICEP2 dust, …)
  → `tests/eval/claim_blame_gold_v1.jsonl` (claim_text + verified origin bibcode +
  acceptable-origin set).
- Run claim_blame over them; emit **origin-recall@1/@5, retraction-overlay precision,
  lineage-chronology, and COVERAGE%** — `v_claim_edges` is ~821K of 299.3M edges
  (bead prose; unverifiable without a DB read).
- Output `results/claim_blame_gold_v1.md` in the `fusion_sweep_1200q.md` shape.
- **Decision gates (pre-registered in the bead):** ≥70% origin-recall → adopt the hard
  intent filter; <40% → keep current weights.

Confound to report with any number: `v_claim_edges` is refreshed at `daily_sync.sh`
Step 6, downstream of the broken Step 5 embed (`set -euo pipefail` at HEAD aborts the
script), so the MV is stale since ~2026-06-29 while `s7cy` is open —
**PROVISIONAL pending Stephanie (Q2)**. A coverage % measured on a stale MV must say so.

**You have a result when:** origin-recall@1 exists for ≥30 real-corpus cases with the
coverage % printed beside it. Note the existing `eval/claim_extraction_gold_standard.jsonl`
(15 lines, `GOLD`-prefixed placeholder bibcodes) is the claim-_extraction_ shape set —
it cannot answer this question; don't repurpose it.

## 4. External positioning — the ADASS thesis and what may be claimed

**The thesis** (`docs/paper_outline.md`): expose the _structural topology of science_
— citation-graph analytics, communities, the entity graph, provenance — rather than
ranked lists; corpus completeness is load-bearing (6-year window resolves 17.8% of
citation edges; full corpus 99.6%). The differentiators vs Elicit/SciSpace/paper-qa
are the graph/provenance tools, **not** dense-retrieval supremacy — which is exactly
why the honest retrieval headline (§1) is survivable.

**The outline is stale — do not copy claims from it.** Its abstract still says a
"13-tool MCP server" and RRF fusion with `text-embedding-3-large`. Committed reality:
the tool surface is 15 (triple-enforced cap, see `scix-mcp-tool-surface`) and the
OpenAI lane was removed (`git log --format="%h %s" -1 8b9cc90` → "remove dead OpenAI
dense lane in hybrid_search", bead `7gb4`; `grep -c "text-embedding" src/scix/search.py`
→ 0). Revising the outline is real work under an explicit bead — never silently.

**Claims discipline** (each row is binding until the evidence changes):

| Claim                                                    | Status                                                              | What it needs before ADASS                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------- |
| Full-corpus graph resolution 99.6% vs 17.8% windowed     | Shipped in outline; verify source table before quoting              | Cite the generating artifact                                |
| Dense nDCG@10 0.864 vs BM25 0.088                        | Real but **instrument-specific** (27q dense-sensitive; bead `dfba`) | Always name the instrument; never present as corpus-general |
| Hybrid +0.098 nDCG@10 over dense-alone (1200q)           | Real, **with the weak-dense caveat verbatim** (§1, PROVISIONAL Q3)  | Per-decile breakdown (Phase 2a) + coverage % (Phase 2c)     |
| "Hybrid beats BM25 by +0.0495 (0.4786 vs 0.4291)"        | True on 1200q; weaker framing but skeptic-proof                     | Same as above                                               |
| INDUS earns its place as the dense lane                  | **OPEN — the campaign question**                                    | Phase 2/3 outcomes                                          |
| claim_blame provenance accuracy                          | **NO NUMBER EXISTS** (§3)                                           | `6ajy` gold set run                                         |
| 49–67% hybrid error reduction "expected from literature" | Outline hedge, **not our result**                                   | Delete or measure; never state as ours                      |

**Reproducibility standards** (what "someone else can check it" means here):

1. **The instrument ships or the number doesn't.** `eval/recall_gold_v1.jsonl` is
   untracked + gitignored (`git check-ignore eval/recall_gold_v1.jsonl` → matches
   `*.jsonl`) — a clean clone cannot reproduce the headline. Before publication:
   either commit it (licensing check first — bibcodes + short queries), or commit its
   generator with a pinned seed (it originated on branch `bd/h2sj-recall-gold-v1`).
   `eval/retrieval_50q.jsonl` IS tracked; the asymmetry is the gap.
2. **Pin models and revisions** the way `results/indus_ranker_benchmark_m2.json` does
   (`benchmark_revision` sha, model id, k, n_queries).
3. **Null results stated plainly.** The house style is the sweep verdict: "Premise
   does NOT reproduce on this gold set." Copy that register.
4. **Every number carries: instrument, n, per-bucket table, lane provenance, date,
   and (while `s7cy` is open) vector-coverage %.**

## 5. Wrong paths, fenced off

| Do not                                                                     | Why                                                                                                                                      |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Retune RRF k for quality                                                   | k=10..100 spans ≤0.002 nDCG@10 on 1200q; prod already runs the winner (k=60)                                                             |
| Quote `dfba` 0.864 as the system's retrieval quality                       | 27q dense-sensitive instrument; §1                                                                                                       |
| Draw fusion conclusions from the 50q set                                   | Its own verdict: BM25-anchored, findability gaps, "diagnostic only"                                                                      |
| Add a paid-API embedding lane (e.g. text-embedding-3-large) "to fix dense" | Banned (CLAUDE.md, `feedback_no_paid_apis`); the removed lane is dead, not parked                                                        |
| Use binary quantization to make a bigger model fit                         | >40% nDCG@10 loss; ADR-pinned ban                                                                                                        |
| Run any live sweep/eval outside `scix-batch`                               | oomd kills the co-hosted supervisor; the reference run itself ran in a scix-batch scope (first line of `results/fusion_sweep_1200q.log`) |
| Treat the uncommitted direct-to-Qdrant embed fix as the standard path      | In-flight, `s7cy` OPEN — **PROVISIONAL pending Stephanie (Q2)**                                                                          |
| Auto-sling the `6ajy` gold-set build                                       | `gc.no_route: true`; hand-labeling needs domain judgment                                                                                 |
| Publish a dense number without the un-vectored-papers coverage %           | The s7cy gap grows daily and biases anti-dense                                                                                           |

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`, HEAD
`452ab86` (not `main`). Working tree carried the uncommitted s7cy remediation; every
code claim above was checked against `git show HEAD:<file>`, every number against
`results/` artifacts or bead prose (`bd show isve 9sv1 dfba 6ajy`). Unverified-here
(need DB/Qdrant access this skill must not exercise casually): the ~83K un-vectored
count, the ~821K `v_claim_edges` count, the dfba full-run artifact location.

One-line re-verification (all read-only, safe):

```bash
git branch --show-current && git rev-parse --short HEAD
cat results/fusion_sweep_1200q.md            # the headline table + verdict
grep -n "RRF_K = " src/scix/search.py        # production fusion constant
wc -l eval/recall_gold_v1.jsonl              # 1200 (instrument present?)
git check-ignore eval/recall_gold_v1.jsonl   # still the repro gap?
bd show s7cy 6ajy                            # are the two live fires still open?
git show HEAD:src/scix/embed.py | grep -c paper_embeddings   # committed embed still broken?
grep -n "13-tool\|text-embedding-3-large" docs/paper_outline.md | head -3  # outline still stale?
```

If `s7cy` closes, `6ajy` closes, or a fusion/dense re-run lands in `results/`, the
affected section above is stale — update it in the same change.
