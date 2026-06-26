# External-Facing Copy Accuracy Audit — rerank-eval legitimacy scrub

**Date**: 2026-06-26
**Bead**: `scix_experiments-0yp5` (trigger: Stephanie's Slack flag, 2026-06-26,
that external copy still presents the within-paper cross-encoder rerank eval as
a legitimate result).
**Scope**: in-repo external-facing docs only. The live website source is **not**
in this repo — fixing it is gated on an operator pointer + publish approval and
is out of scope here.

## TL;DR

- The invalid within-paper rerank eval (`results/within_paper_rerank_eval.md`)
  is **not cited by any in-repo external-facing doc**. The file itself already
  carries a `SUPERSEDED / INVALID` banner pointing to the valid evals. If
  Stephanie is seeing it presented as legitimate, that copy lives on the **live
  website**, which is out of this repo's scope.
- One real in-repo finding: **`deploy/README.md` "Rerank rollout"** cited the
  superseded **April** ablation (`retrieval_eval_50q_rerank_local.md`) on a
  pgvector pool that was **dropped** in the ADR-013 Qdrant migration, and omitted
  the current authoritative June-12 re-baseline plus the domain-tuned
  `nasa-smd-ibm-ranker` result that the gate (bead `4skc`) actually hinged on.
  **Corrected on this branch.**
- All other quantitative retrieval claims found in scope trace to real
  authoritative result files. Provenance caveats are noted below; none required a
  correction.

## Authoritative reranker evals (the "valid" set, per bead 4skc)

| File | What it is | Headline |
| --- | --- | --- |
| `results/retrieval_eval_50q_rerank_indus.md` | 2026-06-12 4-way A/B on the **current** Qdrant production stack (ADR-013) | baseline `hybrid_indus` 0.2242; `minilm` +0.0489 (p=0.199, n.s.); `bge_large` +0.0197 (p=0.513, n.s.); `indus_ranker` **−0.0400 (p=0.074) → NO-GO** |
| `results/indus_ranker_benchmark_m2.json` | Home-benchmark wiring sanity on `nasa-smd-IR-benchmark` | bm25 nDCG@10 0.7535 → bm25+indus_ranker 0.7590 (model is wired correctly; it just doesn't transfer) |

The **invalid / void** one (must never be cited as evidence):
`results/within_paper_rerank_eval.md` — synthetic IMRaD fixtures + a Python
token-count stub scorer put the baseline at nDCG@3 = 1.0000 (ceiling by
construction); its −0.0185 "delta" measures nothing.

The **superseded-but-sound** one:
`results/retrieval_eval_50q_rerank_local.md` (April, commit `06a6cc3`) —
methodologically sound, but tested only web-domain MS-MARCO cross-encoders on
the legacy `paper_embeddings.embedding` pgvector pool, which was dropped in the
Qdrant migration. Its absolute numbers are no longer reproducible and it predates
the domain-tuned reranker eval; not a primary citation source anymore.

## Findings

### F1 — `deploy/README.md` "Rerank rollout" cited a superseded eval (CORRECTED)

- **Where**: `deploy/README.md:165-178` (evidence table + framing) and
  `:198-205` (latency / quality-implication bullets).
- **Claim (before)**: "the M1 ablation (commit `06a6cc3`,
  `results/retrieval_eval_50q_rerank_local.md`) showed both candidate rerankers
  regress" with the table `hybrid_indus` 0.3255 / `minilm` −0.0453 (p=0.042) /
  `bge-large` −0.0556 (p=0.026); latency "MiniLM p95 ≈ 70 ms; bge ≈ 570 ms".
- **Why wrong / stale**:
  1. Those numbers are on the **April pgvector pool that no longer exists**
     (dropped in the ADR-013 Qdrant migration), so they are not reproducible on
     the production stack.
  2. It **omits the current authoritative eval**
     (`retrieval_eval_50q_rerank_indus.md`, 2026-06-12) and the
     **domain-tuned `nasa-smd-ibm-ranker`** result — the config the gate (bead
     `4skc`) actually turned on.
  3. The "both rerankers regress" framing is pool-specific and **flips sign** on
     the current Qdrant pool (`minilm` and `bge` are nominally positive there,
     though not significant). The decisive current result is that the
     domain-tuned reranker **regresses** (−0.0400, p=0.074) → NO-GO.
  4. Latency figure "bge p95 ≈ 570 ms" was actually the April **p50** (766 ms
     p95); the current eval measured bge p95 ≈ 362 ms on GPU.
- **Correction applied**: rewrote the evidence block to cite
  `retrieval_eval_50q_rerank_indus.md` + `indus_ranker_benchmark_m2.json`,
  refreshed the 4-config table (incl. `indus_ranker` NO-GO), kept the `off`
  decision (still correct), and updated the latency/quality bullets to the
  current measurements. The `off` default and operator-flip instructions are
  unchanged — only the evidence and its framing were stale.
- **Note**: the conclusion (`SCIX_RERANK_DEFAULT_MODEL=off`) was never wrong;
  only the evidence cited for it was outdated. This was a *stale citation*, not
  an over-claimed result.

### F2 — within-paper invalid eval is NOT cited externally (NO ACTION — in-repo)

- Searched `README.md`, `AGENTS.md`, `architecture/` (`README.md`, `*.c4`),
  `deploy/README.md`, `deploy/demo/README.md`, `docs/paper_outline.md`, and all
  `docs/*.md` + `results/*.md` for `within_paper`, `within-paper`,
  `section-level rerank`, `nDCG@3`, `1.0000`, and narrative rerank claims.
- **Result**: zero references in any in-repo external-facing doc. The file itself
  (`results/within_paper_rerank_eval.md`) already opens with a
  `SUPERSEDED / INVALID` banner pointing to the valid evals.
- **Full-tree confirmation** (scope item 1's broad clause — *any* `docs/*.md` /
  `results/*.md`): a repo-wide grep for `within_paper` / `within-paper` returns
  hits only in PRDs and the tool audit, and **every one refers to the legitimate
  `search_within_paper` MCP tool or the within-paper retrieval capability — none
  cite the void eval's result.** Specifically:
  - `docs/prd/prd_full_text_applications_v2.md:138` mentions `nDCG@3` only as a
    **forward-looking acceptance target** for a *future* M5 cross-encoder rerank
    deliverable ("nDCG@3 improves by ≥5% over the BM25-only baseline"), not as an
    achieved result. This is the legitimate future version of what the void eval
    faked; it does not present the bad eval as evidence. No action.
  - `docs/mcp_tool_audit_2026-04.md`, `prd_body_chunk_embeddings.md`,
    `prd_full_text_applications.md`, the `tool-audit/*` artifacts, and
    `architecture/model.c4` reference only the `search_within_paper` tool /
    within-paper retrieval — no eval claims.
  - **No tracked doc anywhere links to `within_paper_rerank_eval.md` as
    evidence.** (`grep -rn within_paper_rerank_eval` over `docs/ README.md
    AGENTS.md architecture/ deploy/ results/*.md` returns only this audit memo.)
- **Implication**: the copy Stephanie flagged is on the **live website**, whose
  source is not in this repo. That fix is gated on an operator pointer + publish
  approval (out of scope for this bead). Flagging for the website-copy follow-up.

### F3 — `docs/paper_outline.md` retrieval numbers (VERIFIED; provenance caveat)

- The 50-query model-comparison table (`nomic` 0.459, `hybrid_indus` 0.428,
  `indus` 0.427, `specter2` 0.402, `lexical` 0.200) traces **exactly** to
  `results/retrieval_eval_50q.md` (2026-04-06). The "hybrid_indus (0.428) vs
  indus-only (0.427)… not significant, p=0.65" line matches the source
  (Δ=+0.0007, p=0.654721). Not fabricated, not over-claimed.
- **Provenance caveat (not a correction)**: this eval is on a **10K stratified
  sample**, not the full corpus, and on the 50-query curated gold set, which has
  documented findability gaps and is dense-insensitive (the hybrid-vs-dense Δ is
  within noise — consistent with the fusion sweeps and the standing note that
  this set cannot measure dense quality). The outline already presents the
  non-significance honestly. Left as-is; any rewrite of paper numbers is the
  author's call, not a copy-accuracy fix.

### F4 — `architecture/*.c4` reranker description (VERIFIED ACCURATE)

- `architecture/model.c4:146` describes the rerank stage as "bge-reranker-large +
  nasa-smd-ibm-ranker as optional rerank stage over fused candidates". Both are
  wired in `src/scix/search.py`; this is accurate and contains **no eval
  numbers**. No over-claim. No action.

### F5 — `README.md` corpus / latency claims (VERIFIED IN-FAMILY; not eval claims)

- README's quantitative claims (32.4M papers, 299M edges, 99.6% edge resolution,
  46% body coverage, HNSW p95 < 10 ms, hybrid p95 < 200 ms) are corpus-statistics
  and latency claims, not retrieval-quality eval results, and are consistent with
  the corpus/coverage figures used elsewhere in the docs. None cite or depend on
  the invalid rerank eval. No action.

## What changed on this branch

- `deploy/README.md` — "Rerank rollout" evidence block + latency/quality bullets
  updated to the current authoritative eval (F1).
- `results/external_copy_accuracy_audit_2026-06-26.md` — this audit (new).

## Out of scope / follow-ups

- **Live website copy** (the actual trigger): not in this repo. Needs an operator
  pointer to the website source + publish approval before any edit. Recommend a
  separate bead once the pointer exists.
- `docs/paper_outline.md` numbers are accurate-to-source; refreshing them against
  a full-corpus eval (if/when one exists) is an authoring decision, not a
  copy-accuracy fix.
