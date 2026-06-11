# ADR-014 Phase 2 (qajc) — full-corpus BM25 build HELD on disk

**Status (2026-06-11): cheap prep landed; the 32M build is HELD per the qajc
dispatch guard. Acceptance criteria 1, 2, 4, 5 are blocked on NVMe headroom.**

## Why held — disk precondition (qajc dispatch guard)

The guard requires, before any full-corpus build: estimate the throwaway
collection footprint and verify `free >= estimate + 20 GB` safety margin; if
short, do the cheap prep only and HOLD, reporting blocked-on-disk. "Do not fill
a 98% disk."

### Footprint estimate (measured from the i5oa pilot collection)

The pilot collection `scix_sparse_pilot_v1` is the ground truth for scaling —
same script, same `on_disk` sparse-BM25 config, just a bounded universe.

| quantity | pilot | full-corpus projection |
|---|---|---|
| points | 52,443 | 32.4M (`title IS NOT NULL`) |
| scale factor | 1× | **617.8×** |
| segments (steady-state index) | 37 MB | **~22.8 GB** |
| total incl. WAL | 73 MB | ~45 GB |
| **peak during bulk insert + segment merge/optimize** | — | **~40–46 GB** |

Qdrant rewrites/merges segments during a bulk load, so peak on-disk transiently
approaches ~2× the steady-state segment size before the WAL truncates and old
segments are vacuumed.

### Gate evaluation

- Free on `/dev/nvme1n1p2`: **49 GB (98% used)**.
- Guard threshold: `free >= estimate + 20 GB`.
  - Optimistic (22.8 GB steady-state): need **42.8 GB** → only 6 GB above the
    gate.
  - Realistic (40–46 GB peak): need **60–66 GB** → **FAILS by 11–17 GB**.

The host also OOM-killed postgres **twice today** (see memory
`pg_workmem_parallel_oom_2026-06-11`, `prod_disk_full_2026-06-03`). Building a
tens-of-GB throwaway index onto a 98%-full disk in that state is exactly the
failure the guard exists to prevent. **HELD.**

### Unblock condition

The `paper_embeddings → Qdrant-on-NAS` relief (beads `khug` / `dluh`, per the
guard and memory `qdrant_nas_migration`) frees the NVMe. Once free space clears
~70 GB, the full build can run:

```bash
scix-batch python scripts/qdrant_sparse_pilot.py --full-corpus \
    --collection scix_sparse_full_v1 --disable-stemmer
# then the confound-clean A/B over 32M (no --restrict-universe):
scix-batch python scripts/eval_sparse_hybrid_pilot.py --collection scix_sparse_full_v1
# DROP the throwaway collection immediately after the eval:
curl -s -X DELETE http://127.0.0.1:6633/collections/scix_sparse_full_v1
```

## What landed (cheap prep — runnable today, no disk impact)

1. **Full-corpus streaming build.** `scripts/qdrant_sparse_pilot.py` gains
   `--full-corpus` (server-side cursor over all 32.4M papers, one batch in
   memory at a time), `--collection`, and a smoke test (ADR-013 rule: one query
   must return). The bounded-universe pilot path is unchanged.
2. **OR-semantics attribution arm.** `lexical_search` gains
   `tsquery_mode={plain_and,plain_or}`; `plain_or` flips the same `scix_english`
   lexemes from `&` to `|` (verified: `'x-ray' & 'binari'` → `'x-ray' | 'binari'`).
   `eval_sparse_hybrid_pilot.py` adds the `lex_pg_or` lane and `pg_or+dense` /
   `pg_or+body+dense` arms. This splits BM25's pilot win into
   **AND→OR parsing** (`pg_or − pg_lex`) vs **BM25 scoring** (`bm25 − pg_or`).
   The eval now ranks the **full** Postgres lexical match set
   (`SCIX_LEXICAL_POOL=INF`) instead of the 30k production TID cap: the cap
   clips a larger share of the OR arm's bigger match set, so leaving it would
   make the attribution partly TID-bias. Note the i5oa artifacts were produced
   under the 30k cap, so a fresh `pg_lex` number is not directly comparable to
   them — re-run both arms together.
3. **Tokenizer tuning for `title_matchable`.** New `scripts/_sparse_bm25.py`
   centralizes the FastEmbed BM25 config so the build and eval tokenizers cannot
   drift (a mismatch would silently corrupt BM25 scores). `--disable-stemmer`
   mirrors `scix_english` `simple_nostem`; the build records its config to a
   sidecar so the eval auto-matches. Verified: nostem keeps more distinct
   scientific tokens (11 vs 9 on a sample `X-ray … Lyman-alpha … z=2.5` query).

## Acceptance-criteria status

| # | criterion | status |
|---|---|---|
| 1 | 32M collection built + smoke-tested | **BLOCKED on disk** (code + smoke test ready) |
| 2 | eval re-run over 32M, results in `results/` | **BLOCKED on disk** (harness ready) |
| 3 | `websearch_to_tsquery` OR-semantics arm | **DONE** (implemented as a clean AND→OR lexeme rewrite — see note below) |
| 4 | `title_matchable` delta after tokenizer tuning | **BLOCKED on disk** (tuning lever ready) |
| 5 | ADR-014 flipped Accepted/rejected with full-corpus numbers | **BLOCKED on disk** (stays Proposed) |

**Note on AC 3 mechanism:** the guard named `websearch_to_tsquery`, but that
function is *also* AND-by-default and would not isolate the confound. The clean
control is to rewrite `plainto_tsquery`'s output operators (`&`→`|`) over the
identical lexeme set, so the *only* thing that changes between the arms is the
boolean operator — which is exactly the attribution we want.
