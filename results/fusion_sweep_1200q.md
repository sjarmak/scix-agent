# Fusion-calibration sweep — v1

- Gold set: `eval/recall_gold_v1.jsonl` (1200 queries)
- nDCG/MRR cutoff: 10; Recall cutoffs: 10, 20, 50
- Lanes: INDUS dense (Qdrant, ADR-013) vs combined PG BM25 (title+abstract ⊕ body, RRF). READ-ONLY.

## Overall (ranked by nDCG@10)

| Fusion config | nDCG@10 | MRR@10 | Recall@10 | Recall@20 | Recall@50 | ΔnDCG vs dense_only |
|---|---|---|---|---|---|---|
| naive_rrf(k=60) | 0.4786 | 0.8926 | 0.4373 | 0.4448 | 0.4572 | +0.0983 |
| naive_rrf(k=100) | 0.4784 | 0.8924 | 0.4372 | 0.4448 | 0.4572 | +0.0981 |
| naive_rrf(k=30) | 0.4774 | 0.8915 | 0.4362 | 0.4446 | 0.4572 | +0.0971 |
| naive_rrf(k=10) | 0.4767 | 0.8908 | 0.4348 | 0.4420 | 0.4572 | +0.0963 |
| weighted_sum(w_dense=0.5) | 0.4753 | 0.8918 | 0.4321 | 0.4410 | 0.4539 | +0.0949 |
| rank_cutoff_rrf(cutoff=20) | 0.4747 | 0.8873 | 0.4341 | 0.4419 | 0.4550 | +0.0944 |
| rank_cutoff_rrf(cutoff=10) | 0.4736 | 0.8866 | 0.4338 | 0.4415 | 0.4516 | +0.0933 |
| rank_cutoff_rrf(cutoff=5) | 0.4725 | 0.8848 | 0.4330 | 0.4397 | 0.4517 | +0.0922 |
| bm25_only | 0.4291 | 0.8152 | 0.3988 | 0.4104 | 0.4258 | +0.0488 |
| weighted_sum(w_dense=0.7) | 0.4151 | 0.7896 | 0.3723 | 0.4236 | 0.4550 | +0.0348 |
| weighted_sum(w_dense=0.8) | 0.4008 | 0.7618 | 0.3573 | 0.3738 | 0.4507 | +0.0205 |
| dense_prior(lam=0.2) | 0.3970 | 0.7545 | 0.3536 | 0.3675 | 0.4413 | +0.0167 |
| weighted_sum(w_dense=0.9) | 0.3906 | 0.7410 | 0.3517 | 0.3621 | 0.3933 | +0.0103 |
| dense_prior(lam=0.1) | 0.3903 | 0.7405 | 0.3516 | 0.3620 | 0.3862 | +0.0100 |
| weighted_sum(w_dense=0.95) | 0.3868 | 0.7326 | 0.3500 | 0.3596 | 0.3804 | +0.0065 |
| dense_prior(lam=0.05) | 0.3866 | 0.7319 | 0.3498 | 0.3596 | 0.3796 | +0.0063 |
| dense_only | 0.3803 | 0.7207 | 0.3453 | 0.3565 | 0.3765 | +0.0000 |

## Verdict

**Premise does NOT reproduce on this gold set.** dense_only nDCG@10 0.3803 is *below* bm25_only 0.4291 on `eval/recall_gold_v1.jsonl` (1200 queries), so the "naive RRF hurts top-rank vs dense-alone" regime — which assumes a dominant dense lane — does not hold here. The dense lane is not dominant on this set, so any hybrid lift over dense-alone partly reflects outrunning a non-dominant dense lane rather than a clean top-rank fusion gain. Best config here is `naive_rrf(k=60)` at nDCG@10 0.4786 (+0.0983 over dense-alone); report the hybrid number with that caveat, not as a dense-dominant result.

## Per-bucket nDCG@10 (`naive_rrf(k=60)` vs dense_only)

| Bucket | dense_only | naive_rrf(k=60) |
|---|---|---|
| recall_decile | 0.3803 | 0.4786 |
