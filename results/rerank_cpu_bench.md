# Cross-encoder rerank CPU latency benchmark

Synthetic single-query rerank, top_n=20, runs=50, warmup=3 untimed runs.
CPU forced via `CUDA_VISIBLE_DEVICES=''` and `device='cpu'` on the CrossEncoder. Run on the project RTX 5090 host with the GPU hidden.

Runtime: torch 2.11.0+cu130 | cuda_available=False | threads=8

| Model | p50 (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 250.5 | 988.5 | 324.8 | 213.7 | 1182.2 |
| `BAAI/bge-reranker-large` | 4160.9 | 6466.2 | 4401.9 | 3196.3 | 7429.8 |

Threshold for no-GPU default: p95 < 400 ms.

Verdict: ms-marco-MiniLM-L-12-v2 is NOT suitable as the no-GPU deployment default — measured p95 988.5 ms exceeds the 400 ms threshold.
