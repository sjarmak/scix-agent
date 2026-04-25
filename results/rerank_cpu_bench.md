# Cross-encoder rerank CPU latency benchmark

Synthetic single-query rerank, top_n=20, runs=50, warmup=3 untimed runs.
CPU forced via `CUDA_VISIBLE_DEVICES=''` and `device='cpu'` on the CrossEncoder. Run on the project RTX 5090 host with the GPU hidden, wrapped in `scix-batch` (transient systemd scope, `MemoryHigh=20G`, `ManagedOOMPreference=avoid`) to isolate from concurrent gascity / eval / embedding jobs sharing the host.

Runtime: torch 2.11.0+cu130 | cuda_available=False | threads=8

| Model | p50 (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 267.5 | 419.6 | 299.7 | 218.8 | 697.9 |
| `BAAI/bge-reranker-large` | 3707.0 | 5092.3 | 3919.5 | 3065.7 | 5573.5 |

Threshold for no-GPU default: p95 < 400 ms.

Verdict: ms-marco-MiniLM-L-12-v2 is NOT suitable as the no-GPU deployment default — measured p95 419.6 ms exceeds the 400 ms threshold.
