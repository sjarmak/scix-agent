# Qdrant `set_payload` write-path — root cause + fix (bead `scix_experiments-tqg4`, 2026-06-15)

**Status: BRANCH-READY — root cause confirmed, fix specified + scripted, throughput
quantified from real-corpus data. Prod apply is operator-gated (manual-sjarmak);
surfaced below, NOT executed.**

Stephanie 2026-06-15 (Slack): "fix the write paths."

## TL;DR

Per-point `set_payload` is the wrong primitive for `scix_indus_v2_papers_s1`. On
this collection a single `set_payload` op costs **~47–63 s to apply** — proven
intrinsic, not contention (see §2). With `wait=False` the client acks ops far
faster than the ~0.02 ops/s apply rate can drain, so the WAL fills with unapplied
ops (currently **149,296 queued, static**), only ~8,322 of 32.4M ever landed, and
every container restart replays that WAL serially → the ~35-min recovery
poisoning.

**Fix: never backfill payload via per-point `set_payload`. Carry the full payload
at (re)upsert/load time.** The upsert-with-payload path is already measured at
**733 pts/s sustained over the full corpus** (12.3 h end-to-end, clean WAL) — vs a
`set_payload` apply rate that is effectively zero. **Go: re-ingest with payload.
No-go: any per-point `set_payload` backfill at scale.**

---

## 1. Confirmed collection structure (read-only, live `scix_indus_v2_papers_s1`, :6633)

`GET /collections/scix_indus_v2_papers_s1` at 2026-06-15 ~21:24 UTC (after the
container finished its ~35-min recovery this tick):

| Field | Value | Relevance |
|---|---|---|
| `points_count` | 32,383,535 | full corpus |
| `segments_count` | **25** | → **~1.3 M points/segment** (few huge segments) |
| `optimizer_config.max_segment_size` | **null** | nothing caps segment growth |
| `optimizer_config.default_segment_number` | 8 | matches `qdrant_full_load.py` |
| `params.on_disk_payload` | **true** | payload lives on disk (mmap), not RAM |
| `params.vectors.on_disk` | true / float16 | byte-identical to loader |
| `quantization.scalar.always_ram` | false | quantized layer mmap'd (RAM-walled host) |
| payload_schema | 7 indexed fields present, **~8,322 points each** | the backfill applied to only ~8 k points before stalling |
| `update_queue.length` | **149,296 (static over 4 s)** | acked-but-unapplied `set_payload` ops piling in the WAL |
| `status` / `optimizer_status` | green / ok | **misleading** — both report healthy while writes silently drop (consistent with the 06-12 / 06-15 pilots) |

Config is byte-identical to `scripts/qdrant_full_load.py` (ADR-013 validity anchor):
`VectorParams(768, COSINE, FLOAT16, on_disk=True)`, `HnswConfigDiff(m=32,
on_disk=False)`, `ScalarQuantization(INT8, 0.99, always_ram=False)`,
`OptimizersConfigDiff(default_segment_number=8)`.

## 2. Root cause — the per-op cost is **intrinsic**, not external contention

The decisive new evidence this bead adds over the prior pilots is the **WAL
recovery log** from the container boot this tick (`docker logs scix-qdrant-v2`):

```
WARN ... Slow WAL operation during recovery: set_payload took 47.43s ... op_num: 24174
WARN ... Slow WAL operation during recovery: set_payload took 48.25s ... op_num: 24175
... (every replayed set_payload 46.7–63.0 s) ...  41-segment shard replay, ~35 min
```

WAL recovery is **single-threaded with no concurrent search/scan traffic** — yet
each `set_payload` still takes ~47–63 s to apply. That rules out the prior
hypothesis (an external `--live` recall canary holding segment read-locks and
starving appliers, per `payload_backfill_pilot_2026-06-15.md` / bead nnim) as the
*primary* cause. An external scan can make it **worse**, but the floor cost is
structural:

> A single `set_payload` by point-id must locate the point inside a ~1.3 M-point,
> `on_disk` segment and update both the on-disk payload storage **and** the 7
> on-disk payload indexes for that segment. With `max_segment_size=null` the
> segments are enormous, so each in-place payload mutation does work proportional
> to segment structure, not to the one point changed. At ~47–63 s/op the apply
> rate is **~0.02 ops/s** — hopelessly below any backfill need.

`wait=False` then converts a slow-apply problem into a **silent-data-loss +
recovery-poisoning** problem: the submission path acks in ~3 ms (200 OK), the
WAL/update_queue grows without bound (→ 149,296), and on restart the whole queue
replays at 47–63 s/op.

**This reconciles the two prior pilots:** they correctly observed "acks every
write, applies none" and a stalled optimizer, and attributed it to a hung scan.
The recovery replay shows the apply primitive is itself the wall — so "restart +
stop the scan" (their recommended action) treats a symptom; the write **path**
must change.

## 3. Throughput — before/after (real-corpus measured data, committed)

No new heavy run was performed (see §5 for why a ≤250 k scratch can't reproduce
the pathology and why the host can't safely host one right now). The
authoritative numbers already exist in `results/qdrant_backfill/`:

| Write path | Submission rate | **Apply behavior** | Full-corpus (32.38 M) | Restart impact |
|---|---|---|---|---|
| per-point `set_payload`, `wait=False` (current `backfill_qdrant_filter_fields.py`) | 166 rows/s @ 5 ms interval (`pilot_2026-06-12`) | **stalls — 8,322 landed, 149,296 queued; replay 47–63 s/op** | never completes (apply ≈ 0.02 ops/s) | **poisons restart: ~35-min serial WAL replay** |
| **upsert carrying payload** (`qdrant_full_load.py` pattern) | **733 pts/s sustained**, 993–1,320 pts/s early (`backfill_20260427_125951.log`, full 32.38 M load) | normal point writes; applies | **~12.3 h end-to-end** | clean (point-write replay is fast — this is how the collection was originally loaded) |

The fix meets the bead's **≥1 k pts/s** target in the early window and sustains
**733 pts/s** across the full corpus — versus a `set_payload` apply rate that is
effectively zero and actively harmful.

**Caveat on the 733 pts/s number:** the 2026-04-27 run read vectors from
`paper_embeddings` (PG binary fetch was the bottleneck then). That table is now
dropped (§4); the fix reads vectors via Qdrant `scroll(with_vectors=True)`
instead, whose throughput is not yet independently measured at scale. The
*write* (upsert) side is unchanged and is the path being fixed; 733 pts/s is a
reasonable planning figure but the scroll-bound end-to-end rate should be
confirmed by the pilot (`qdrant_reload_with_payload.py --limit 100000` under
scix-batch in a RAM-headroom window). `qdrant_writepath_bench.py` isolates and
measures the pure write side (synthetic vectors) independently.

## 4. The fix (branch-ready; operator-gated to apply)

**Eliminate per-point `set_payload`. Carry payload in the point itself via
upsert into a fresh collection.**

**Constraint discovered while building the fix:** `paper_embeddings` was **dropped
2026-06-14** (the Qdrant migration; confirmed — only `section_embeddings` remains
in PG). So `qdrant_full_load.py`'s "read vectors from PG" path is dead — the INDUS
vectors now live *only* inside `scix_indus_v2_papers_s1`. The fix must therefore
**scroll vectors out of the existing collection (read-only) and enrich each point
with payload looked up from PG `papers` + `paper_metrics` by bibcode**, upserting
into a new collection. Indexes built after load.

Script: **`scripts/qdrant_reload_with_payload.py`** (new, this branch). Validated
end-to-end in `--dry-run` (read-only): scrolls `(id, vector, bibcode)` pages from
the source, enriches via PG (the *already-unit-tested* `_build_payload()` from
`backfill_qdrant_filter_fields.py`), parallel-gRPC upserts `(id, vector, full
payload)` into `--target-collection` (defaults to a NON-prod `*_payload` name),
then creates the 7 ADR-008 payload indexes. Disk-floor guarded; source never
mutated. Then swap the alias.

This also clears the **149,296-op WAL debt** on the live collection (a fresh
collection has an empty WAL), removing the recurring recovery poisoning. No
in-place re-upsert fallback is offered: it would churn 32 M tombstones on the live
serving collection and is strictly worse than building fresh + swapping.

**Do NOT** lower `--call-interval-ms` or switch to `batch_update_points` as the
fix: both only reduce HTTP round-trips, and §2 shows the wall is the per-op
**apply** cost, not submission. Batching set_payload would still emit ops that
apply at ~47–63 s each and re-poison the WAL.

## 5. Why no new ≤250 k scratch run (and the ready tool for when one is safe)

- **Validity:** the pathology is segment-size-dependent. A ≤250 k byte-identical
  scratch (default_segment_number=8 → ~31 k pts/segment, ~40× smaller than prod's
  1.3 M) would make `set_payload` *fast* and **understate** the problem —
  misleading, not validating. To surface it at small scale you must deliberately
  force one large segment (`default_segment_number=1`, large `max_segment_size`),
  which is *not* byte-identical config.
- **Safety:** host is in a RAM emergency right now — `free`=434 MB, MemAvailable
  ≈2.5 G, **swap 8192/8192 exhausted**, and v2 is draining (well, *not* draining)
  a 149 k queue. Per CLAUDE.md, oomd kills at 50 % pressure and "frequently kills
  the gascity supervisor." Loading even 50–250 k points into a second Qdrant now
  risks taking down the supervisor — exactly the "do not contend" constraint.

**`scripts/qdrant_writepath_bench.py`** (new, this branch) is the controlled
before/after tool for a safe window: it targets the **isolated, empty
`scix-qdrant-gpu` instance (:6433, same image as prod)**, refuses to run if
MemAvailable is below a floor, builds a scratch collection (byte-identical config
**plus** an optional `--force-large-segment` mode to reproduce the pathology),
benchmarks per-point `set_payload` vs upsert-with-payload vs `batch_update_points`,
prints a pts/s table, and drops the scratch collection. It never touches
`scix_indus_v2_papers_s1`.

## 6. Go / no-go for prod (operator decision — surfaced, not taken)

- **NO-GO:** any per-point `set_payload` backfill of `scix_indus_v2_papers_s1` at
  any scale, with or without `wait`, with or without batching. It cannot apply at
  a viable rate and poisons restart.
- **GO (recommended):** payload-carrying re-ingest into a fresh collection +
  post-load index build + swap, via `scripts/qdrant_reload_with_payload.py`
  (scroll source → enrich from PG → upsert; clean WAL). Pilot with `--limit
  100000` first to confirm the scroll-bound rate (planning figure ~733 pts/s ⇒
  ~12 h full corpus, to be verified). Run under `scix-batch`, in a RAM-headroom
  window, against a `_payload` target, then swap the alias.
- **Separately, operator:** the live collection holds 149,296 unapplied ops that
  will re-poison the next restart. The fresh re-ingest makes this moot; until
  then, expect another ~35-min recovery on any v2 restart.

## Constraints honored

- No mutation of live `scix_indus_v2_papers_s1` (read-only GETs only).
- No heavy scratch load on the RAM-walled host (would risk the supervisor).
- Halted at branch-ready: root cause + fix script + throughput + go/no-go.
  Blocks 7ede (backfill pilot) and resolves c63d; the fix supersedes 7ede's
  "restart + stop the scan" with a write-path change.
