# Qdrant payload backfill — pilot report (bead nnim, 2026-06-12)

**Status: BLOCKED — live collection's update pipeline is stalled; pilot cannot land writes.**
Code (script + tests) is complete, reviewed, and committed. The ≤100k pilot slice and
the operator-gated full run are on hold until the stall is resolved.

## What shipped

- `scripts/backfill_qdrant_filter_fields.py` — backfills the ADR-008 payload schema
  (7 indexed fields + title/first_author/citation_count/pagerank) into
  `scix_indus_v2_papers_s1`, keyed by `uuid5(NAMESPACE_URL, bibcode)` (byte-identical
  to `qdrant_full_load.py`). Keyset-paginated on bibcode, idempotent, per-call
  throttled (`--call-interval-ms`, default 5 ms), `--limit` caps the pilot slice.
- `tests/test_backfill_qdrant_filter_fields.py` — 26 unit tests, all passing.
- **Post-run verification gate** (`verify_sample`): samples up to 10 written
  (bibcode, payload) pairs, polls `retrieve` until they appear, exits 1 if they
  don't. Added because of the incident below — without it the pilot would have
  reported "100k written" while writing nothing.

All 7 payload indexes already exist on the collection (created by the April full
load), so `ensure_indexes` is a no-op; points carry only `bibcode` today.

## Incident: update pipeline stalled on `scix_indus_v2_papers_s1`

The collection **acks every write into the WAL but never applies it**. Reads/search
serve normally, so the dense lane looks healthy while silently dropping updates.

Evidence (all 2026-06-12, container `scix-qdrant-v2` on :6633):

- 1,000-row smoke run: every `set_payload?wait=false` returned 200 OK at 166 rows/s;
  none of the sampled points carried the payload afterwards.
- `set_payload` with `wait=true` blocked the full 60 s server-side and returned
  `wait_timeout` (operation_id 173474); still unapplied 120 s later.
- Shard telemetry: optimizer `last_responded 2026-06-11T16:11:43Z` — no optimizer or
  update activity since, despite ~1,000+ acked ops. No failed/in-progress
  optimization in the log; no panic in container logs.
- Counter-evidence isolating the stall to this shard: `scix_sparse_pilot_v1` was
  created **after** the stall (2026-06-11 21:06Z) and applied all 52,443 points fine.
- Likely trigger: the exact-scan canary eval — a Search timed out server-side at
  2026-06-11T16:29Z after 60 s; the container shows 8.06 TB cumulative block reads
  and a steady ~23% CPU, consistent with a hung scan task still holding segment
  read locks and starving the shard's update appliers.

### Recommended operator action

Restart the `scix-qdrant-v2` container. WAL replay on startup should apply the
~1,000 acked-but-unapplied smoke ops (first ~1,000 bibcodes alphabetically will gain
payloads — harmless and idempotent). Then re-verify the update path with:

```bash
QDRANT_URL=http://127.0.0.1:6633 .venv/bin/python scripts/backfill_qdrant_filter_fields.py --limit 5
```

Exit 0 = pipeline healthy; the pilot can then run:

```bash
QDRANT_URL=http://127.0.0.1:6633 scix-batch .venv/bin/python scripts/backfill_qdrant_filter_fields.py --limit 100000
```

## Throughput / sizing (submission-side; application stalled, so treat as lower bound)

- Smoke measured **166 rows/s** at the default 5 ms call interval (interval-dominated).
- Full corpus (32,383,535 points) at 166 rows/s ⇒ **~54 h**.
- If 54 h is unacceptable: `batch_update_points` (one HTTP call per batch of
  `SetPayloadOperation`s, supported by qdrant-client 1.17.1) would cut HTTP round
  trips ~100×; or simply lower `--call-interval-ms` post-restart and watch memory.
  Either change should be sized against the live-serving + tight-memory constraint
  (host MemAvailable was 5.2 Gi, swap saturated during this session).

## Scope notes

- **OA flag deliberately excluded.** The bead text says "year, bibstem, OA flag,
  etc.", but ADR-008 is the canonical schema (7-index ceiling, change requires
  ADR-first) and has no OA field; `SearchFilters` has no OA filter either. Adding
  `is_oa` needs an ADR-008 amendment — flagged to PL rather than smuggled in.
- **Outbox sync will wipe backfilled payloads.** `scripts/qdrant_outbox_sync.py:302`
  upserts points with `payload={"bibcode": bibcode}`; a full-point upsert replaces
  the payload, so every re-embedded paper loses its filter fields and new papers
  never get them. ADR-008 backfill discipline step 4 ("update the upserter")
  applies — follow-up bead filed.
