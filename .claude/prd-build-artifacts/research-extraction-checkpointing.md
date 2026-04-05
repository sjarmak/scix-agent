# Research: Extraction Checkpointing

## Existing Pipeline Flow

`run_extraction_pipeline()` in `src/scix/extract.py`:

1. Import anthropic, validate API key, create client + DB connection
2. `select_pilot_cohort(conn, limit=pilot_size)` — returns `list[ExtractionRequest]`
3. Pre-flight budget check: `estimate_cost()` vs `budget_usd`
4. Loop over cohort in `batch_size` chunks:
   - `submit_batch(client, batch_reqs, model)` → `(batch_id, id_to_bibcode)`
   - `poll_batch(client, batch_id, interval)` → waits for completion
   - `save_results_jsonl(client, batch_id, jsonl_file, id_to_bibcode)` → JSONL checkpoint
   - `load_results_to_db(conn, jsonl_file, extraction_version)` → upserts to DB
5. Returns total rows loaded

## Key Data Structures

- `ExtractionRequest(frozen=True)`: bibcode, title, abstract
- `ExtractionRow(frozen=True)`: bibcode, extraction_type, extraction_version, payload
- `EXTRACTION_VERSION = "v1"`

## Cost Estimation

`estimate_cost(num_requests, model)` computes USD based on `_MODEL_COSTS` dict. Currently only used for pre-flight check (total cohort). No cumulative tracking across batches.

## DB Schema

`extractions` table has `ON CONFLICT (bibcode, extraction_type, extraction_version)` upsert — already idempotent at DB level.

## Pre-existing Test Failures

1. `test_user_message_at_end`: checks `"My Title" in last_msg["content"]` but content is a list of dicts (not a string). Needs fix.
2. `test_submits_batch_and_returns_id`: `submit_batch()` now returns `(batch_id, id_to_bibcode)` tuple but test expects single string.

## Checkpoint Requirements

- File: `data/extractions/.checkpoint.json`
- Schema: `{"version": "...", "processed_bibcodes": [...], "cumulative_cost_usd": 0.0}`
- On restart: load checkpoint, skip bibcodes already in `processed_bibcodes`
- After each batch: update checkpoint with newly processed bibcodes + accumulated cost
- At 80% of budget_usd: halt with message containing 'budget'

## Idempotency Requirements

- Before submitting a batch, query DB for bibcodes that already have extractions for current version
- Filter those out of the batch request
- Combined with checkpoint file: two layers of skip logic
