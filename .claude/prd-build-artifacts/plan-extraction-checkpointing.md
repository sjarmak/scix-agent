# Plan: Extraction Checkpointing

## Step 1: Add checkpoint helper functions to extract.py

- `_checkpoint_path(output_dir, version)` -> Path to `.checkpoint.json`
- `_load_checkpoint(output_dir, version)` -> dict with processed_bibcodes set + cumulative_cost_usd
- `_save_checkpoint(output_dir, version, processed_bibcodes, cumulative_cost_usd)` -> writes JSON

## Step 2: Add DB idempotency check function

- `_get_existing_bibcodes(conn, version)` -> set of bibcodes already in extractions table for given version
- Simple SELECT DISTINCT bibcode FROM extractions WHERE extraction_version = %s

## Step 3: Modify run_extraction_pipeline()

- After selecting cohort, load checkpoint + query DB for existing bibcodes
- Filter cohort: remove bibcodes in checkpoint OR in DB
- In batch loop: after each batch completes and loads to DB:
  - Add batch bibcodes to processed set
  - Compute batch cost estimate and add to cumulative_cost_usd
  - Save checkpoint
  - Check if cumulative_cost_usd >= 0.8 \* budget_usd -> halt with 'budget' message

## Step 4: Fix pre-existing test failures

- Fix `test_user_message_at_end`: check inside list content
- Fix `test_submits_batch_and_returns_id`: unpack tuple return

## Step 5: Add new tests

- `test_checkpoint_save_and_load`: write then read checkpoint
- `test_checkpoint_skips_processed_bibcodes`: pipeline skips bibcodes in checkpoint
- `test_db_idempotency_skips_existing`: pipeline skips bibcodes already in DB
- `test_budget_halt_at_80_percent`: cumulative cost triggers halt with 'budget' message
- `test_duplicate_submission_no_duplicates`: submitting same cohort twice doesn't duplicate

## Step 6: Run all tests, fix any failures
