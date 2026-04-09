# Test Results: Extraction Checkpointing

## Run: 35 passed, 0 failed

### Pre-existing tests (23): ALL PASS

- Fixed `test_user_message_at_end`: content is a list of blocks, not a string
- Fixed `test_submits_batch_and_returns_id`: `submit_batch()` returns `(batch_id, id_to_bibcode)` tuple

### New tests (12):

**Checkpoint tests (5):**

- `test_checkpoint_path_includes_version` — path format correct
- `test_load_checkpoint_returns_defaults_when_missing` — empty defaults when no file
- `test_save_and_load_roundtrip` — bibcodes and cost survive serialization
- `test_save_creates_parent_dirs` — nested directories created
- `test_checkpoint_overwrites_on_update` — latest state wins

**DB idempotency tests (2):**

- `test_returns_set_of_bibcodes` — queries extractions table correctly
- `test_returns_empty_set_when_no_results` — handles empty table

**Pipeline integration tests (5):**

- `test_checkpoint_skips_processed_bibcodes` — resumes from checkpoint, skips already-processed
- `test_db_idempotency_skips_existing` — skips bibcodes already in DB
- `test_duplicate_submission_no_duplicates` — second run is a no-op
- `test_budget_halt_at_80_percent` — raises ValueError with 'budget' in message
- `test_cost_accumulates_across_batches` — cumulative cost persists in checkpoint
