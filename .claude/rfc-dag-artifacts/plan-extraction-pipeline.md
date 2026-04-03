# Plan: Extraction Pipeline

## Step 1: Create src/scix/extract.py

### Data Structures

- `ExtractionRequest` frozen dataclass: bibcode, title, abstract
- `ExtractionResult` frozen dataclass: bibcode, extraction_type, extraction_version, payload

### Functions

1. **select_pilot_cohort(conn, limit=10000) -> list[ExtractionRequest]**
   - Query papers ORDER BY citation_count DESC WHERE length(abstract) > 100
   - Return list of ExtractionRequest

2. **build_extraction_prompt(title, abstract) -> dict**
   - System prompt explaining entity extraction task
   - 3+ few-shot examples across astronomy subfields (solar physics, cosmology, exoplanets)
   - Tool-use schema with tools for methods/datasets/instruments/materials
   - Returns dict suitable for Anthropic Messages API

3. **submit_batch(client, requests, model) -> str**
   - Build batch request items from ExtractionRequests
   - Call client.messages.batches.create()
   - Return batch_id

4. **poll_batch(client, batch_id, interval=60) -> BatchResult**
   - Poll client.messages.batches.retrieve() until terminal state
   - Return batch object

5. **save_results_jsonl(client, batch_id, output_path) -> Path**
   - Stream results from batch API
   - Write each result as a JSONL line to output_path
   - Return the path

6. **load_results_to_db(conn, jsonl_path, extraction_version, chunk_size=500) -> int**
   - Read JSONL file line by line
   - Parse tool_use results into extraction rows
   - Chunk into groups of 500
   - INSERT ... ON CONFLICT (bibcode, extraction_type, extraction_version) DO UPDATE
   - COMMIT between chunks
   - Return total rows written

## Step 2: Create scripts/extract.py CLI

- Arguments: --pilot-size, --batch-size, --output-dir, --dsn, --model, -v
- Orchestrate: select cohort -> submit batch -> poll -> save JSONL -> load to DB

## Step 3: Create tests/test_extract.py

- TestSelectPilotCohort: SQL construction, ordering, abstract filter
- TestBuildExtractionPrompt: system prompt present, few-shot examples, tool schema, all 4 entity types
- TestSaveResultsJsonl: writes valid JSONL, checkpoint before DB
- TestLoadResultsToDb: chunked writes, idempotent upserts, commit between chunks
- TestSubmitBatch: request construction
- At least 10 tests total
