# Plan: citation-context-pipeline

## Implementation Steps

### 1. Data Structures (frozen dataclasses)

- `CitationMarker`: marker_text, marker_numbers (list[int]), char_start, char_end
- `CitationContext`: source_bibcode, target_bibcode, context_text, char_offset, section_name (optional), intent (None for now)

### 2. extract_citation_contexts(body: str) -> list[CitationMarker]

- Regex patterns for:
  - Single: `\[(\d+)\]`
  - Comma-separated: `\[(\d+(?:\s*,\s*\d+)+)\]`
  - Range: `\[(\d+)\s*-\s*(\d+)\]`
- Combined into one pattern or processed sequentially
- For each match, capture ~125 words before and ~125 words after
- Return list of CitationMarker with context window text and char offsets

### 3. resolve_citation_markers(markers: list[CitationMarker], references: list[str]) -> list[CitationContext]

- For each marker, for each number N in marker_numbers:
  - If 1 <= N <= len(references): map to references[N-1]
  - Else: skip (out of bounds)
- Return CitationContext objects (one per resolved target bibcode)

### 4. process_paper(bibcode, body, references, sections=None) -> list[CitationContext]

- Call parse_sections(body) to get section info
- Call extract_citation_contexts(body) to find markers
- Enrich each marker with section name based on char offset
- Call resolve_citation_markers to get final contexts
- Set source_bibcode on all results

### 5. run_pipeline(dsn, batch_size=1000, limit=None)

- Connect to DB
- SELECT bibcode, body, raw FROM papers WHERE body IS NOT NULL
- Filter for papers where raw contains 'reference' key
- Process in batches
- COPY results into citation_contexts table
- Log progress

### 6. Tests

- Single [1] marker extraction
- Multiple [1,2,3] markers
- Range [1-3] markers
- Author-year style graceful skip (no matches)
- Edge cases: marker at start/end, N > len(references)
- process_paper integration
- Section enrichment
