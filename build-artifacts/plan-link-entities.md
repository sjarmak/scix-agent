# Plan: link-entities

## Overview

Build a batch pipeline that reads extracted mentions from the extractions table, resolves them against the entities/entity_aliases tables, and writes links into document_entities.

## Key Decisions

1. **EntityResolver**: Since no EntityResolver class exists, build a simple one inline in link_entities.py that:
   - Queries entities + entity_aliases for case-insensitive match
   - Returns (entity_id, confidence, match_method) tuples
   - exact canonical match = 1.0 confidence, alias match = 0.9

2. **Payload format**: Support both formats:
   - Per-type rows: extraction_type="instruments", payload={"entities": [...]}
   - Combined row: extraction_type="entity_extraction_v3", payload={"instruments": [...], "datasets": [...], ...}

3. **Chunked commits**: Process batch_size bibcodes at a time, commit after each batch.

4. **Resume**: Skip bibcodes already present in document_entities.

## Files

- `src/scix/link_entities.py` — core module with link_entities_batch(), get_linking_progress(), EntityResolver
- `scripts/link_entities.py` — CLI entrypoint
- `tests/test_link_entities.py` — pytest suite
