-- Migration 017: Add provenance metadata to extractions
--
-- Tracks the origin and confidence of each extraction record:
--   source           — where the extraction came from (llm, metadata, ner, etc.)
--   confidence_tier  — high / medium / low
--   extraction_model — which model produced it (nullable for non-LLM sources)

BEGIN;

ALTER TABLE extractions
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'llm',
    ADD COLUMN IF NOT EXISTS confidence_tier TEXT NOT NULL DEFAULT 'medium',
    ADD COLUMN IF NOT EXISTS extraction_model TEXT;

-- Constrain to known values
ALTER TABLE extractions
    ADD CONSTRAINT chk_extractions_source
        CHECK (source IN ('metadata', 'ner', 'llm', 'openalex', 'citation_propagation'));

ALTER TABLE extractions
    ADD CONSTRAINT chk_extractions_confidence_tier
        CHECK (confidence_tier IN ('high', 'medium', 'low'));

-- Index for filtering by provenance
CREATE INDEX IF NOT EXISTS idx_extractions_source ON extractions(source);
CREATE INDEX IF NOT EXISTS idx_extractions_confidence_tier ON extractions(confidence_tier);

COMMIT;
