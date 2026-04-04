-- Add dedicated body column for full-text paper content.
-- Previously stored inside the raw JSONB blob; now a first-class column
-- for full-text search and RAG pipelines.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS body TEXT;
