-- 018_openalex_linking.sql
-- Add OpenAlex work ID and topic annotations to papers table.

BEGIN;

ALTER TABLE papers ADD COLUMN IF NOT EXISTS openalex_id TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS openalex_topics JSONB;

CREATE INDEX IF NOT EXISTS idx_papers_openalex_id ON papers(openalex_id);

COMMIT;
