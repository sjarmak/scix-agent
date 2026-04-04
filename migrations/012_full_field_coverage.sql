-- 012_full_field_coverage.sql
-- Add dedicated columns for all ADS API fields harvested by scripts/harvest_full.py.
-- Previously these fields were either dropped or buried in the raw JSONB blob.
-- Promotes them to first-class columns for indexing, filtering, and agent queries.
--
-- All columns are nullable — most papers won't have all fields populated.
-- ALTER TABLE ... ADD COLUMN IF NOT EXISTS is catalog-only for nullable columns
-- (no table rewrite), so this migration is safe on large tables.

BEGIN;

-- Text fields
ALTER TABLE papers ADD COLUMN IF NOT EXISTS ack TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS date TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS eid TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS entdate TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS first_author_norm TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS page_range TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS pubnote TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS series TEXT;

-- Array fields (TEXT[])
ALTER TABLE papers ADD COLUMN IF NOT EXISTS aff_id TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS alternate_title TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS author_norm TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS caption TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comment TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS data TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS esources TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS facility TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS grant_facet TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS grant_agencies TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS grant_id TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS isbn TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS issn TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keyword_norm TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keyword_schema TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS links_data TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS nedid TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS nedtype TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS orcid_other TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS simbid TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS vizier TEXT[];

-- Integer fields
ALTER TABLE papers ADD COLUMN IF NOT EXISTS author_count INTEGER;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS page_count INTEGER;

-- Real (float) fields — normalized scores from ADS
ALTER TABLE papers ADD COLUMN IF NOT EXISTS citation_count_norm REAL;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS cite_read_boost REAL;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS classic_factor REAL;

-- Indexes for high-value queryable fields
-- GIN indexes on arrays frequently used for faceted search
CREATE INDEX IF NOT EXISTS idx_papers_data ON papers USING GIN (data);
CREATE INDEX IF NOT EXISTS idx_papers_facility ON papers USING GIN (facility);
CREATE INDEX IF NOT EXISTS idx_papers_esources ON papers USING GIN (esources);
CREATE INDEX IF NOT EXISTS idx_papers_nedid ON papers USING GIN (nedid);
CREATE INDEX IF NOT EXISTS idx_papers_simbid ON papers USING GIN (simbid);
CREATE INDEX IF NOT EXISTS idx_papers_keyword_norm ON papers USING GIN (keyword_norm);
CREATE INDEX IF NOT EXISTS idx_papers_author_norm ON papers USING GIN (author_norm);

-- B-tree index on author_count for range queries
CREATE INDEX IF NOT EXISTS idx_papers_author_count ON papers (author_count);

COMMIT;
