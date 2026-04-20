-- 003_search_infrastructure.sql
-- Search infrastructure: custom text search config for scientific text,
-- tsvector column with GIN index, and embedding metadata columns.

BEGIN;

-- ---------------------------------------------------------------------------
-- Custom text search configuration for scientific text
-- ---------------------------------------------------------------------------
-- The built-in 'english' config strips hyphens (breaking "X-ray", "gamma-ray"),
-- drops numeric tokens (losing wavelengths like "21cm", redshifts like "z=0.5"),
-- and uses generic stop words. This config handles scientific text better.

-- Simple dictionary that preserves tokens as-is (no stemming, no stop words).
-- Used for hyphenated terms and numeric tokens.
DO $$ BEGIN
    CREATE TEXT SEARCH DICTIONARY simple_nostem (
        TEMPLATE = pg_catalog.simple,
        STOPWORDS = english  -- still filter common stop words
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Scientific text search config based on english but with tweaks.
DO $$ BEGIN
    CREATE TEXT SEARCH CONFIGURATION scix_english (COPY = english);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Map hyphenated words and numeric types through the simple dictionary
-- so "X-ray" stays as "x-ray" rather than being split into "x" and "ray".
ALTER TEXT SEARCH CONFIGURATION scix_english
    ALTER MAPPING FOR hword, hword_part, hword_asciipart
    WITH simple_nostem;

ALTER TEXT SEARCH CONFIGURATION scix_english
    ALTER MAPPING FOR int, uint, float
    WITH simple_nostem;

-- ---------------------------------------------------------------------------
-- tsvector column on papers (weighted: title=A, abstract=B, keywords=C)
-- ---------------------------------------------------------------------------
-- Using our custom scix_english config instead of built-in english.
-- GENERATED ALWAYS columns cannot reference custom configs in all PG versions,
-- so we use a trigger-maintained column instead.

ALTER TABLE papers ADD COLUMN IF NOT EXISTS tsv tsvector;

-- Populate existing rows
UPDATE papers SET tsv =
    setweight(to_tsvector('scix_english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('scix_english', coalesce(abstract, '')), 'B') ||
    setweight(to_tsvector('scix_english', coalesce(array_to_string(keywords, ' '), '')), 'C')
WHERE tsv IS NULL;

-- Trigger to keep tsv updated on INSERT/UPDATE
CREATE OR REPLACE FUNCTION papers_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv :=
        setweight(to_tsvector('scix_english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('scix_english', coalesce(NEW.abstract, '')), 'B') ||
        setweight(to_tsvector('scix_english', coalesce(array_to_string(NEW.keywords, ' '), '')), 'C');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_papers_tsv ON papers;
CREATE TRIGGER trig_papers_tsv
    BEFORE INSERT OR UPDATE OF title, abstract, keywords ON papers
    FOR EACH ROW EXECUTE FUNCTION papers_tsv_trigger();

-- GIN index for fast tsvector queries
CREATE INDEX IF NOT EXISTS idx_papers_tsv ON papers USING GIN (tsv);

-- ---------------------------------------------------------------------------
-- Embedding metadata columns
-- ---------------------------------------------------------------------------
ALTER TABLE paper_embeddings ADD COLUMN IF NOT EXISTS input_type TEXT NOT NULL DEFAULT 'title_abstract';
ALTER TABLE paper_embeddings ADD COLUMN IF NOT EXISTS source_hash TEXT;

-- Partial HNSW index per model_name for blue-green transitions
-- (The existing idx_embed_hnsw covers all models; add per-model indexes as needed.)

-- ---------------------------------------------------------------------------
-- First-author normalized index for author search
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_papers_first_author ON papers (first_author);

-- Index for year range queries (already exists from 001, but ensure it's there)
-- CREATE INDEX IF NOT EXISTS idx_papers_year ON papers (year);

COMMIT;
