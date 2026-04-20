-- 008_community_taxonomic.sql
-- Add taxonomic community column populated from papers.arxiv_class

BEGIN;

ALTER TABLE paper_metrics ADD COLUMN IF NOT EXISTS community_taxonomic TEXT;

CREATE INDEX IF NOT EXISTS idx_pm_community_taxonomic
    ON paper_metrics(community_taxonomic);

COMMIT;
