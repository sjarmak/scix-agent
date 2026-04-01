-- 006_graph_metrics.sql
-- Precomputed graph metrics: PageRank, HITS, Leiden communities

BEGIN;

CREATE TABLE IF NOT EXISTS paper_metrics (
    bibcode TEXT PRIMARY KEY REFERENCES papers(bibcode),
    pagerank DOUBLE PRECISION,
    hub_score DOUBLE PRECISION,
    authority_score DOUBLE PRECISION,
    community_id_coarse INTEGER,
    community_id_medium INTEGER,
    community_id_fine INTEGER,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pm_community_coarse ON paper_metrics(community_id_coarse);
CREATE INDEX IF NOT EXISTS idx_pm_community_medium ON paper_metrics(community_id_medium);
CREATE INDEX IF NOT EXISTS idx_pm_community_fine ON paper_metrics(community_id_fine);
CREATE INDEX IF NOT EXISTS idx_pm_pagerank ON paper_metrics(pagerank DESC);

CREATE TABLE IF NOT EXISTS communities (
    community_id INTEGER NOT NULL,
    resolution TEXT NOT NULL CHECK (resolution IN ('coarse', 'medium', 'fine')),
    label TEXT,
    paper_count INTEGER NOT NULL DEFAULT 0,
    top_keywords TEXT[] NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (community_id, resolution)
);

COMMIT;
