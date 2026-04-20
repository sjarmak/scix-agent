-- 007_uat_concepts.sql
-- Unified Astronomy Thesaurus (UAT) concept hierarchy

BEGIN;

CREATE TABLE IF NOT EXISTS uat_concepts (
    concept_id TEXT PRIMARY KEY,
    preferred_label TEXT NOT NULL,
    alternate_labels TEXT[] NOT NULL DEFAULT '{}',
    definition TEXT,
    level INTEGER
);

CREATE INDEX IF NOT EXISTS idx_uat_preferred_label ON uat_concepts(preferred_label);
CREATE INDEX IF NOT EXISTS idx_uat_alternate_labels ON uat_concepts USING GIN (alternate_labels);

CREATE TABLE IF NOT EXISTS uat_relationships (
    parent_id TEXT NOT NULL REFERENCES uat_concepts(concept_id),
    child_id TEXT NOT NULL REFERENCES uat_concepts(concept_id),
    PRIMARY KEY (parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_uat_rel_child ON uat_relationships(child_id);

CREATE TABLE IF NOT EXISTS paper_uat_mappings (
    bibcode TEXT NOT NULL REFERENCES papers(bibcode),
    concept_id TEXT NOT NULL REFERENCES uat_concepts(concept_id),
    match_type TEXT NOT NULL CHECK (match_type IN ('exact', 'fuzzy', 'parent')),
    PRIMARY KEY (bibcode, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_pum_concept ON paper_uat_mappings(concept_id);
CREATE INDEX IF NOT EXISTS idx_pum_match_type ON paper_uat_mappings(match_type);

COMMIT;
