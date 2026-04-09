-- 025_entity_audit_log.sql
-- Audit log for entity merge and split operations.
-- Tracks when resolution decisions change (entity A merged with B, or split).

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. entity_merge_log — records when two entities are merged
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_merge_log (
    id SERIAL PRIMARY KEY,
    old_entity_id INT NOT NULL,
    new_entity_id INT NOT NULL REFERENCES entities(id),
    reason TEXT,
    merged_by TEXT,
    merged_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_merge_log_old ON entity_merge_log(old_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_merge_log_new ON entity_merge_log(new_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_merge_log_at ON entity_merge_log(merged_at);

-- ---------------------------------------------------------------------------
-- 2. entity_split_log — records when an entity is split into children
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_split_log (
    id SERIAL PRIMARY KEY,
    parent_entity_id INT NOT NULL,
    child_entity_ids INT[] NOT NULL,
    reason TEXT,
    split_by TEXT,
    split_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_split_log_parent ON entity_split_log(parent_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_split_log_at ON entity_split_log(split_at);

COMMIT;
