-- 013_entity_dictionary.sql
-- Entity dictionary for canonical names, aliases, and metadata

BEGIN;

CREATE TABLE IF NOT EXISTS entity_dictionary (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source TEXT NOT NULL,
    external_id TEXT,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    UNIQUE (canonical_name, entity_type, source)
);

CREATE INDEX IF NOT EXISTS idx_entity_dict_type ON entity_dictionary(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_dict_aliases ON entity_dictionary USING GIN (aliases);

COMMIT;
