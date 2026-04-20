-- 014_discipline_and_indexes.sql
-- Add discipline column to entity_dictionary with backfill and functional index

BEGIN;

-- 1. Add nullable discipline column
ALTER TABLE entity_dictionary ADD COLUMN IF NOT EXISTS discipline TEXT;

-- 2. Btree index on discipline
CREATE INDEX IF NOT EXISTS idx_entity_dict_discipline
    ON entity_dictionary (discipline);

-- 3. Functional index for case-insensitive canonical_name lookups
CREATE INDEX IF NOT EXISTS idx_entity_dict_canonical_lower
    ON entity_dictionary (lower(canonical_name));

-- 4. Backfill discipline='astrophysics' for all known astronomy sources
UPDATE entity_dictionary
   SET discipline = 'astrophysics'
 WHERE source IN ('ascl', 'aas', 'physh', 'pwc', 'astromlab', 'vizier', 'ads_data')
   AND discipline IS NULL;

COMMIT;
