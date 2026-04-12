-- 029_ontology_version_pinning.sql
-- S4: Pin each entity to the ontology version it was harvested from, and
-- allow a newer entity row to supersede an older one via self-reference.
--
-- Note on type: entities.id is INTEGER (SERIAL). The PRD spec text said
-- "supersedes_id BIGINT" but the foreign-key column MUST match the referent.
-- We use INTEGER here; if the id sequence is migrated to bigint later, a
-- follow-up migration will widen supersedes_id in lockstep.
--
-- Idempotent: safe to re-run.

BEGIN;

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS source_version TEXT;

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS supersedes_id INTEGER;

-- Self-FK guarded by DO block for idempotency
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'entities_supersedes_id_fkey'
           AND conrelid = 'public.entities'::regclass
    ) THEN
        ALTER TABLE entities
            ADD CONSTRAINT entities_supersedes_id_fkey
            FOREIGN KEY (supersedes_id)
            REFERENCES entities(id)
            ON DELETE SET NULL;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_entities_supersedes_id
    ON entities(supersedes_id)
    WHERE supersedes_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_entities_source_version
    ON entities(source, source_version)
    WHERE source_version IS NOT NULL;

COMMIT;
