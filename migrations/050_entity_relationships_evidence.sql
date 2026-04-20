-- migration: 050_entity_relationships_evidence — M2 (entity-enrichment epic)
--
-- Add an `evidence` JSONB column to public.entity_relationships so hierarchy
-- extractors (GCMD path, SPASE region dot-notation, SsODNet sso_class,
-- curated mission->instrument) can record the derivation that produced
-- each edge.  Keeping evidence alongside the edge makes the graph
-- auditable and lets downstream MCP tools show "why is X parent_of Y?".
--
-- The column is NULLABLE so existing inserts via
-- src/scix/harvest_utils.py::upsert_entity_relationship keep working
-- without modification.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS + CREATE INDEX IF NOT EXISTS.
--
-- OPERATOR NOTE: the ALTER TABLE needs a brief AccessExclusiveLock on
-- entity_relationships.  If concurrent REFRESH MATERIALIZED VIEW
-- CONCURRENTLY holds an AccessShareLock on the table, the ALTER will
-- block indefinitely.  Run during a quiet window, or set a
-- `lock_timeout` and retry.  Because the table is tiny (<2M rows
-- expected after this bead lands), the ALTER itself is near-instant
-- once the lock is acquired.

BEGIN;

-- Keep the ALTER in its own statement so a lock_timeout failure
-- rolls back this statement only, not the whole transaction.
SET LOCAL lock_timeout = '30s';

ALTER TABLE public.entity_relationships
    ADD COLUMN IF NOT EXISTS evidence JSONB;

COMMENT ON COLUMN public.entity_relationships.evidence IS
    'Optional derivation metadata: {"method": "...", "path": "...", "source_field": "..."}';

-- Index creation can proceed without holding an ACCESS EXCLUSIVE lock
-- long-term (CREATE INDEX CONCURRENTLY is better, but it can't run
-- inside a transaction block so we keep regular CREATE INDEX here
-- and accept the brief write-block it implies).
CREATE INDEX IF NOT EXISTS idx_entity_relationships_evidence
    ON public.entity_relationships USING gin (evidence jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_entity_relationships_source
    ON public.entity_relationships (source);

CREATE INDEX IF NOT EXISTS idx_entity_relationships_predicate
    ON public.entity_relationships (predicate);

COMMIT;
