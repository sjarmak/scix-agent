-- Migration 037: Citation consistency proxy column + entity link disputes.
--
-- Part of PRD §S1 (citation consistency precision proxy) and §S5 (researcher
-- feedback loop). See u14-should-haves work unit and
-- .claude/prd-build-artifacts/plan-u14-should-haves.md.
--
-- S1: document_entities.citation_consistency stores the fraction of outbound
--     citations from `bibcode` that also link to `entity_id` at the same
--     link_type. Null means "not computed yet".
--
-- S5: entity_link_disputes is a lightweight append-only table where
--     researchers (or human curators) can flag suspected incorrect links.
--     Not joined into the hot path — consumed by offline audit jobs.

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. document_entities.citation_consistency
-- -----------------------------------------------------------------------------
ALTER TABLE document_entities
    ADD COLUMN IF NOT EXISTS citation_consistency REAL;

COMMENT ON COLUMN document_entities.citation_consistency IS
    'Fraction of outbound citations from bibcode that also link to entity_id '
    '(precision proxy, 0..1). NULL = not yet computed. See PRD §S1.';

-- -----------------------------------------------------------------------------
-- 2. entity_link_disputes
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_link_disputes (
    id          BIGSERIAL PRIMARY KEY,
    bibcode     TEXT,
    entity_id   BIGINT,
    reason      TEXT,
    reported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    tier        SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_entity_link_disputes_bibcode
    ON entity_link_disputes(bibcode);

CREATE INDEX IF NOT EXISTS idx_entity_link_disputes_entity
    ON entity_link_disputes(entity_id);

CREATE INDEX IF NOT EXISTS idx_entity_link_disputes_reported_at
    ON entity_link_disputes(reported_at);

COMMENT ON TABLE entity_link_disputes IS
    'Append-only researcher feedback on suspected incorrect document->entity '
    'links. Consumed by offline audit jobs, not the hot path. See PRD §S5.';

COMMIT;
