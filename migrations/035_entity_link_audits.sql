-- 035_entity_link_audits.sql
-- M9: Human + LLM-judge audit table for entity linking.
--
-- Stores per-(tier, bibcode, entity_id, annotator) labels from either
-- human annotators or an LLM-judge. Labels are one of
-- ('correct','incorrect','ambiguous').
--
-- Primary key is (tier, bibcode, entity_id, annotator) so multiple
-- annotators can label the same link and disagreement is first-class.
--
-- Idempotent: safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS entity_link_audits (
    tier       SMALLINT NOT NULL,
    bibcode    TEXT     NOT NULL,
    entity_id  BIGINT   NOT NULL,
    annotator  TEXT     NOT NULL,
    label      TEXT     NOT NULL CHECK (label IN ('correct','incorrect','ambiguous')),
    note       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tier, bibcode, entity_id, annotator)
);

-- Secondary indexes for analytics reads
CREATE INDEX IF NOT EXISTS idx_entity_link_audits_tier_label
    ON entity_link_audits (tier, label);

CREATE INDEX IF NOT EXISTS idx_entity_link_audits_annotator
    ON entity_link_audits (annotator);

COMMIT;
