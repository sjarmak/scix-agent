-- 033_fusion_mv.sql
-- M8: Fused confidence materialized view for document_entities.
--
--   - tier_weight(SMALLINT) IMMUTABLE SQL function mapping tier -> calibration weight.
--   - tier_weight_calibration_log table records each weight version.
--   - document_entities_canonical materialized view computes fused confidence via
--     noisy-OR:  fused = 1 - exp(sum(ln(1 - c_t * w_t)))
--   - fusion_mv_state table records dirty bit and last_refresh_at for the
--     rate-limited refresh loop.
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. tier_weight SQL function (IMMUTABLE LEAKPROOF PARALLEL SAFE)
--
--    Placeholder calibration weights — will be replaced by calibration runs
--    that write to tier_weight_calibration_log. The function body is
--    recreated by each calibration migration.
--
--    Tier semantics (u04 / M1):
--      1 = exact id match (high precision)
--      2 = alias + context
--      3 = deprecated keyword tier
--      4 = LLM-adjudicated fallback
--      5 = JIT path
--      else (including 0 / legacy) -> 0.50 default
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION tier_weight(tier SMALLINT)
RETURNS DOUBLE PRECISION
LANGUAGE sql
IMMUTABLE
LEAKPROOF
PARALLEL SAFE
AS $$
    SELECT CASE tier
        WHEN 1::SMALLINT THEN 0.98::float8
        WHEN 2::SMALLINT THEN 0.85::float8
        WHEN 3::SMALLINT THEN 0.92::float8
        WHEN 4::SMALLINT THEN 0.50::float8
        WHEN 5::SMALLINT THEN 0.88::float8
        ELSE 0.50::float8
    END
$$;

-- ---------------------------------------------------------------------------
-- 2. tier_weight_calibration_log — one row per calibration version
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tier_weight_calibration_log (
    id          SERIAL PRIMARY KEY,
    version     TEXT NOT NULL UNIQUE,
    weights     JSONB NOT NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO tier_weight_calibration_log (version, weights, notes)
VALUES (
    'placeholder_2026-04-12',
    '{"1": 0.98, "2": 0.85, "3": 0.92, "4": 0.50, "5": 0.88, "default": 0.50}'::jsonb,
    'Initial placeholder weights. Tier 3 is deprecated; tier 5 is the JIT path. Replace via a calibration run.'
)
ON CONFLICT (version) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3. fusion_mv_state — dirty bit + last refresh timestamp for the refresh job
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fusion_mv_state (
    id              INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    dirty           BOOLEAN NOT NULL DEFAULT true,
    last_refresh_at TIMESTAMPTZ
);

INSERT INTO fusion_mv_state (id, dirty, last_refresh_at)
VALUES (1, true, NULL)
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 4. document_entities_canonical materialized view
--
--    Noisy-OR fusion of per-tier confidences:
--        fused = 1 - exp(sum(ln(1 - LEAST(0.9999, c_t * w_t))))
--
--    The LEAST(0.9999, ...) clamp prevents ln(0) when a (confidence, weight)
--    pair happens to land on 1.0. For typical weights (<= 0.98) and
--    confidences (<= 1.0), the clamp is never binding, so the closed-form
--    equivalence holds within floating-point tolerance.
--
--    Drop + recreate (idempotent). The UNIQUE index below is what makes
--    REFRESH MATERIALIZED VIEW CONCURRENTLY legal.
-- ---------------------------------------------------------------------------

DROP MATERIALIZED VIEW IF EXISTS document_entities_canonical CASCADE;

CREATE MATERIALIZED VIEW document_entities_canonical AS
SELECT
    de.bibcode,
    de.entity_id,
    1 - exp(
        sum(
            ln(
                1 - LEAST(
                    0.9999::float8,
                    GREATEST(
                        0.0::float8,
                        de.confidence::float8 * tier_weight(de.tier)
                    )
                )
            )
        )
    ) AS fused_confidence,
    count(*)                           AS link_count,
    array_agg(DISTINCT de.tier ORDER BY de.tier) AS contributing_tiers,
    max(de.tier_version)               AS max_tier_version,
    max(de.harvest_run_id)             AS latest_harvest_run_id
FROM document_entities de
WHERE de.confidence IS NOT NULL
GROUP BY de.bibcode, de.entity_id;

-- UNIQUE index required for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_dec_bibcode_entity
    ON document_entities_canonical (bibcode, entity_id);

-- Entity-scoped top-k lookups (hot path for resolver reads)
CREATE INDEX IF NOT EXISTS idx_dec_entity_fused
    ON document_entities_canonical (entity_id, fused_confidence DESC);

-- Bibcode lookups (document-centric reads)
CREATE INDEX IF NOT EXISTS idx_dec_bibcode
    ON document_entities_canonical (bibcode);

COMMIT;
