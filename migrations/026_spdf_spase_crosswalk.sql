-- 026_spdf_spase_crosswalk.sql
-- Explicit mapping between CDAWeb (SPDF) dataset IDs and SPASE ResourceIDs.
-- e.g. AC_H2_MFI -> spase://NASA/NumericalData/ACE/MAG/L2/PT16S

BEGIN;

CREATE TABLE IF NOT EXISTS spdf_spase_crosswalk (
    id SERIAL PRIMARY KEY,
    spdf_id TEXT NOT NULL,
    spase_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'spdf_harvest',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (spdf_id, spase_id)
);

CREATE INDEX IF NOT EXISTS idx_crosswalk_spdf ON spdf_spase_crosswalk(spdf_id);
CREATE INDEX IF NOT EXISTS idx_crosswalk_spase ON spdf_spase_crosswalk(spase_id);

-- Populate from existing datasets harvested by SPDF
-- The canonical_id column holds the CDAWeb dataset ID, and
-- properties->>'spase_resource_id' holds the SPASE ResourceID.
INSERT INTO spdf_spase_crosswalk (spdf_id, spase_id, source)
SELECT
    canonical_id,
    properties->>'spase_resource_id',
    'spdf_harvest_seed'
FROM datasets
WHERE source = 'spdf'
  AND properties->>'spase_resource_id' IS NOT NULL
  AND properties->>'spase_resource_id' != ''
ON CONFLICT DO NOTHING;

COMMIT;
