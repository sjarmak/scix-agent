# Plan: harvest-spdf

## Overview

Create `scripts/harvest_spdf.py` that harvests datasets, observatories, and instruments from the CDAWeb REST API into the entity graph tables (migration 021) and entity_dictionary (backward compat).

## Implementation Steps

1. **CLI + logging setup** — argparse with --dsn, --dry-run, --verbose
2. **ResilientClient instantiation** — Accept: application/json header for all requests
3. **Fetch phase** — three GET calls:
   - `/dataviews/sp_phys/datasets` -> dataset list
   - `/dataviews/sp_phys/observatories` -> observatory list
   - `/dataviews/sp_phys/instruments` -> instrument list
4. **Parse phase** — extract structured records from JSON responses
5. **DB write phase** (within a single transaction):
   a. Create harvest_run (status='running')
   b. Upsert observatories into `entities` (entity_type='observatory', discipline='heliophysics', source='spdf')
   c. Upsert instruments into `entities` (entity_type='instrument')
   d. Upsert datasets into `datasets` table (source='spdf', canonical_id=CDAWeb dataset ID)
   e. Create entity_identifiers for SPASE ResourceIDs (id_scheme='spase_resource_id')
   f. Create entity_identifiers for datasets (id_scheme='spdf_dataset_id')
   g. Create entity_relationships: instrument -> at_observatory
   h. Create dataset_entities: dataset -> from_instrument
   i. Write to entity_dictionary via bulk_load for backward compat
   j. Update harvest_run with counts and status='completed'
6. **Error handling** — update harvest_run with status='failed' on exception

## entity_dictionary compat note

`ALLOWED_ENTITY_TYPES` doesn't include 'observatory'. Observatories will be stored as entity_type='instrument' in entity_dictionary (same pattern as GCMD platforms). In the entities table they get the proper 'observatory' type.

## Test Plan

- Mock all ResilientClient.get() calls with realistic JSON fixtures
- Verify datasets, entities, relationships, identifiers stored correctly
- Verify harvest_run completion record
