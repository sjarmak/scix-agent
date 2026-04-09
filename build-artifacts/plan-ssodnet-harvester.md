# Plan: SsODNet Harvester

## scripts/harvest_ssodnet.py

1. Module docstring, imports (sys.path hack, argparse, logging, pathlib, json, hashlib, io)
2. Import from scix: ResilientClient, HarvestRunLog, upsert_entity, upsert_entity_identifier, upsert_entity_alias, get_connection
3. Constants: PARQUET_URL, API_BASE, SOURCE='ssodnet', DISCIPLINE='planetary_science'
4. Lazy \_get_client() for ResilientClient
5. WELL_KNOWN_OBJECTS list for seed mode (Ceres, Vesta, Pallas, etc.)

### Bulk mode functions:

6. download_parquet(url, dest_path) - download with ResilientClient, SHA-256 checksum, HTTP Range resume support
7. read_parquet(path) - read with pyarrow.parquet.read_table(), return as list of row dicts
8. parse_sso_record(row) -> dict with canonical_name, entity_type, source, properties, identifiers, aliases
9. write_staging_entities(conn, records) - COPY to staging.entities
10. write_staging_identifiers(conn, records) - COPY to staging.entity_identifiers
11. write_staging_aliases(conn, records) - COPY to staging.entity_aliases
12. run_bulk_harvest(dsn, parquet_url, dry_run) - full pipeline: download, parse, stage, promote

### Seed mode functions:

13. fetch_ssocard(client, name) -> dict - query SsODNet API
14. parse_ssocard(data) -> parsed record
15. run_seed_harvest(dsn, objects, dry_run) - fetch each object, upsert via helpers

### CLI:

16. run_harvest(dsn, mode, dry_run) - dispatch to bulk or seed
17. main() with argparse: --dsn, --mode bulk|seed, --dry-run, -v
18. if **name** == '**main**': main()

## tests/test_harvest_ssodnet.py

1. Test imports work
2. Test parse_sso_record with sample data
3. Test bulk mode uses staging schema (mock conn, verify COPY and promote_entities calls)
4. Test seed mode uses upsert helpers
5. Test entity_identifiers with id_scheme='ssodnet' and 'sbdb_spkid'
6. Test entity_aliases populated from other_designations
7. Test properties JSONB has diameter, albedo, taxonomy
8. Test HarvestRunLog lifecycle (start/complete/fail)
9. Test CLI argument parsing
10. Test dry-run mode skips DB writes
