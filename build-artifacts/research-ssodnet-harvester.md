# Research: SsODNet Harvester

## Reference Patterns

### harvest_gcmd.py

- sys.path.insert for src/ imports
- Lazy `_get_client()` for ResilientClient
- HarvestRunLog lifecycle: start(config) -> complete(records_fetched, records_upserted, counts) / fail(error)
- upsert_entity, upsert_entity_identifier, upsert_entity_alias from harvest_utils
- argparse CLI with --dsn, --dry-run, -v
- \_write_entity_graph() helper for batch entity+identifier+alias writes

### harvest_spdf.py

- Similar pattern, uses FetchResult NamedTuple
- store_harvest() with HarvestRunLog inside
- Writes datasets + entities + identifiers + relationships
- conn.commit() after each logical batch

### harvest_utils.py

- HarvestRunLog(conn, source).start(config) -> run_id
- .complete(records_fetched, records_upserted, counts)
- .fail(error_message)
- upsert_entity(conn, canonical_name, entity_type, source, discipline, harvest_run_id, properties) -> entity_id
- upsert_entity_identifier(conn, entity_id, id_scheme, external_id, is_primary)
- upsert_entity_alias(conn, entity_id, alias, alias_source)

### http_client.py

- ResilientClient with .get(url, params, \*\*kwargs) -> Response|CachedResponse
- Response has .text, .json(), .content
- Constructor: max_retries, backoff_base, rate_limit, cache_dir, cache_ttl, user_agent, timeout

### 022_staging_entities.sql

- staging.entities (id SERIAL, canonical_name, entity_type, discipline, source, properties JSONB, UNIQUE(canonical_name, entity_type, source))
- staging.entity_identifiers (entity_id INT, id_scheme, external_id, PK(id_scheme, external_id))
- staging.entity_aliases (entity_id INT, alias, alias_source, PK(entity_id, alias))
- staging.promote_entities() -> INTEGER: upserts from staging to public, remaps IDs, truncates staging

### 021_entity_graph.sql

- entities table has harvest_run_id FK to harvest_runs
- Note: staging.entities does NOT have harvest_run_id column

## SsODNet Data Model

- ssoBFT Parquet: ~2.1GB, columns include sso_name, sso_number, other_designations (pipe-separated), spkid, diameter, albedo, taxonomy_class
- API: https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/{name}
- Bulk URL: https://ssp.imcce.fr/data/ssoBFT-latest_Asteroid.parquet

## Design Decisions

1. Bulk mode: download Parquet -> read with pyarrow -> COPY to staging tables -> promote_entities()
2. Seed mode: query API for well-known objects -> upsert_entity helpers directly
3. Properties JSONB: {diameter, albedo, taxonomy}
4. entity_identifiers: id_scheme='ssodnet' (name), id_scheme='sbdb_spkid' (SPK-ID)
5. entity_aliases: all other_designations + sso_number if present
6. Note: staging.entities lacks harvest_run_id, so bulk mode sets it after promote via UPDATE
