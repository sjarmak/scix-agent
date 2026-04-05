# Research: enrich-wikidata-multi

## Existing Pattern (enrich_wikidata_instruments.py)

The existing script queries Wikidata for AAS instruments one-at-a-time:

- Builds individual SPARQL queries per entity name (rdfs:label exact match)
- Uses `execute_sparql()` with retry logic (3 attempts, exponential backoff)
- Parses QID, altLabels, and sub-components (P527) from results
- Merges aliases case-insensitively, updates metadata with wikidata_qid
- Rate limits with `time.sleep(delay)` between each request
- No caching to disk
- No batching (1 query per entity)

## New Requirements Differ

The multi-discipline enricher must:

1. Support 3 entity types: gcmd/instrument, pds4/mission, pds4/target
2. Batch queries: 50 names per SPARQL VALUES clause (not 1-at-a-time)
3. Cache SPARQL JSON to disk
4. > = 2 second sleep between batches

## SPARQL Batch Query Pattern

Use VALUES clause to batch multiple names:

```sparql
SELECT ?item ?itemLabel ?altLabel ?name WHERE {
  VALUES ?name { "Name1"@en "Name2"@en ... }
  ?item rdfs:label ?name .
  FILTER(LANG(?name) = "en")
  OPTIONAL { ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
```

The ?name variable in bindings links results back to the input name.

## Entity Dictionary Schema

- Table: entity_dictionary
- Unique on: (canonical_name, entity_type, source)
- Key fields: aliases (list[str]), metadata (JSONB)
- Functions: upsert_entry(), bulk_load(), lookup()
- Allowed entity_types: instrument, dataset, software, method, mission, observable, target

## Source/Type Combos to Enrich

| source | entity_type | origin harvester |
| ------ | ----------- | ---------------- |
| gcmd   | instrument  | harvest_gcmd.py  |
| pds4   | mission     | harvest_pds4.py  |
| pds4   | target      | harvest_pds4.py  |
