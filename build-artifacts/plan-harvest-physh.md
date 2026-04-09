# Plan: harvest-physh

## Step 1: Create scripts/harvest_physh.py

1. Download physh.json.gz from GitHub (raw URL, master branch)
2. Parse JSON-LD: build by_id index
3. Find Techniques facet (UUID fa2a6718-de5c-4c05-bf00-f169a55234d5)
4. Collect all concepts: walk physh_rdf:contains on sub-facets, then BFS via skos:broader children map
5. For each concept, build entity_dictionary entry:
   - canonical_name: skos:prefLabel @value
   - entity_type: 'method'
   - source: 'physh'
   - external_id: the DOI URI (@id)
   - aliases: list of skos:altLabel @values
   - metadata: {parent_names: [...], child_names: [...], parent_ids: [...], child_ids: [...], facet: sub-facet name}
6. Call dictionary.bulk_load(conn, entries)
7. CLI: argparse with --dsn, --cache-dir, -v options

## Step 2: Create tests/test_harvest_physh.py

1. Create sample JSON-LD fixture with ~5 concepts mimicking PhySH structure
2. Unit tests (no DB, no network):
   - Test parse_physh_jsonld returns correct concept count
   - Test aliases extracted from altLabels
   - Test parent/child relationships in metadata
   - Test Monte Carlo-like hierarchy traversal
3. Integration tests (marked @pytest.mark.integration):
   - Test bulk_load into entity_dictionary
   - Test lookup('Monte Carlo') finds entry with parent/child metadata
   - Test count > 200 after full load
4. Mock HTTP for download function tests

## Key Design Decisions

- Use JSON-LD (not TTL/RDF-XML) since it's trivially parseable with stdlib json
- Download gzipped file for efficiency
- Cache downloaded file to data/ directory (same pattern as UAT)
- Store hierarchy info in metadata JSONB, not separate relationship table
- entity_type='method' to distinguish from UAT concepts
