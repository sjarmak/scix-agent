# Plan: entity-graph-schema

## Steps

1. Create migrations/021_entity_graph.sql with:
   - entities table (replaces entity_dictionary as canonical store)
   - entity_identifiers table (normalized external IDs)
   - entity_aliases table (normalized aliases)
   - entity_relationships table (entity-to-entity links)
   - document_entities bridge table (bibcode-entity links, no FK on bibcode)
   - datasets table (external datasets)
   - dataset_entities table (dataset-entity links)
   - document_datasets table (bibcode-dataset links, no FK on bibcode)
   - entity_dictionary_compat view (backward compat for dictionary.py)
   - Seed migration: copy data from entity_dictionary into new tables

2. Create tests/test_entity_graph_schema.py:
   - Parse SQL file and verify all tables, indexes, constraints, view, seed statements

3. Run tests, fix failures

4. Commit
