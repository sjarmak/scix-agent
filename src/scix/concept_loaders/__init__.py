"""Per-vocabulary concept loaders for the cross-discipline substrate.

Each module exports a ``load(conn)`` function that downloads the source,
parses it into ``Concept`` / ``ConceptRelationship`` records, and writes
them via :mod:`scix.concepts` (which manages staging tables, COPY, and
FK-safe upsert).
"""
