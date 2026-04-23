# Qdrant pilot — find_similar_by_examples

Prototype for the MLOps Community talk. Lets an agent issue
"more like these, less like those" queries over INDUS embeddings, with
optional filters on year, arxiv class, or semantic community. Additive to
the Postgres/pgvector stack — feature-flagged via `QDRANT_URL`.

## Bring the stack up

```bash
# one-time: container starts with --restart unless-stopped
docker run -d --name scix-qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v /home/ds/projects/scix_experiments/.qdrant_storage:/qdrant/storage \
    --restart unless-stopped \
    qdrant/qdrant:latest

# load pilot (top-N PageRank papers that have BOTH a semantic community
# assignment and an arxiv_class, so filter demos are meaningful)
scix-batch --mem-high 6G --mem-max 10G \
    .venv/bin/python scripts/qdrant_upsert_pilot.py --limit 400000 --batch 1000

# sanity-check
.venv/bin/python -c "
from qdrant_client import QdrantClient
c = QdrantClient('http://127.0.0.1:6333')
print(c.get_collection('scix_papers_v1'))
"
```

Upsert throughput on this box: ~660 points/s (RTX 5090 host, Postgres on
loopback). 400K points = ~10 min. Storage footprint: ~2.6 GB on disk.

## Use the tool

```bash
# end-to-end demo — three queries: positive-only, positive+negative,
# positive + payload filter
QDRANT_URL=http://127.0.0.1:6333 \
    .venv/bin/python scripts/demo_qdrant_find_similar.py
```

Saved example output: `docs/slides_assets/qdrant_demo_output.txt`.

## MCP integration

When `QDRANT_URL` is set, the MCP server registers a 14th tool
`find_similar_by_examples`. Without it, the tool is omitted and the
startup self-test continues to expect exactly 13 tools. Both modes are
covered by `tests/test_qdrant_tools.py`.

Tool signature:

```json
{
  "positive_bibcodes": ["2019A&A...622A...2P", "..."],
  "negative_bibcodes": ["..."],
  "limit": 10,
  "year_min": 2015,
  "year_max": 2024,
  "doctype": ["article", "review"],
  "community_semantic": 14,
  "arxiv_class": ["astro-ph.EP"]
}
```

Returns:

```json
{
  "backend": "qdrant",
  "collection": "scix_papers_v1",
  "results": [
    {
      "bibcode": "...",
      "title": "...",
      "year": 2020,
      "first_author": "...",
      "score": 0.91,
      "arxiv_class": ["..."],
      "community_semantic": 14,
      "doctype": "article"
    }
  ]
}
```

## Design notes

- **Not a source of truth.** Postgres+pgvector keeps the full 32M-paper
  corpus. Qdrant holds a 400K subset (top PageRank among papers with
  community + arxiv assignments) as an enrichment layer.
- **Citation-Leiden communities are currently all `-1`** in paper_metrics
  (Leiden OOM'd at full-graph scale). We use the semantic-community
  signal (k-means on INDUS embeddings) for filter demos instead. When
  citation-Leiden is re-run, the upsert can be updated to carry both.
- **Point IDs** are 63-bit hashes of bibcode via blake2b — stable and
  collision-free for practical purposes. Makes upserts idempotent.
- **Named vectors**: current pilot has one vector per point (`indus`).
  When text-embedding-3-large coverage is wider than the 20K pilot,
  adding a second named vector is a schema-only change — agents can
  then fuse both in a single Qdrant query.
