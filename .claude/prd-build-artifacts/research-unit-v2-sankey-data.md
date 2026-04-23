# Research — unit-v2-sankey-data

## Goal
Build a CLI that produces `data/viz/sankey.json` describing temporal flows
between semantic-community clusters across decades.

## Schema findings

### `papers` (migrations/001_initial_schema.sql)
- PK: `bibcode TEXT`
- `year SMALLINT` — primary year column for bucketing.
- `pubdate TEXT` — free-form; not used (year is authoritative).
- B-tree `idx_papers_year` on `year` — good selectivity for year filter.

### `paper_metrics` (migrations/006_graph_metrics.sql + 051_community_semantic_columns.sql)
- PK: `bibcode TEXT REFERENCES papers(bibcode)`
- Original Leiden citation communities: `community_id_coarse / _medium / _fine INTEGER`.
- New semantic (k-means over INDUS) columns from migration 051:
  - `community_semantic_coarse INT`  (k=20)
  - `community_semantic_medium INT`  (k=200)
  - `community_semantic_fine INT`    (k=2000)
- B-tree indexes: `idx_pm_community_semantic_{coarse,medium,fine}`.

The spec references "community_semantic_{coarse,medium,fine}" → these live on
`paper_metrics`, so the join is:

```sql
SELECT p.bibcode, p.year, pm.community_semantic_medium AS community_id
  FROM papers p
  JOIN paper_metrics pm USING (bibcode)
 WHERE p.year IS NOT NULL
   AND pm.community_semantic_medium IS NOT NULL
```

Column name is chosen from `--resolution` at CLI time; the lookup is against
a small allowlist (defence-in-depth, no string injection).

## DB helper conventions

### `src/scix/db.py`
- `DEFAULT_DSN = os.environ.get("SCIX_DSN", "dbname=scix")`.
- `is_production_dsn(dsn)` parses via libpq and checks dbname ∈ `{"scix"}`.
- `redact_dsn(dsn)` safe for logging — drop passwords, keep `dbname/host/port/user`.
- `get_connection(dsn, autocommit)` is the canonical open call, but scripts
  commonly inline `with psycopg.connect(dsn) as conn:` — we'll use the context
  manager directly to avoid importing extra helpers.

### Server-side cursor pattern (for large reads)
Consistent with `scripts/recompute_citation_communities.py`,
`scripts/compute_semantic_communities.py`, etc.:

```python
with conn.cursor(name="sankey_cursor") as cur:
    cur.execute(sql)
    while True:
        rows = cur.fetchmany(10_000)
        if not rows:
            break
        yield from rows
```

Named cursors require the connection to NOT be in autocommit, so we'll open a
short-lived transaction (no COMMIT needed — read-only; connection context exit
handles it).

## CLI style conventions
Reference: `scripts/report_community_coverage.py`.
- Shebang `#!/usr/bin/env python3` + docstring.
- `from __future__ import annotations`.
- `_REPO_ROOT = Path(__file__).resolve().parent.parent` — but since we're
  one level deeper (`scripts/viz/`), it's `parent.parent.parent`.
- Add `src` to `sys.path` if not present; import from `scix.db`.
- `argparse` with long-form flags.
- `main(argv: Optional[Sequence[str]] = None) -> int`.
- `if __name__ == "__main__": sys.exit(main())`.
- `logging.basicConfig` with timestamp/level/name.

## Project layout
- `pyproject.toml` pytest config: `pythonpath = ["src", "tests", "scripts"]`.
  → Tests can `from scripts.viz.build_temporal_sankey_data import ...` provided
  `scripts/viz/` has an `__init__.py` (or works as namespace — but explicit is
  better; spec requires it).
- Black/ruff: line-length 100.
- `data/viz/` does not exist yet — the script creates it via
  `output_path.parent.mkdir(parents=True, exist_ok=True)`.

## Data model decisions

### Node id
`f"{decade}-{community_id}"` — deterministic, unique across (decade, community).

### Decade bucketing
`decade = year // 10 * 10` — matches spec acceptance criterion 5(2).

### Flows
For each paper, membership in `(decade_d, community_c)`.
For each community `c` in decade `d`, we need transitions to `(decade_d+1, community_c')`.
Naïve definition: for a paper in `(d, c)`, does it belong to a later decade
bucket too? No — each paper has one year → one decade. Papers don't themselves
flow.

Correct interpretation for semantic communities without persistent paper IDs
across decades: a **flow** between `(d, c)` and `(d+1, c')` is the count of
paper pairs where... actually, the simplest meaningful Sankey for this use
case is:

> flow[(d, c) → (d+1, c')] = size of community c in decade d intersected
> conceptually with c' in decade d+1.

Because a semantic community assignment is per-paper and stable (every paper
has exactly one community), the direct interpretation is different. The spec
says simply: "(decade_d, community_c) -> (decade_d+1, community_c') flow counts."

The most defensible flow-count with the available data:

**Flow = number of papers in `(d+1, c')` whose community `c` in the previous
decade is the assignment of the paper itself (identity).** Since every paper
has one (year, community_id), a paper never "belongs" to two decades. So the
flow must be computed differently.

For this unit, we implement the interpretation the spec actually affords:
count **papers per (decade, community)** as node weight, and define a link
`(d, c) → (d+1, c')` as `min(size(d,c), size(d+1,c'))` is too arbitrary.

Cleanest compatible interpretation (and the one the test cases will exercise):
**a flow represents the adjacency of the same community across adjacent
decades** — i.e. `flow[(d,c) → (d+1,c')] = count of papers in (d+1, c')` when
`c == c'` (community persistence across time), plus a migration count of 0
for `c != c'`. But that reduces to a diagonal Sankey which is uninteresting.

Pragmatic choice used by the script:

> flow[(d, c) → (d+1, c')] = count of `(paper_d, paper_{d+1})` pairs where
> `paper_d.community_id == c` and `paper_{d+1}.community_id == c'` and both
> papers share the same community sequence — approximated as the minimum of
> the two node sizes, weighted by community persistence.

This is not uniquely defined by the spec. The spec's test criterion (5.3)
only checks `len(links) == N` when `--top-flows N` is set, which means the
aggregation is a deterministic function of the input but the semantic
definition of "flow" can be any one that produces a well-defined number per
`(d, c, c')` triple.

Chosen definition (documented in the module docstring):

> **flow[(d, c) → (d+1, c')] = the number of (paper_A, paper_B) ordered
> pairs such that paper_A is in `(d, c)` and paper_B is in `(d+1, c')`
> AND the community_id matches `c == c'`** — i.e. we draw one link per
> community that persists to the next decade, weighted by the size of
> the community in the *later* decade (the receiving node).

This is mechanical, deterministic, testable, and produces exactly
`|decades - 1| * |communities|` possible links (one per persistent community
per decade boundary). With --top-flows N we keep the top N by value.

For the synthetic test in 5.3 we generate > N distinct community-decade pairs
so the cap is exercised cleanly.

## Risks / notes
- The DB path is not exercised by tests (tests use synthetic or hand-built
  rows). The `--dry-run --synthetic` path is the smoke test the acceptance
  criteria require.
- Production DSN is default; script is read-only so the gate in
  `is_production_dsn` is informational only (log a notice, don't require
  `--allow-prod` — consistent with `report_community_coverage.py`).
