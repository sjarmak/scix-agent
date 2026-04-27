# Citation-contexts coverage

> Generated 2026-04-27 in support of bead `scix_experiments-7avw`. Refresh
> after major `citation_contexts` ingest by re-running the queries below
> against the live `scix` database.

## Why this matters

Two MCP tools — `claim_blame` and `find_replications` — read from the
`v_claim_edges` materialized view (migration 057), which joins
`citation_contexts` to `citation_edges` and `papers`. They power "who
claimed this first?" and "who replicated / refuted this paper?" workflows.

Both tools can return an empty result for two very different reasons:

1. **No events** — the seed paper(s) ARE present in `citation_contexts`,
   but the citing literature genuinely has no replication / blame
   events. The empty result is real and trustworthy.
2. **No coverage** — the seed paper(s) are NOT in `citation_contexts`
   at all. The empty result is silent about the underlying topic;
   the agent should fall back to `citation_traverse` (full edge
   coverage), `concept_search`, or broaden its query.

Without a coverage signal an agent treats both cases as case (1), which
is wrong about half the time on out-of-corpus subdomains. The
`coverage` block surfaced on every response from these two tools makes
the distinction explicit.

## Coverage stats (as of bead 79n, 2026-04-27)

| Surface                                              |              Value |
| ---------------------------------------------------- | -----------------: |
| Total citation edges (`citation_edges`)              |        299,329,159 |
| Citation contexts (`citation_contexts` rows)         |          ~825,000  |
| Edges covered by at least one context                |          ~810,000  |
| Edge coverage percentage                             |             ~0.27% |
| Distinct source papers with any context              |           ~30,000  |
| Distinct cited papers with any context               |          ~250,000  |

The 0.27% headline number is what the response `note` reports. It is
intentionally pessimistic — it counts edges, not papers — because that's
the surface `claim_blame` walks.

## How the response block is computed

Both tools call `scix.citation_contexts_coverage.compute_coverage(conn,
seeds)`. The probe issues one SQL query against `v_claim_edges`:

```sql
SELECT COUNT(DISTINCT bib) FROM (
  SELECT source_bibcode AS bib FROM v_claim_edges
    WHERE source_bibcode = ANY($seeds)
  UNION
  SELECT target_bibcode      FROM v_claim_edges
    WHERE target_bibcode = ANY($seeds)
) AS covered;
```

This uses both indexes built by migration 057
(`idx_v_claim_edges_source_intent`, `idx_v_claim_edges_target_intent`)
so the probe stays in the <10 ms p95 budget on 825K rows.

For `claim_blame`, the seeds are the candidate bibcodes returned by
the seed query (i.e. the papers whose reverse references the tool is
about to walk). For `find_replications`, the seed is the single
`target_bibcode` whose forward citations are being enumerated.

## Response shape

Both tools' responses now include:

```json
{
  "coverage": {
    "covered_seeds": 7,
    "total_seeds": 20,
    "coverage_pct": 0.35,
    "note": "citation_contexts has ~0.27% edge coverage; results may be undercounting. See docs/citation_contexts_coverage.md for the no-events vs no-coverage distinction."
  }
}
```

* `covered_seeds == 0 && total_seeds > 0` → **no coverage**. The empty
  result is silent. Switch tools or broaden the query.
* `covered_seeds > 0 && len(papers) == 0` → **no events**. The empty
  result is informative — the citing literature genuinely doesn't
  contain replication / blame events for these seeds.
* `coverage_pct < 0.5` → results are likely undercounting. Treat as
  partial evidence; consider broader-coverage tools as a complement.
* `coverage_pct >= 0.8` → results are reasonably complete; trust them.

## Future work

The 0.27% gap is being closed incrementally as the citation-context
extraction pipeline runs over more papers (bead `79n` tracks
backfill). When edge coverage exceeds ~10% the `note` string in
`scix/citation_contexts_coverage.py` should be updated to reflect the
new figure.
