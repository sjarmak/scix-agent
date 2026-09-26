# MCP search latency profile, 2026-09-26

## Decision

The 30-day search tail is not caused by reciprocal-rank fusion or steady-state
query embedding. It is concentrated in two serial lanes:

1. body-text lexical search when its GIN scan encounters a cold, large match
   set;
2. filtered Qdrant retrieval, which fetches 500 neighbors and then hydrates and
   filters them in PostgreSQL because the live Qdrant payload contains only
   `bibcode`.

The first implementation priority is the existing Qdrant filter-payload
backfill, `scix_experiments-2xi8`. The next independent performance bead should
benchmark and bound the body lane's candidate working set without changing
retrieval quality blindly. Persisting per-lane timings should accompany the
next search change so future regressions can be attributed from telemetry.

## Scope and safety

This investigation used read-only queries against the production `scix`
database, read-only Qdrant collection and point inspection, and local source
inspection. No index, configuration, database row, or Qdrant point was changed.
Production replays ran with `default_transaction_read_only=on` and a 30-second
statement timeout. CPU embedding matched the repository-local MCP configuration.

## Historical request shape

`query_log` contains 209 non-test search calls in the 30 days ending
2026-09-26. The calls occurred on only seven calendar dates, with the latest on
2026-09-23, so this is a small, bursty sample rather than a continuous workload.

| Slice | n | Mean | p50 | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All search | 209 | 5.361 s | 1.043 s | 24.126 s | 34.302 s |
| Hybrid | 174 | 5.391 s | 1.388 s | 21.681 s | 34.302 s |
| Keyword | 30 | 5.952 s | 0.194 s | 30.081 s | 30.134 s |
| Semantic | 5 | 0.764 s | 0.451 s | 2.408 s | 2.895 s |

The distribution is strongly bimodal: 103 calls completed below one second,
while 19 took at least 20 seconds. Fourteen rows recorded errors. Three were
aborted-transaction errors averaging 31.095 seconds and two were statement
timeouts averaging 30.077 seconds. Eight policy-guard rejections averaged 3.5
milliseconds. Failed requests are included with
`success = FALSE OR error_msg IS NOT NULL`; filtering only on both conditions
would omit structured errors.

Filtered calls account for 198 of 209 requests and have a 23.913-second p95.
The eleven nominally unfiltered calls are too few and too mixed with guard
responses for a useful comparison.

## Lane attribution

The current handler embeds the query and then executes title/abstract lexical,
body lexical, Qdrant dense, and RRF lanes serially. Existing `query_log` rows
store only end-to-end latency, so representative historical requests were
replayed one lane at a time against the current services.

| Historical percentile sample | Embed | Title/abstract lexical | Body lexical | Dense total | Qdrant portion | PostgreSQL hydration | RRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p10 | 47 ms | 59 ms | 67 ms | 1,634 ms | 836 ms | 199 ms | 0.06 ms |
| p50 | 701 ms | 251 ms | 706 ms | 776 ms | 598 ms | 178 ms | 0.06 ms |
| p75, `earthquake ground motion validation`, year >= 1990 | 66 ms | 740 ms | 7,784 ms | 3,637 ms | 2,098 ms | 1,539 ms | 0.08 ms |
| p95 | 188 ms | 133 ms | 559 ms | 633 ms | 496 ms | 136 ms | 0.05 ms |
| p99 | 244 ms | 12 ms | 17 ms | 1,195 ms | 1,048 ms | 146 ms | 0.03 ms |

The first dense call also paid about 599 milliseconds of client/import setup.
Loading the INDUS model into a fresh CPU process took 7.647 seconds. Warm query
embeddings were normally 47--244 milliseconds; one replay took 701
milliseconds. Cold process startup can therefore create a large first-request
penalty, but embedding does not explain the steady warm tail.

The current service did not reproduce the old p95 and p99 requests at their
historical latencies. This is expected for a cache-sensitive workload measured
against current process and service state. The p75 replay did reproduce the
failure shape: body search plus filtered dense retrieval consumed 11.421
seconds before the other serial work.

## Why the two lanes are slow

### Body lexical

The body lane has no candidate-pool bound. It finds matches through the body
GIN expression index, but orders every surviving match by the title/abstract
`ts_rank_cd` value before applying the result limit.

For `earthquake ground motion validation` with `year >= 1990`, a warm
`EXPLAIN (ANALYZE, BUFFERS)` reported:

- 29,674 body-index matches;
- 28,994 rows remaining after the year filter;
- 27,902 exact heap blocks;
- 175,782 shared-buffer hits, approximately 1.34 GiB of buffer accesses;
- 321 milliseconds execution after the same query had warmed the cache.

The initial lane replay took 7.784 seconds. The warm-plan result therefore does
not contradict the tail; together they show that a large working set is cheap
when resident and expensive when cold. Unlike title/abstract lexical search,
which caps its candidate pool at 30,000, body lexical has no bound before its
ranking sort.

### Filtered dense retrieval

For any SQL or entity filter, `_vector_search_qdrant` raises the requested
neighbor count tenfold, capped at 500. Hybrid search requests 60 dense
candidates, so every filtered hybrid call asks Qdrant for 500 points. PostgreSQL
then fetches those bibcodes, applies the filters, and reconstructs Qdrant rank
order.

A read-only scroll of the live `scix_indus_v2_papers_s1` collection returned a
payload containing only `bibcode`. Year, document type, arXiv class, bibstem,
community, and retraction state cannot yet be pushed into Qdrant. This matches
the source comment and the existing operator bead `scix_experiments-2xi8`,
which will backfill the approved filter fields.

The p75 replay spent 2.098 seconds in Qdrant and another 1.539 seconds hydrating
and filtering in PostgreSQL. Even the faster samples spent roughly 0.5--1.0
seconds in Qdrant and 0.14--0.20 seconds in hydration. RRF took less than 0.1
millisecond in every sample.

## Ranked fixes

1. Complete `scix_experiments-2xi8`, then change dense search to push supported
   filters into Qdrant and request only the candidates needed. This removes the
   systematic 500-point over-fetch and most PostgreSQL post-filter work. The
   change should retain SQL post-filtering only for unsupported filters and
   verify result count, relevance, and p95 before retiring that fallback.
2. Evaluate body-lane bounding in `scix_experiments-anqp9`. Compare the current
   query with a pre-ranking candidate cap and with body search disabled for
   abstract-only requests. Use a representative body-query set and measure
   nDCG/recall alongside cold and warm p50/p95. Do not copy the title lexical
   cap without evaluation because heap-order candidate selection can lose
   relevant documents.
3. Persist lane timings in `query_log` under `scix_experiments-zh84g`:
   embedding, title lexical, body lexical, Qdrant request, PostgreSQL hydration,
   reranking, and total. The search result already carries most lane timings,
   but the handler logs only total latency. This is the smallest diagnostic
   change that converts the next tail report from replay-based inference to
   direct attribution.

CPU model preload or standardizing the serving process on the already-supported
GPU configuration may reduce first-request latency for short-lived MCP
sessions. That is cold-start work, not the primary fix for warm p95, and the
current evidence does not justify a separate implementation bead yet.

## Acceptance measurements for follow-up work

Use a fixed query corpus containing broad body matches, narrow searches, and
each supported Qdrant filter. Measure from both cold and warm cache states.
Report end-to-end and per-lane p50/p95, error count, and timeout count. Dense
filter work must also report returned-result count and retrieval quality; body
lane work must report recall and nDCG against the uncapped behavior. A fix is a
regression if it lowers latency by silently returning fewer or worse results.

## Limits

This profile does not claim that every historical 20--30-second request had the
same cause. The persisted telemetry cannot support that precision, the sample
spans heterogeneous process states, and current replays occurred after later
error-handling changes. It does establish the current serial critical path,
reproduce one multi-second tail shape, rule out RRF, bound warm embedding cost,
and identify the concrete over-fetch and unbounded-working-set mechanisms.
