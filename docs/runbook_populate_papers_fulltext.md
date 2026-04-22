# Runbook — populate `papers_fulltext`

Operational runbook for `scripts/populate_papers_fulltext.py`, the D4 driver
defined by [`docs/prd/prd_structural_fulltext_parsing.md`](./prd/prd_structural_fulltext_parsing.md).

The driver iterates the [`papers`](../migrations/041_papers_fulltext.sql) table
in bibcode-sorted chunks, builds a `RouteInput` per bibcode, dispatches it
through `scix.sources.route.route_fulltext_request`, runs the matching tier
parser, and bulk-writes parsed rows into `papers_fulltext` via Postgres `COPY`.
Parse failures are recorded in the
[`papers_fulltext_failures`](../migrations/047_papers_fulltext_failures.sql)
negative cache with R15 exponential backoff (24h / 3d / 7d / 30d). Existing
`papers_fulltext` rows are always skipped, so the driver is safe to re-run.

Flags referenced below come straight from
`python scripts/populate_papers_fulltext.py --help`; do not invent new ones.

---

## Launch

Dev / scratch run against `scix_test` (no prod guard, no systemd scope
required):

```bash
python scripts/populate_papers_fulltext.py \
  --dsn 'dbname=scix_test' \
  --chunk-size 1000 \
  --limit 10000
```

Full production run against `dbname=scix`. This path is gated: the script
refuses to proceed unless `--allow-prod` is passed AND it is running inside a
systemd scope (see `CLAUDE.md` §"Memory isolation — coexisting with gascity").
`scix-batch` sets `SYSTEMD_SCOPE` automatically via `systemd-run --scope`:

```bash
scix-batch --mem-high 10G --mem-max 15G \
  python scripts/populate_papers_fulltext.py \
    --dsn 'dbname=scix' \
    --allow-prod
```

Useful variations:

```bash
# Tighter batches (more frequent commits, slightly slower throughput):
scix-batch --mem-high 10G --mem-max 15G \
  python scripts/populate_papers_fulltext.py \
    --dsn 'dbname=scix' --allow-prod --chunk-size 500

# Suppress INFO logs — only warnings/errors go to stderr:
scix-batch --mem-high 10G --mem-max 15G \
  python scripts/populate_papers_fulltext.py \
    --dsn 'dbname=scix' --allow-prod --quiet
```

`--chunk-size` is capped at 10000 internally; anything higher is silently
clamped.

---

## Resume

The driver is resumable in two independent ways:

1. `--resume-from BIBCODE` — the candidate iterator filters `p.bibcode > %s`,
   so every bibcode lexicographically at or below the passed value is skipped
   before work is dispatched. This is the recovery path after a crash where
   you know the last successfully-committed bibcode.

2. Regardless of `--resume-from`, the iterator also runs a
   `NOT EXISTS (SELECT 1 FROM papers_fulltext pf WHERE pf.bibcode = p.bibcode)`
   predicate, so any bibcode that already has a `papers_fulltext` row is
   skipped automatically. Re-running the driver with no flags is always safe —
   it will only consider papers that have not yet been written.

The iterator additionally skips bibcodes whose most recent
`papers_fulltext_failures` row has `retry_after > now()`, so backed-off
failures are not retried prematurely.

Typical resume after a systemd-oomd kill:

```bash
# Find the last bibcode written:
psql 'dbname=scix' -c "SELECT max(bibcode) FROM papers_fulltext;"

# Relaunch with that value — redundant with the NOT EXISTS filter but cheaper
# than scanning from the head of papers:
scix-batch --mem-high 10G --mem-max 15G \
  python scripts/populate_papers_fulltext.py \
    --dsn 'dbname=scix' --allow-prod \
    --resume-from '2024ApJ...900A...1X'
```

---

## Kill

`SIGTERM` and `SIGINT` are safe to send. Writes are batched: the driver
accumulates up to `--chunk-size` `ParsedRow` objects, calls `COPY` + `commit()`
as a single Postgres transaction, then clears the in-memory buffer.

- If a kill arrives between batches, no rows are lost — all prior batches are
  already committed.
- If a kill arrives mid-`COPY` / before the trailing `conn.commit()`, psycopg's
  transaction rolls back and **at most one batch (1000 rows by default) is
  lost**. The lost bibcodes simply reappear in the next run's candidate set
  because they have no `papers_fulltext` row yet.

Graceful stop:

```bash
# Find the process started under systemd-run scope:
pgrep -af populate_papers_fulltext.py

# Polite SIGTERM — the in-flight COPY will either commit or roll back cleanly:
kill <pid>
```

`scix-batch` wraps the job in a transient `systemd-run --scope` unit with
`ManagedOOMPreference=avoid`; if systemd-oomd decides to kill the process
under memory pressure it fires `SIGTERM` first, which psycopg handles the
same way. Re-running the driver (see [Resume](#resume)) picks up where it
stopped.

---

## Memory

The driver is designed to stream:

- Candidate papers are read through a named (server-side) Postgres cursor
  (`populate_papers_fulltext_iter`), so the full result set is never
  materialised in Python.
- Each row is parsed individually; only the current `--chunk-size` batch
  of `ParsedRow` objects is held in memory.
- `COPY` writes are buffered by psycopg and flushed per batch.

Expected RSS: **a few GB at most** under the default chunk size, dominated
by the Python interpreter, psycopg connections, and the parser's per-row
working set. There is no in-memory accumulation proportional to the 32M-row
candidate set.

The `scix-batch --mem-high 10G --mem-max 15G` limits in the
[Launch](#launch) section are deliberately generous — they exist so that a
runaway parser on a pathological body cannot take down the gascity supervisor
sharing `user@1000.service`. Do not raise them without a concrete reason;
lower is fine if you want tighter guardrails.

---

## Rate limits

The current driver does not make outbound HTTP calls. Tier 1 parsing runs
entirely against `papers.body` (already ingested into Postgres) and the
sibling-clone short-circuit is a single-row SQL lookup against
`papers_external_ids` / `papers_fulltext`. There is nothing to throttle
today, and the driver is CPU-bound on the parser + IO-bound on `COPY`.

When Tier 2 ar5iv fetching is wired in (currently `tier3_docling` and any
future ar5iv network path are recorded as failures with the appropriate
backoff, not fetched), the expected posture will be:

- Prefer a locally-mirrored ar5iv snapshot if one is available on the host —
  fully offline, no external rate limit applies.
- Fall back to the public `ar5iv.labs.arxiv.org` endpoint only when the
  local mirror does not have the paper. External fetches must be throttled
  to well below the published arXiv rate limits (4 requests/second is the
  conservative ceiling), with a single shared token-bucket across the driver
  process.
- Any 429 / 5xx from the external endpoint should be recorded as a failure
  with the R15 backoff ladder rather than retried in a tight loop.

Until that wire-up lands, treat this section as forward-looking.

---

## Failure handling

Every parse failure upserts a row into `papers_fulltext_failures` via
`_FAILURE_UPSERT_SQL` in the driver. The row carries:

- `bibcode` — primary key (one row per paper).
- `parser_version` — which parser pin saw the failure.
- `failure_reason` — a stable tag. Current values:
  - `tier1_parse_error:<ExceptionClassName>` — parser raised on `papers.body`.
  - `tier3_not_yet_wired` — routed to Docling tier, which is not implemented.
  - `abstract_only` — no body and no sibling; nothing to parse.
- `attempts`, `first_attempt`, `last_attempt`, `retry_after`.

On conflict, `attempts` increments and `retry_after` is advanced according
to the **R15 backoff ladder** (see `docs/prd/prd_structural_fulltext_parsing.md`):

| attempts after upsert | retry_after delta |
| --------------------- | ----------------- |
| 1                     | `now() + 24h`     |
| 2                     | `now() + 3 days`  |
| 3                     | `now() + 7 days`  |
| ≥ 4                   | `now() + 30 days` |

`abstract_only` papers are parked at `now() + 365 days` since reattempting
them without new upstream data is pointless.

The candidate iterator's `NOT EXISTS ... retry_after > now()` predicate
means backed-off failures simply do not appear in the candidate stream
until their window expires. To inspect the negative cache:

```bash
psql 'dbname=scix' -c "
  SELECT failure_reason, count(*), min(retry_after) AS next_retry
    FROM papers_fulltext_failures
   GROUP BY failure_reason
   ORDER BY count(*) DESC;
"
```

To force a retry of a specific bibcode (e.g. after fixing a parser bug):

```bash
psql 'dbname=scix' -c "
  DELETE FROM papers_fulltext_failures WHERE bibcode = '2024ApJ...900A...1X';
"
```

---

## Idempotency

The driver is idempotent per `(bibcode, parser_version)`:

- `papers_fulltext.bibcode` is a primary key. The iterator's `NOT EXISTS`
  filter ensures the driver never tries to write a second row for a bibcode
  that already has one — so a re-run with the **same** `parser_version`
  (i.e. the same `ADS_PARSER_VERSION` / `AR5IV_PARSER_VERSION` pins baked
  into the driver at that point in time) is a no-op for every already-parsed
  bibcode.
- `papers_fulltext_failures.bibcode` is also a primary key, and the upsert
  statement advances `attempts` / `retry_after` deterministically on conflict.
  Replaying the same failure state across a crash does not corrupt the
  backoff schedule.

When a parser is **upgraded** (a new `parser_version` pin ships in the code),
`papers_fulltext` rows written under the older pin are still present and will
still be skipped by the `NOT EXISTS` filter. The supported path for backfilling
under the new pin is the `--reparse-from-version` flag documented in the next
section — no manual `DELETE` required.

Routine operational re-runs (cron, post-reboot, resume-after-crash) need no
cleanup — just relaunch the driver and let the idempotency filters do their
work.

---

## Re-parse (parser_version bump)

Use this mode when the ADS body parser pin is bumped and existing
`papers_fulltext` rows written under the older pin need to be refreshed
in-place. The **current instance** of this workflow is the
`ads_body_regex@v1` → `ads_body_inline_v2` bump that ships with the
inline-keyword-anchor parser rewrite (see
`docs/prd/prd_ads_body_parser_rewrite.md`). Every future parser bump follows
the same pattern — only the `OLD_VERSION` string changes.

### When to use

Run `--reparse-from-version OLD_VERSION` when **all** of the following are
true:

- The `PARSER_VERSION` constant in `src/scix/sources/ads_body_parser.py` (or
  another tier parser) has been bumped in a merged commit.
- You want to replace the `papers_fulltext.sections` / `inline_cites` /
  `parser_version` fields of already-written rows with fresh output from
  the new parser, rather than only picking up papers that had no row at all.
- You accept that the write path switches from bulk `COPY` to
  `INSERT ... ON CONFLICT (bibcode) DO UPDATE`, with the associated
  throughput cost described below.

Do **not** use this flag for routine resume-after-crash or post-reboot
relaunches — the default new-row mode already handles those via the
`NOT EXISTS` filter (see [Idempotency](#idempotency)).

### Command

Production reparse for the current parser bump. `scix-batch` is required
(the driver's production guard requires a systemd scope, and `scix-batch`
wraps the process in `systemd-run --scope` with `MemoryMax` / oomd
preference — see `CLAUDE.md` §"Memory isolation"):

```bash
SCIX_DSN="dbname=scix" scix-batch --mem-high 10G --mem-max 15G \
  python scripts/populate_papers_fulltext.py \
    --dsn 'dbname=scix' \
    --reparse-from-version 'ads_body_regex@v1' \
    --allow-prod
```

Dev / scratch reparse against `scix_test` (no systemd scope, no
`scix-batch`, no `--allow-prod`):

```bash
python scripts/populate_papers_fulltext.py \
  --dsn 'dbname=scix_test' \
  --reparse-from-version 'ads_body_regex@v1' \
  --limit 1000
```

In reparse mode the candidate iterator is driven by a SQL predicate that
selects rows from `papers_fulltext` where
`parser_version = OLD_VERSION AND source = 'ads_body'`. Rows whose
`parser_version` is already the new pin are not revisited — so once a
bibcode has been upgraded, it drops out of the candidate set automatically.

### Throughput and wall-clock budget

The write path difference costs real time:

| Mode              | Write statement                         | Budget       |
| ----------------- | --------------------------------------- | ------------ |
| Default (new row) | `COPY papers_fulltext FROM STDIN`       | ~2000 rps    |
| `--reparse-from-version` | `INSERT ... ON CONFLICT DO UPDATE` | ~500–1000 rps |

The PRD acceptance target is **≤ 4 h hard cap** of wall-clock time for a
full-corpus reparse at the current ~15M-row `papers_fulltext` size. If the
job has not finished after 4 hours, investigate (lock contention,
autovacuum backlog, pathological parser input) rather than letting it run.

### Resume semantics

Reparse mode is fully resumable, in two independent ways:

1. **Automatic resume across runs.** The candidate iterator's SQL
   predicate filters on the current `parser_version` column in
   `papers_fulltext`. Every bibcode that the driver commits is updated
   to the new pin, so a second run with the same
   `--reparse-from-version OLD_VERSION` transparently skips every already-
   upgraded bibcode and picks up only the remaining old-version rows.
   Relaunching after a systemd-oomd kill or a planned stop needs no
   manual bookkeeping — just re-run the same command.

2. **`--resume-from BIBCODE` lower bound.** As in the default mode,
   `--resume-from` adds a `p.bibcode > %s` clause on top of the
   `parser_version` filter. Use it when you know the last successfully-
   committed bibcode and want to skip the iterator's startup scan through
   already-upgraded rows. `--resume-from` composes cleanly with
   `--reparse-from-version`; both filters apply.

Typical recovery after a mid-run crash:

```bash
# Find the last bibcode updated to the new pin:
psql 'dbname=scix' -c "
  SELECT max(bibcode) FROM papers_fulltext WHERE parser_version = 'ads_body_inline_v2';
"

# Relaunch — redundant with the parser_version filter but skips the head
# of the candidate scan:
SCIX_DSN="dbname=scix" scix-batch --mem-high 10G --mem-max 15G \
  python scripts/populate_papers_fulltext.py \
    --dsn 'dbname=scix' \
    --reparse-from-version 'ads_body_regex@v1' \
    --resume-from '2024ApJ...900A...1X' \
    --allow-prod
```

`--reparse-from-version` **cannot** be combined with the default new-row
mode in a single invocation — the iterator picks a different candidate
set in each mode. If you need to both upgrade old rows and cover
never-parsed bibcodes, run the reparse pass first and then a plain
follow-up pass without the flag.
