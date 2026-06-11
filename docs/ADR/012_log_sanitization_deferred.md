# ADR-012: Log-Record Schema-Name Sanitization Deferred (Local-Only Deployment)

- **Status**: Accepted (2026-06-05)
- **Deciders**: SciX maintainers
- **Scope**: All `logger.warning("...: %s", exc)` patterns in `src/scix/` that may render a `psycopg.Error` containing PostgreSQL schema identifiers
- **Supersedes**: none
- **Related**: bead `scix_experiments-iwdi` (security-reviewer wave `vm1r-b647-u0j1-3ozn` Phase 4 LOW finding)

## Context

The error guards in `src/scix/claim_blame.py` and `src/scix/synthesize.py` log the
caught exception at WARNING level:

- `claim_blame.py` L457 / L508 / L555 / L580
- `synthesize.py` L558 / L603 / L642 / L711
- 27 such `logger.warning("...: %s", exc)` patterns across `src/scix/` in total.

For a real PostgreSQL failure, `str(exc)` can embed internal schema identifiers —
e.g. `column "papers.bibcode" does not exist`. These strings **never reach the
agent**: each guard returns an empty value after logging, so the schema name stays
server-side. The only exposure path is a server log being forwarded to a
third-party aggregator (Datadog, Splunk, Loki, etc.), where the schema names would
travel with the record.

Two facts bound the actual risk (verified against the tree on 2026-06-01 and
re-verified 2026-06-05):

- **No third-party log aggregator exists anywhere in the stack.** No
  datadog/splunk/fluentd/logstash/loki references in `src/`, `scripts/`, `deploy/`,
  or `pyproject.toml`. (OpenTelemetry mentions are confined to the gascity/enterprise
  lit-review HTML *generator* scripts as a documentation topic — not the scix MCP
  server's logging path.) There is no `logging.Filter` sanitization layer.
- **The deployment is deliberately local-only.** The public endpoint is
  intentionally down; the forward-looking ops wishlist is Prometheus metrics export,
  not log forwarding.

The logging pattern itself is consistent with the rest of the codebase
(`citation_contexts_coverage.py`) and is operationally useful for debugging. This
is therefore a **policy** decision, not a code defect.

## Decision

**Defer (wontfix for now).** Do not build a log-sanitization layer while the
deployment is local-only. The schema-name-in-`str(exc)` path is real but
unreachable as an exfiltration vector: the value never reaches the agent, and there
is no log sink that leaves the host. Building a `logging.Filter` now is YAGNI
infrastructure for a threat that cannot occur in the current architecture, and it
would add per-record regex cost plus a maintenance surface to every WARNING log in
`src/scix/`.

No code changes to `src/scix/`. The existing `logger.warning("...: %s", exc)`
patterns stay as-is for their debugging value.

## Revisit Trigger

Reopen this decision and implement the mitigation **if and when a third-party log
aggregator / log-forwarding sink is added to the stack** (e.g. Datadog, Splunk,
Loki, Fluent Bit, or any remote log shipper). At that point:

1. Add a `logging.Filter` that strips PostgreSQL table/column identifiers from
   `LogRecord` messages before they leave the host (regex over the rendered
   message, or — preferred — narrow the guards to log `exc.diag.message_primary` /
   a sanitized summary rather than full `str(exc)`).
2. Apply it across all 27 `logger.warning("...: %s", exc)` patterns in `src/scix/`,
   not only this wave's helpers — the concern is cross-cutting.
3. Supersede this ADR with one recording the aggregator addition and the chosen
   sanitization approach.

## Alternatives Considered

1. **Build the `logging.Filter` now.** Rejected as YAGNI. The exfiltration path
   requires a remote log sink that does not exist and is not on the roadmap;
   meanwhile every WARNING record would pay regex cost and the codebase would carry
   a sanitization layer with no consumer.

2. **Drop the `%s`/`exc` rendering (log a generic message).** Rejected. The full
   exception text is the primary debugging signal for these guarded DB calls in a
   local-only deployment; destroying it to defend against a non-existent sink trades
   real diagnostic value for theoretical safety.

## Consequences

Positive:

- No new code, no per-record regex overhead, no maintenance surface.
- Full exception detail remains available for local debugging.
- The decision and its precise revisit trigger are documented, so the forward-looking
  policy commitment survives beyond the originating bead's notes.

Negative / accepted:

- If an operator wires up a third-party log aggregator **without** consulting this
  ADR first, schema names could flow to it until the `logging.Filter` is added. The
  mitigation is this ADR's explicit revisit trigger plus the local-only deployment
  convention in `CLAUDE.md`.
