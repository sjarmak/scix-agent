# Research journey contracts, version 1

This is the boundary design for `scix_experiments-nmnd`, not a claim that the
runner, live experiment, or cohort is finished. Schemas live in
`src/scix/journeys/`; small synthetic examples live in `examples/`. Their hashes
are illustrative, not live evidence. All semantic decisions are made by a
researcher, independent auditor, or human; validation checks structure only.

## Execution design and reuse

Use a dedicated journey workflow, separate from `LiteratureReviewWorkflow`.
A task-scoped MCP subprocess lasts for one run. Baseline, treatment, repeat,
and held-out runs receive distinct subprocesses and working sets. The
researcher receives `Task.researcher_context()` only. The auditor receives the
frozen trace plus hidden verifier/reference bundle, independently of the
researcher and task curator. Auditor outputs retain model/prompt identity and
adjudication provenance. Neither researcher nor treatment overlay can read
hidden references. Public artifact alias checks supplement, but cannot replace,
review of curator-supplied text for answer leakage.

The researcher's finish decision carries `ResearchGap` observations, each with
unresolved status, exact recorded evidence, affected conclusion, and competing
explanations. It cannot supply auditor identity or adjudication. The controller
exports these as hashed trace gap artifacts and checks task/run identity and
call-to-passage provenance. The separate `Gap` contract remains the independent
auditor's diagnosis, with its own model, prompt, and adjudication evidence.

Reuse audit, 2026-09-23:

| Component | Decision |
| --- | --- |
| `src/scix/session.py::SessionState` | Existing working-set semantics; process isolation per run, never a shared default session. Restore state explicitly after worker loss. |
| `src/scix/eval/metrics.py` | Reuse callable-injected evaluation pattern for run-local response overlays. |
| `src/scix/eval/audit.py`, `wilson.py` | Reuse sampling and binomial intervals where denominators support them; no semantic score thresholds. |
| `src/scix/mcp_runtime.py::_log_query`, `_emit_trace_event` | Secondary diagnostics only: best-effort, incomplete payloads, no task identity. |
| `durable_research/artifacts.py::ArtifactStore` | Adapt content-addressed storage and containment pattern; add read-time hash verification and exclusive publication. Existing check-then-`os.replace` is not concurrent write-once protection. |
| `durable_research/external_calls.py::journaled_call_tool` | Adapt stable logical request IDs and response journal; retain separate attempt records and structured errors. |
| `durable_research/mcp_client.py::call_tool` | Reuse MCP SDK session/stdio pattern, not the per-call subprocess lifetime: the journey needs its working set across calls. |
| `durable_research/workflow.py`, `agent_lifecycle.py` | Reuse deterministic workflow/activity separation and reconnect-by-ID lifecycle, not the literature-review input schema or research policy. |

The durable research source checkout is
`~/temporal_devrel/presentation/temporal-literature-review`.
It is not an installed SciX dependency; port only the bounded mechanisms with
attribution and tests instead of importing an operator-specific absolute path.
`run-durable-research/scripts/doctor` found Temporal at `localhost:7233`, but
neither workflow nor activity worker pollers on `temporal-literature-review`.
No worker was started during this audit. A dedicated journey worker and queue
must be configured before live durable execution.

Research before implementation included GitHub searches for the TB-Science
repository, Temporal artifact-store examples and Pydantic frozen models. The
project already has Pydantic via MCP; declare it explicitly in the `journeys`
extra for standalone contract use. The package registry metadata for installed
Pydantic 2.13.5 and [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/)
support strict/frozen models and schema generation. The local graphify query
returned no matches, so reuse discovery used direct source inspection.

## Identity and evidence

- `task_id` + `task_version`: immutable question/adaptation identity. A change in
  question, permitted inputs, or references requires a new version.
- `run_id`: one arm/repetition; distinct from Temporal execution IDs. Persist
  workflow and execution IDs in the runtime journal when Temporal is used.
- `session_id`: one MCP process incarnation. A recovered run records subsequent
  incarnations in its runtime journal; never claim the original process lived.
- `call_id`: stable logical tool call; `sequence` is contiguous from zero and
  optional `parent_call_id` must refer to an earlier call.
- `attempt_id`: unique transport attempt; numbers contiguous from one. Retries
  retain errors/interruption instead of overwriting an attempt.
- Each artifact reference contains a relative canonical path, byte size,
  media type and SHA-256. Every read must verify bytes, length, and digest.
- A passage uses Unicode character offsets into a persisted UTF-8 text artifact;
  its exact quote must match that slice. Raw binary/PDF responses must first have
  a separately hashed text extraction with a recorded derivation relation.
- `Source` pins locator, version, snapshot, retrieval time, rights and permitted
  use. Access is checked separately from assertion confidence.
- A resource's exact-product, collection, derived-product, software,
  metadata-standard or merely-related type is explicit. `explicit` versus
  `inferred` is not collapsed into a confidence score. Schema field proposals
  name paths, meanings, coverage, update/version semantics and acquisition.
- `Gap` keeps alternative explanations and retrieval/corpus/agent/task defects
  separate from metadata diagnoses. `experimentally_supported` requires an
  experiment reference; existence of that reference is not scientific proof.
- An experiment shares pinned model/prompt/tool/corpus/index/code/budgets across
  arms. It requires unique runs and exactly one raw outcome per arm/repetition.
  Null and negative results are valid. Link counts cannot mechanically determine
  semantic conclusions; judgments need calibrated independent evidence review.

Boundary input uses `Model.model_validate_json(text)`; strict Python construction
requires tuples and typed nested objects, not JSON lists. Models are frozen and
contain immutable members. Do not use unvalidated `model_construct` or
`model_copy(update=...)` for incoming data.

## Artifact layout and recovery

Use an operator-configured directory outside the git checkout on local NVMe,
e.g. `~/scix_artifacts/research_journeys/`. Separate access roots:

```
registry/public/<task>/<version>/
registry/private/<task>/<version>/
runs/<run>/objects/<sha256>
runs/<run>/journal.sqlite
experiments/<experiment>/objects/<sha256>
```

Researcher processes must not receive the private root or database credentials.
The runtime persists request intent before dispatch, attempt start before IO,
then immutable full response/error artifacts before committing completion.
SQLite transactions serialize logical effects and preserve attempt history;
Temporal histories carry compact refs, never response bodies. Recovery attaches
to existing IDs and checks artifact integrity before reusing a response.

A crash between external response and local commit can repeat a read-only call,
and upstream results may change. Preserve the interrupted attempt and record the
new snapshot; never claim exactly-once external computation. Rebuild a lost MCP
working set from recorded state transitions, not fresh semantic reasoning. Paid
or mutating calls require provider idempotency before admission. Only read-only
research tools belong in this evaluation runner.

## Access, budgets, and evaluation boundaries

No production writes, bulk ingestion, embedding/index architecture changes, or
new MCP tools. The 15-tool cap remains unchanged. SciX telemetry can write even
for read-only tools: a live client must explicitly disable telemetry or redirect
it to `scix_test`, never silently emit production query-log rows.
`SCIX_TELEMETRY=disabled` now skips both query-log and trace-publication hooks
in `mcp_server.call_tool`; default `enabled` preserves ordinary telemetry, and
unknown values fail before database access. This only controls telemetry: the
runner still needs a read-only tool policy and database protections for handler
side effects. Full-text AI
requires OA/preprint evidence; metadata and abstracts have separate permitted-use
records. Public upstream access is not a blanket redistribution license.

Each run pins maximum logical calls, attempts per call, wall seconds, response
bytes and model tokens. Enforce these at IO/model boundaries and report
`budget_exhausted` with preserved evidence. Heavy processes use `scix-batch`.
Fetching must reject unsafe hosts/redirects, enforce response bounds and respect
source policies. These are runner responsibilities, not claims made by the
contract validators.

The registry split uses source/paper families, not random rows. References and
verifiers are private. Relatedness and near-duplicate judgments belong to a
curator, with explicit group assignments mechanically checked for split overlap.
The pilot proposes integrations only after independent source/span verification.
Human calibration is required before claims about evaluator reliability; human
comprehension claims additionally require sampled human assessment. Neither
fixture tests nor multiple models agreeing constitute that evidence.

## Verification

Implemented contract checks:

```bash
uv sync --extra dev --extra journeys
SCIX_TEST_DSN=dbname=scix_test .venv/bin/pytest -q tests/test_journey_contracts.py tests/test_journey_trace_contract.py
.venv/bin/ruff check src/scix/journeys tests/test_journey_contracts.py tests/test_journey_trace_contract.py
.venv/bin/black --check src/scix/journeys tests/test_journey_contracts.py tests/test_journey_trace_contract.py
```

Full vertical-slice acceptance additionally requires bounded live task execution,
source/span/hash verification, task isolation, kill/resume, equivalent replay
arms, independent held-out evaluation and a human calibration sample. Runnable
commands for those will be recorded with the runner CLI and pinned task registry;
these contract tests alone do not satisfy them.

Standalone validation and schema export (no database or worker):

```bash
.venv/bin/python -m scix.journeys validate task docs/research_journeys/examples/task.json
.venv/bin/python -m scix.journeys schema trace > /tmp/scix-journey-trace.schema.json
# Expected exit 1: unsupported experimental promotion, missing experiment artifact.
.venv/bin/python -m scix.journeys validate gap docs/research_journeys/examples/invalid_gap.json
```

A successful validation confirms structure only, not artifact existence,
scientific correctness, independent review or experiment efficacy.

The source adapters in `scix.journeys.adapters` expose pinned expert material
to a curator. `adapt_terminal_bench_task` verifies and parses instruction/TOML
snapshots; `adapt_scholarqa_record` validates bounded JSONL and selects an exact
record ID. Both preserve source rights, versions, hashes and additional metadata.
Neither adapter invents gold answers or assigns semantic task families. The
curator supplies those decisions in the registry and private reference bundles.

Registry validation additionally checks artifact bytes, SHA-256 digests, exact
reference passages, reference identities, and curated split keys. Export creates
a fresh directory containing only the researcher context and permitted inputs:

```bash
.venv/bin/python -m scix.journeys registry-check /absolute/path/registry.json --artifact-root /absolute/path/trusted
.venv/bin/python -m scix.journeys export-researcher /absolute/path/registry.json --artifact-root /absolute/path/trusted --task-id TASK --task-version 1 --output /absolute/path/new-researcher-inputs
```

The trusted root contains private curation and reference artifacts. Public inputs
cannot alias these artifacts or designated private reference evidence by path or
hash, including references belonging to other tasks. Export reserves
`context.json` and rejects duplicate input paths. This is a file projection;
the runner must separately deny researchers access to the trusted root. These
checks do not establish semantic independence or scientific correctness, and do
not prohibit independent retrieval of the same public scientific facts.

Persistence and MCP-boundary behavior, including the implemented `trace-check`
command and synthetic kill/recovery evidence, are documented in
[runner_persistence.md](runner_persistence.md). Full run orchestration remains
in progress.

## Locked vertical-slice verification interface

The following acceptance interface is reserved for children .3–.7. These
commands are specifications for the runner implementation, **not available
subcommands today**. Implement them without expanding the MCP tool surface.
Task and experiment requests pin IDs, source snapshots and numeric budgets;
commands do not choose scientific questions or infer success from exit codes.

```bash
# Durable isolated run; emits workflow/run IDs and artifact manifest location.
scix-batch python -m scix.journeys run /absolute/path/pilot-request.json
# Resume requires the original identity, never a freshly generated run.
python -m scix.journeys status RUN_ID
scix-batch python -m scix.journeys resume RUN_ID
# Verify all bytes/hashes/spans and evidence edges, including retained errors.
python -m scix.journeys verify /absolute/path/run-manifest.json
# Independent auditor request includes private references; researcher does not.
scix-batch python -m scix.journeys audit /absolute/path/audit-request.json
# Run-scoped overlay, independent arms/repetitions, same settings and budgets.
scix-batch python -m scix.journeys replay /absolute/path/experiment-request.json
# Reuse frozen proposal and settings on independent source-family task.
scix-batch python -m scix.journeys run /absolute/path/heldout-request.json
python -m scix.journeys report /absolute/path/experiment-manifest.json
```

The first acceptance run must retain positive, negative, failed and unresolved
outcomes as observed, include a no-effect fixture, and demonstrate kill/resume
with preserved attempt history. Successful commands and a human-readable report
are insufficient unless every evidence link resolves and the independent
held-out and calibration results are present.
