# Durable feedback for rejected researcher decisions

The GEO attempt12 stopped before its first fetch. Its first completed provider
receipt used 62,278 tokens and proposed a citation whose offsets did not match
the quote length. `validate_decision()` correctly rejected it. The runner had no
recorded correction path, so starting another run would merely repeat the same
failure opportunity. The original receipt must remain unchanged.

## Execution boundary

Record an `ActionRejection` separately from dispatched `Call` records. A rejected
decision is not a tool response, consumes no call budget, changes no working set,
and cannot ground a scientific claim. Model usage and wall time still count.
Keep the existing provider-failure and unknown-usage recovery behavior.

Each rejection binds a completed model invocation and its immutable provider
receipt, the submitted decision, and bounded deterministic validation details.
Malformed typed actions must be representable without first constructing the
invalid `Decision`. Do not repair offsets, select a new source, or choose a
replacement action in Python. The model receives the rejection and decides how
to correct its request.

Persist the rejection in the journal before requesting another model decision.
The trace verifier checks its invocation/receipt identity and reproduces the
supported deterministic validation failure against the same public evidence
available at that point. Artifact-integrity, configuration, persistence, provider,
and transport failures must propagate rather than becoming correction feedback.
Only explicitly typed pre-dispatch input failures qualify; never catch every
`ValueError` around execution.

## Controller and evidence changes

Model invocation order and actual call order become separate counters. During
recovery, skip decisions with verified rejection records and match the remaining
decisions to actual calls in order. A completed rejected decision is never
redispatched. Observations still refer to the preceding actual call, even when
one or more rejected decisions intervene. Retain existing call IDs and parent
relationships; do not reinterpret prior traces.

The researcher prompt includes prior submitted decisions and structured
validation feedback. Rejections are excluded from successful discovery IDs,
response authorization, checkpoint recovery, and resource/link/field evidence.
The independent auditor may use the retained provider receipt as diagnostic
evidence under its existing rules, but never as scientific source evidence.

## Implementation sequence and acceptance oracle

1. Add typed rejection records and deterministic validators. Test malformed
   citation spans and well-formed spans that mismatch their source, plus valid
   requests, altered receipts, altered details, and unavailable sources.
2. Add journal persistence and trace verification. Crash after rejection commit
   must reopen with the same receipt and rejection; no external call may exist.
3. Adapt controller indexing and prompt feedback. A scripted provider first emits
   the invalid request, then a corrected locator/fetch sequence. Assert the
   fetcher was never called for the invalid request, feedback reaches decision 2,
   and the correction produces exactly the intended actual calls.
4. Test recovery without redispatch, model-token and wall exhaustion during
   corrections, observations across rejection boundaries, and rejection evidence
   exclusion. A transport error after a valid dispatch remains an ordinary call.
5. Review independently and run the journey suite. Then run a fresh scientific
   journey with pinned source, prompt, model, and budget. Do not modify attempt12
   or infer scientific success from structural fixtures.

## Structural rejection implementation

The first slice is implemented. `rejection_records.ActionRejection` binds a
completed successful invocation, its receipt, the run's tool inventory, a fixed
error code, and a bounded validation message. Verification reproduces the
structural `validate_decision()` failure. Malformed tool configuration and failed
or unknown provider outcomes propagate instead of becoming correction feedback.

`RunJournal.record_rejection()` commits the record before more model IO. Trace
validation requires unique, ordered invocation references. The controller skips
rejected decisions on recovery and numbers actual calls independently. The
researcher prompt includes the rejected submission and mechanical feedback;
rejection payload bytes count toward its size bound.

Five controller regressions pass: feedback before the next dispatch, reopening
after a crash immediately following rejection commit, corrected call numbering
and call-budget accounting, model-token exhaustion, and observation identity
across a rejected decision. Seven record-level tests pass, including receipt,
inventory, message, provider-status, and configuration checks. Independent review
passed 32 focused tests and found no defect. Additional tests now explicitly
reject reassignment to a different completed invocation and carry an invalid
fetch decision through correction to a fetched source, evidenced claim, and
reopen without external redispatch; all eight controller/fetch cases pass.
The full journey coverage gate passed: 532 tests and 90.98% coverage
(`test_journey_*.py`, session 44463). The two additional integration/identity
cases added after collection passed in the separate eight-case focused run.

## Source-content correction

`PassageContentMismatch` now distinguishes a candidate quote mismatch from
corrupt, missing, or non-UTF-8 source bytes. The controller catches only that
typed mismatch during the current fetch request's basis preflight, before
dispatch. Existing trace verification and fetch execution remain outside the
catch. Observations and final claims are not covered by this correction slice.

The rejection pins its successful receipt, tool inventory, exact offending
basis passage, and prior source/call prefix. Verification parses the original
decision, requires a fetch action, and reproduces `verify_basis()` on that
request. An arbitrary mismatched passage cannot be attached to a finish receipt.
Authorized artifact refs are canonicalized because execution and replay list
fetch body/response refs in different orders; prior call identities and attempt
statuses remain ordered and bound separately.

Four integration cases pass in `test_journey_source_rejection_controller.py`:
correction through an actual fetch, crash after feedback commit and reopen,
forged finish-receipt reassignment, and correction after an earlier fetch.
Reopening completed runs makes no external calls. Review found and corrected
the initial receipt-binding omission and overly broad catch; additional
adversarial review added five passing tests for an alternate valid tool inventory,
same-artifact quote substitution, changed prior-call metadata, source-byte
tampering, and corrupted prior inspected passages. The combined record,
controller, and integrity suite passed 19 cases. The updated full-suite gate
passed: 546 tests, 88.70% total coverage (session 51421). A new overlay module
appeared after collection and was counted as uncovered; this gate does not
validate that separate module.

Fresh scientific attempt13 completed with launch
`e64c50641cc21d758ad7a8dfd0609a7f8fb3f5a3cfba97c199d4ee83f9a6a260`.
Its first decision failed structural validation; the persisted feedback led to
six successful calls (five passage lookups and one fetch), eight model
invocations, and 782,697 accounted model tokens. The terminal trace is
`6ca61a9a180a830cadec34ee4666b7575d8685203dcf0d5e0a8e7c114394844d`
(19,258 bytes), under
`~/scix_artifacts/research_journeys/scientific_smoke_20260924/attempt13/geo-scientific-smoke-a5aa5cb5-1040-4e91-bd5f-621549785b40`.
`inspect_run()` revalidated the completed journal and its evidence after exit.

The submission contains three claims (series identity, taxon, and clinical
descriptors in sample titles), one candidate resource proposal, and one
unresolved gap. Its fetched GEO esummary body is pinned as
`d78e15440207e265d131b3373fac3e6e56131561869c21f791ecc09ee5dd8edf`.
No SciX retrieval call was made, and SOFT/MINiML sample characteristics were not
fetched. This execution therefore cannot establish upstream absence or a SciX
capture/exposure defect. Independent semantic audit remains pending; completed
execution and mechanically verified passages do not establish scientific support.

The correction mechanism does not replace
the independent audit, human calibration, controlled experiment, held-out task,
or cohort requirements of the parent epic.
