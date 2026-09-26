# Controlled provenance replay

`Experiment` currently validates manifest shape and repetition coverage. It does
not run comparisons or prove their equivalence. No controlled scientific result
is accepted yet.

The standalone `overlay_records.py` projection is implemented. Twenty-one
focused cases pass with 90% module coverage, including run/request mismatches,
immutable link values, field collisions, malformed/error responses, strict JSON,
and input/output byte limits. Review caught an unbounded unmatched-response
path; it now checks the byte limit before returning unchanged text. This module
does not establish independent scientific authorization and is not installed in
the server. Binding/export verification and controlled execution remain open.

## Pilot intervention

Use the existing `get_paper` tool. Its handler returns JSON text containing
`papers`, `total`, and `timing_ms`; `result_schema_for_tool("get_paper")` permits
additive fields. A treatment can add `papers[0].resource_links` to the one exact
paper named by a pinned request. Preserve every existing paper field, the paper
list, counts, timing, and errors. An existing field collision must fail rather
than overwrite production metadata.

The proposed link must first have an accepted independent audit. The audit must
support its relation to the target paper, not merely the existence of a GEO
accession. A supplied bibcode or identifier string is not proof of that relation.
The model or trusted reviewer supplies that semantic mapping and classifies the
contrast as navigation-only or new evidence. Python checks the frozen identities,
source passages, and accepted verdicts without inferring semantic equivalence.

Current `ProposalVerdict` has only a proposal ID, evidence, sources, and field
verdicts. It does not identify the target paper/request or the exact injected
value. Introduce an independently judged binding with proposal ID, target
`get_paper` request, restricted destination field, exact public value, verdict,
evidence, and rationale. Include that binding in the frozen independent judgment;
do not derive it from an identifier substring. A verified resource-existence
verdict alone is insufficient.

The auditor may consult hidden reference sources, but export authorization is
separate. A treatment binding must be supported by researcher-public inputs or
successful recorded public tool/fetch evidence. Hidden reference snapshots,
private rationales, and diagnostic receipts cannot become treatment payloads.
Export only the exact approved public value and provenance projection, not the
entire audit result. Scientific review and public-source authorization are both
required; passing one does not imply the other.

Bind the intervention to tool name/version, exact canonical argument digest,
target bibcode, permitted field, public payload artifact, audited proposal, and
experiment identity. Do not implement unrestricted JSON Patch or a replacement
response. A reuse search found OpenStack Ironic's `jsonpatch.apply_patch`; neither
jsonpatch nor jsonpointer is a project dependency, and general patch operations
are unnecessary for this bounded additive intervention.

## Process boundary

`install_checkpoint_handler(server, state, dispatch)` calls a synchronous
`dispatch(name, arguments) -> str`, then builds MCP text content and the session
checkpoint. Install an experiment-only wrapper around that dispatch in the
journey entrypoint. It obtains the ordinary response before applying the patch.
It has no session mutation API and cannot modify `_meta` or `isError`. The
checkpoint handler then captures the ordinary post-handler state.

Baseline launches receive no overlay. Treatment launches receive only the public
intervention projection and its run binding. The private audit root, reference
bundle, verdict rationale, and human labels stay outside both researcher
processes. Copying every audit artifact into a treatment root would violate that
boundary even if the model prompt omitted them.

Persist original response, visible response, and an application record binding
both hashes to request, run, call/attempt, field, and intervention before the
visible response is committed. Recovery verifies the same application rather
than applying a second patch. A nonmatching request returns original bytes;
malformed matched responses and provenance errors fail explicitly.

## Equivalence and outcomes

Use distinct roots, subprocesses, sessions, journals, and model invocations for
every arm and repetition. Pin equal model, prompt, tool contract, corpus/index,
code, and budgets. The current scientific smoke settings explicitly identify
live, unfrozen corpus/index state: reusing those labels cannot establish a
controlled comparison. Use a reproducible corpus snapshot or recorded base
response replay with explicitly restored session effects, and retain the exact
base payloads. Verify equivalence rather than relying on shared setting names.

A verifier must load every raw trace and independent adjudication, bind its
run/task identity to the outcome, compare settings, recompute calls and accounted
tokens, and verify treatment applications. Semantic correctness, missed links,
explanation fidelity, contrast type, and conclusions come from the calibrated
auditor or trusted reviewer. Never derive them from fewer calls or a completed
execution status. Retain failed, unresolved, negative, and no-effect runs.

## Acceptance tests

- A matched request adds only the declared public link field and preserves the
  original response and ordinary checkpoint.
- A different bibcode, argument digest, run, or tool cannot apply the overlay;
  the nonmatching response remains byte-identical.
- Two real journey clients cannot observe each other's working sets or overlays.
- Hidden-reference artifacts, unsupported paper mappings, rejected proposals,
  changed evidence bytes, and pre-existing destination fields are rejected.
- Crash/reopen retains base/visible/application identities without double apply.
- Every repetition has one baseline and treatment with equivalent settings and
  verified base data; forged outcome IDs, costs, and provenance fail validation.
- A no-effect fixture and a negative semantic outcome remain in the reproducible
  experiment manifest, with their raw evidence and uncertainty.

These fixtures establish machinery. A real accepted proposal, calibrated audit,
controlled scientific contrast, and independently held-out task remain required
before the parent epic's recommendation can be supported.
