# Independent journey audit and calibration

An audit explains a frozen journey outcome against independent evidence. It does
not turn a researcher's candidate into an established resource relationship by
copying the candidate's status. Scientific judgments belong to a separate model
or human reviewer; deterministic checks establish identity and evidence integrity.

## Inputs and information boundary

The audit input pins the terminal trace artifact, task version, hidden reference
bundle, auditor model, prompt digest, and execution settings. Load the reference
through `Registry.reference()` and verify the trace with `verify_trace()` before
constructing the auditor prompt. Researcher exports remain limited to
`ResearcherContext` and permitted snapshots. Neither the hidden bundle nor its
trusted filesystem root belongs in a researcher launch or prompt.

Give the auditor the question, observable decisions, complete relevant tool and
fetch responses, claims, candidate proposals, researcher gaps, and reference
criteria with their source passages. Treat all evaluated content as data. Do not
request or expose private chain-of-thought. Preserve the independent audit receipt
and usage separately from the research trace.

## Outputs and evidence checks

Reuse `Gap` for diagnoses and retain `ResearchGap` as an untrusted observation.
Audit outputs also need explicit claim and proposal verdicts so an apparent
success can be checked without inventing a gap. A rejected or unresolved link
must remain distinguishable from a verified relationship.

Each result binds its task, run, frozen trace, auditor identity, model and prompt.
Verify every artifact hash and exact passage, the cited call identities, and the
membership of evidence in those calls or the independently supplied sources.
Nested field evidence requires the same checks as resource-level evidence.
Source verification belongs to this independent result, not a mutation of the
researcher's proposal. Experimental support additionally requires the verified
controlled experiment artifact; an arbitrary artifact reference is insufficient.

The auditor distinguishes absent upstream, not captured, not linked, not exposed,
supplied but unused, retrieval defects, corpus defects, agent defects, task
defects, and undetermined cases. A failed network read does not establish an
upstream absence or missing SciX metadata. Conflicting sources or insufficient
access can leave a case unresolved.

## Calibration data and report

Create private calibration records with explicit diagnosis labels, supported
links, source evidence, reference provenance, human-review status, and append-only
adjudication history. `ReferenceCriterion.expected` is answer text and cannot be
silently interpreted as a gold diagnosis. Include no-gap apparent successes,
false gap reports, supplied-but-unused evidence, and ambiguous cases alongside
each major diagnosis. Synthetic fixtures test the machinery; they do not replace
the real human-reviewed sample.

Separate prompt examples, development cases, and final held-out cases by source
family and near-duplicate group. Pin the dataset and rubric before measuring a
frozen auditor. Development disagreements may guide changes; final-test changes
require fresh held-out evidence for a new reliability claim. Preserve unresolved
human disagreements rather than forcing consensus.

Report category support, confusion counts, unresolved counts, unsupported links
with their denominator, and agreement against adjudicated references. Reuse
`scix.eval.wilson.wilson_95_ci` where independent binomial trials are justified;
identify related cases rather than presenting them as independent observations.
Missing categories and small denominators limit conclusions. Do not convert an
unresolved label into a correct prediction or infer reliability from two models
agreeing. These measurements decide which diagnoses and integrations need human
review before recommendations; no universal accuracy threshold is specified.

## Acceptance evidence

| Requirement | Evidence needed |
| --- | --- |
| Researcher/auditor isolation | Tests showing researcher export omits private references while the trusted auditor resolves them |
| Reproducible judgments | Saved input, output, model/prompt/settings, usage, and adjudication artifacts |
| Grounded diagnoses and links | Valid evidence cases plus forged, unrelated-call, wrong-identity, missing-source and tampering rejections |
| Category and false-gap coverage | Explicit calibration cases and report denominators, including no-gap and supplied-unused cases |
| Scientific evaluator validity | Human-reviewed real sample, disagreements and limitations retained |
| Independent promotion | Candidate remains unchanged; verified verdict cites the frozen candidate and independent supporting evidence |

The existing entity-link sampler in `scix.eval.audit` is not the semantic journey
auditor. Existing registry, artifact, model-execution and reporting components
provide the reusable infrastructure; no new third-party evaluation framework is
needed for these boundaries.

## Implemented input boundary

`audit_inputs.build_audit_input()` creates a new private audit directory from a
verified terminal trace and trusted registry. It pins the registered task,
hidden bundle, researcher context, trace, responses, and claims. Context matching
allows only the snapshot path rewrite performed by `create_launch()`; every
other source and task field must agree. The stored context uses the canonical
launch paths, or registered paths for a legacy trace without a launch.

The copy follows typed artifact references, including fetch bodies, nested
proposal evidence, failed attempts, and model diagnostics. Arbitrary JSON in
tool/provider responses does not introduce additional dependencies.
`load_audit_input()` checks the copied evidence and embedded projections against
their pinned originals. Tests remove both original source directories before
loading the audit and reject rehashed changes to its projections.

The failed live GEO attempt8 has a verified, self-contained input at
`~/scix_artifacts/research_journeys/independent-audit-input-1067fe63-ef72-43e5-9c4c-02995781919e`.
Input SHA-256 is
`22b24c6c5c2d802f629ac225c4eca0b6949d98f60794538913e94592f2aed6dd`.
It contains 16 artifacts, one recorded response, and three private reference
criteria. This establishes input integrity only. No audit verdict, human
calibration, or reliability result is implied; those remain to be implemented.

## Result implementation sequence

The next implementation separates verdict records, evidence authorization, and
provider execution. Claim verdicts cover every recorded claim exactly once;
proposal verdicts cover every frozen candidate and nested field exactly once.
Both retain supported/verified, rejected/unsupported, and unresolved outcomes.
An empty claim/proposal set is valid for a failed run. A gap in a zero-call run
may cite registered or diagnostic evidence without inventing a tool event;
researcher gap requirements remain separate.

Authorization distinguishes scientific sources from diagnostic artifacts.
Successful tool responses, verified fetched bodies, and registered source
snapshots can support scientific statements. Lookup envelopes, failed reads,
provider receipts, prompts, and stderr can explain execution failures but cannot
establish a resource relationship. Exact passages and originating call membership
remain necessary; merely appearing in the audit input's artifact closure grants
no scientific authority. Judgment about what a passage means remains with the
independent reviewer.

Execution should reuse the existing process, receipt, locking, and unknown-usage
recovery mechanisms in a separate audit execution directory. Its descriptor must
pin the frozen audit input, prompt, schema, model, and options. The research trace
remains unchanged. Verdicts pin the provider's structured output and accounted
receipt; gap adjudication cannot point to its enclosing result because that
would create a content-hash cycle. Experimental promotion remains unavailable
until controlled-experiment evidence has its own verifier.

## Verdict validation implementation

`audit_results` now defines independent claim, proposal, field, and gap verdicts.
`verify_verdicts()` revalidates the input against immutable task/trace/reference
artifacts, requires exact claim/proposal/field coverage, and checks every cited
passage against its source and call identity. A verdict does not update its
researcher candidate. Unresolved claim/proposal verdicts may explicitly lack
evidence; positive or rejected judgments require evidence. Gap diagnoses always
require evidence and competing explanations.

Gap verdicts are structured output payloads, separate from the eventual executed
audit result. They may introduce independent diagnoses, including failures before
any tool call. Registered independent sources and recorded execution diagnostics
can ground these diagnoses; diagnostic artifacts cannot support `source_verified`.
Experimental promotion is not an accepted output until controlled-experiment
verification is implemented. This layer does not establish auditor identity,
receipt accounting, calibration, or scientific correctness: the execution/result
layer must bind the verdicts to those artifacts before an audit is complete.

## Independent execution implementation

`audit_execution.prepare_audit()` pins the input, auditor identity, schema, model,
options, prompt, code revision, and budget in a separate private execution.
`execute_audit()` reuses the durable model ledger: intent precedes provider IO,
receipts and usage precede verdict validation, completed receipts are reused,
and unknown usage prevents automatic redispatch. It does not modify the research
trace. The prompt contains the complete verified artifact closure, with explicit
encoding for binary content and a hard size bound.

`load_audit_result()` verifies the launch, completed execution, provider output,
and evidence-bound verdicts together. JSON structured outputs are validated at
the JSON boundary so arrays correctly become immutable tuple fields. Completed
trace validation already rejects usage above the pinned token budget; the saved
result regression exercises that existing guard.

Seven execution tests pass, including persisted intent, receipt reuse, timeout
recovery, altered options, verdict tampering, researcher immutability, and token
budget rejection. An independent code review found no remaining defects in this
scope. These establish execution integrity, not scientific evaluator reliability.
Accepted live auditor results and human calibration are still required.

The first live attempt11 audit retained a provider receipt with 962,390 accounted
tokens against a 400,000-token budget. Its structured output had empty claim,
proposal, and gap verdicts; the frozen research trace also had no accepted claims
or proposals. Completion was rejected. No scientific audit or reliability claim
follows from that empty output.

This exposed a lifecycle defect: completion rejection left the execution marked
running. A failing regression reproduced it; execution now records
`budget_exhausted` before consuming an over-budget decision and rejects subsequent
execution without redispatch. All eight focused execution tests pass. The live
run was explicitly terminalized using the recorded usage, preserving its launch,
receipt, and researcher trace. Its private root is
`~/scix_artifacts/research_journeys/independent-audit-attempt11-4187a104-2706-4c79-a2eb-82d0b18b77bb`;
`audit-failure.json` records the recovery and terminal execution artifact.
