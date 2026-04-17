# ADR-006: arXiv LaTeX Licensing — Internal Use Only

- **Status**: Accepted (2026-04-13)
- **Deciders**: SciX Experiments maintainers
- **Scope**: All code paths that ingest, index, embed, or emit arXiv LaTeX-derived text
- **Supersedes**: none
- **Related**: `docs/prd/prd_external_data_integration.md`, PRD Build 5 (bead `scix_experiments-wqr`)

## Context

PRD Build 5 introduces an external-source ingestion pipeline that pulls full text from arXiv LaTeX via the `ar5iv` HTML path, a local `ar5ivist` LaTeXML fallback, and — for bulk backfill — the arXiv S3 `src/` mirror. LaTeX-derived text is the cleanest structured full-text source for the astronomy-heavy subset of the corpus (~30–40% of ADS papers, higher among modern citation-rich subsets).

arXiv does not hold copyright on the papers it distributes. Authors grant arXiv a **non-exclusive, perpetual distribution license** under the [arXiv Submission Agreement](https://arxiv.org/licenses/assumed-1991-2003/license.html). Individual papers may additionally carry author-selected licenses (CC-BY, CC-BY-NC, CC0, arXiv nonexclusive-distrib, etc.), but the baseline grant to downstream consumers is only the right to read and link back — **not** the right to redistribute verbatim text or derivatives.

This has concrete implications for SciX:

- Internal retrieval, ranking, embedding, and analytics over LaTeX-derived text are permitted under fair use for research purposes.
- Any user-facing artifact — MCP tool response, API payload, notebook figure, exported dataset — that surfaces verbatim LaTeX-derived text is redistribution and must be constrained.
- Abstracts are **not** affected by this ADR. Abstracts are sourced from ADS metadata, which carries its own (more permissive) terms. This ADR governs the **body text** ingested through the arXiv LaTeX path, not ADS-provided fields.

We also considered two alternatives:

1. **Skip arXiv LaTeX entirely** — rely on ADS body + OpenAlex PDF URLs + Docling. Loses the structured-text pipeline that is the entire point of PRD Build 5's Tier 1.2. Rejected.
2. **Request redistribution rights from arXiv** — expensive, slow, outcome uncertain, blocks work unit W4 (`scix_experiments-wqr.5`). Deferred; not a near-term path.

## Decision

**Ingest, parse, index, and embed arXiv LaTeX full text freely for internal use. Cap every user-facing emission of verbatim LaTeX-derived text at a configurable snippet budget (default 500 characters), and always attach `canonical_url=https://arxiv.org/abs/{arxiv_id}` to the response so users can read the paper at the source.**

Specifically:

| Operation                                                                    | Policy                                                                |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Store parsed LaTeX body in `papers_fulltext` (or equivalent)                 | Allowed, table tagged `internal_use_only=true`                        |
| Compute embeddings, BM25 indexes, entity extractions over body               | Allowed, internal use                                                 |
| Graph analytics, retrieval eval, community detection                         | Allowed, internal use                                                 |
| Return verbatim body text in an MCP tool response                            | **Must** wrap through `scix.sources.licensing.enforce_snippet_budget` |
| Return structured metadata (section titles, figure captions, citation spans) | Allowed, provided the emission is not a disguised full-body dump      |
| Export a dataset or dump that includes LaTeX-derived body text               | **Forbidden** without explicit per-paper license verification         |

### Snippet budget rationale

- A snippet is a **retrieval hit context window**, not an abstract. Abstracts come from ADS metadata and are emitted separately (and without a budget).
- 500 characters is approximately 2–3 sentences of English prose — enough to show the matched passage in context, not enough to reconstruct the paper.
- The budget is configurable via the `SCIX_LATEX_SNIPPET_BUDGET` environment variable for ops overrides. Individual MCP tools may request a **lower** budget but must document why; any tool requesting a higher-than-default budget requires an ADR update.
- Truncation is terminal: text beyond the budget is dropped and an ellipsis (`...`) is appended. The total emitted length stays within the budget.

### Canonical URL enforcement

Every response that includes LaTeX-derived snippet text must carry `canonical_url`. The helper treats `canonical_url` as a required, non-empty argument and raises on violation. This is a hard contract, not a convention — callers cannot omit it.

## Consequences

### Positive

- Unlocks the structured-text pipeline for PRD Build 5 without legal ambiguity.
- Encodes the licensing invariant as a reusable, testable helper that downstream ingest code imports, rather than scattering bespoke truncation logic.
- Preserves the reader's path back to the authoritative source (arXiv) on every LaTeX-derived emission.

### Negative / constraints

- Some retrieval tools will occasionally surface obviously-truncated context. This is the cost of the license posture.
- The policy adopts the **worst-case floor** across per-paper licenses. Papers released under CC-BY or CC0 could legally be redistributed verbatim, but SciX does not branch on per-paper licenses at this time. Relaxing for permissive-license papers is future work and requires ingesting license metadata into `papers_fulltext`.
- Enforcement is by convention at emission points, not at the type level. Code review is responsible for verifying every LaTeX-derived text emission flows through `enforce_snippet_budget`.

### Neutral

- No impact on ADS body, OpenAlex metadata, S2ORC body, or any other non-arXiv source. Those carry their own licenses and are governed separately.

## Enforcement

1. **Helper**: `src/scix/sources/licensing.py` exposes `enforce_snippet_budget(body, canonical_url, budget=None) -> SnippetPayload`. Pure function, frozen dataclass return, no side effects.
2. **Default budget**: 500 characters, overridable via `SCIX_LATEX_SNIPPET_BUDGET` environment variable. Invalid values raise `ValueError` at call time.
3. **Call sites** (to be wired in `scix_experiments-wqr.5` and `scix_experiments-wqr.7`):
   - `src/scix/sources/ar5iv.py` body-return code path
   - `papers_fulltext` read path (whatever MCP tool or API endpoint reads the table)
   - Any future MCP tool that returns LaTeX-derived text
4. **Code review checklist**: reviewers MUST verify that any code path touching `papers_fulltext.body` or `ar5iv` parsed body either (a) is internal-only (indexing, embedding, analytics) or (b) wraps its emission through `enforce_snippet_budget` with a real `canonical_url`.
5. **Lint rule** (future): consider a CI check that flags direct access to `papers_fulltext.body` from code that also imports MCP or API response types. Not in scope for wqr.1.

## References

- [arXiv Submission License](https://info.arxiv.org/help/license/index.html)
- [arXiv Non-Exclusive Distribution License](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- [ar5iv / LaTeXML project](https://ar5iv.labs.arxiv.org/)
- PRD Build 5 epic: bead `scix_experiments-wqr`
- Work unit W4 (ar5iv fetch + parser): bead `scix_experiments-wqr.5`
- Work unit W6 (arXiv S3 src/ bulk ingest): bead `scix_experiments-wqr.7`

## Addendum (2026-04-17): Serving LaTeX-derived text under a published bibcode

This addendum extends the original decision to cover a specific emission pattern that was not explicitly addressed in the 2026-04-13 text: returning LaTeX-derived body text in response to a request for a *published* bibcode (e.g., an ApJ or MNRAS bibcode) whose underlying row in `papers_fulltext` was actually ingested from an arXiv LaTeX source (`ar5iv` or `arxiv_local`).

This is the "cross-bibcode LaTeX propagation" case. A preprint is posted to arXiv, later published in a journal under a different bibcode, and the only full-text row SciX holds is the LaTeX-derived one keyed to the arXiv bibcode. Agents that call `read_paper` with the *published* bibcode still deserve access to the content, but the license posture established above does not relax just because the request arrived under a different identifier.

### Rules

(a) **Snippet budget is invariant.** The <=500 character snippet budget (configurable via `SCIX_LATEX_SNIPPET_BUDGET`, see Enforcement §2 above) applies unchanged regardless of which bibcode the request arrived under. A request for the published bibcode that resolves to a LaTeX-derived row gets the same 500-character ceiling as a direct request for the arXiv bibcode. No propagation discount, no bulk mode, no exception.

(b) **`canonical_url` points to the arXiv record, not the publisher DOI.** When the served row is LaTeX-derived, the emitted `canonical_url` must be `https://arxiv.org/abs/{arxiv_id}` — the source of the text we actually parsed and emitted — and not the publisher DOI landing page for the published bibcode. The reader's "go read this at the source" link must match the source of the bytes. Linking to the publisher from arXiv-derived text would misattribute the redistribution origin and breaks the arXiv license contract that makes internal use legal in the first place.

(c) **Responses carry a `source_bibcode` field.** The response envelope must include a `source_bibcode` field exposing the bibcode of the row the text actually came from. If `source_bibcode != requested_bibcode`, the agent can see the propagation happened and reason about it (e.g., display both identifiers, warn the user, decide whether to trust the match). Hiding the origin behind a silent substitution is not acceptable — the agent must be able to observe that a cross-bibcode fetch occurred.

(d) **`LATEX_DERIVED_SOURCES` constant.** Propagation is gated by a single named constant:

```python
LATEX_DERIVED_SOURCES = {"ar5iv", "arxiv_local"}
```

Only rows whose `source` column is in `LATEX_DERIVED_SOURCES` are eligible for cross-bibcode propagation. This keeps the policy decision in one place and makes audit trivial: if a new LaTeX-derived source is added (say, a future `arxiv_s3` bulk path), it gets added to the constant and inherits the propagation policy. Anything not in the set is non-propagating by default.

(e) **Non-LaTeX sources never propagate across bibcodes.** Rows whose `source` is `s2orc`, `ads_body`, or `docling` are explicitly **not** eligible for cross-bibcode propagation. A request for a published bibcode whose only available full-text row is one of these sources must return a **miss-with-hint** — a structured response indicating the text exists under a different bibcode but cannot be served under the requested one — rather than serving the content. This floor exists because those sources carry different licensing postures than the arXiv LaTeX path: S2ORC text is governed by Semantic Scholar's ODC-BY terms keyed to specific paper records, ADS body text is governed by publisher agreements keyed to the published bibcode, and Docling output is derived from publisher PDFs whose licenses are per-paper and do not transfer. None of those licenses grant a cross-bibcode redistribution right, so we do not assume one.

### Rationale

The cross-bibcode case is common enough to matter (a significant fraction of corpus papers have arXiv preprint + journal publication pairs) and legally narrow enough to handle with a named allowlist rather than a general policy. arXiv's non-exclusive distribution license attaches to the arXiv record, which is stable across the preprint-to-publication lifecycle; the text we redistribute under that license does not stop being arXiv-licensed just because the paper later appeared in a journal. The `canonical_url` and `source_bibcode` requirements preserve provenance so downstream agents, reviewers, and users can see exactly what was served and why. The non-LaTeX floor keeps the policy conservative: when in doubt about license transferability, miss rather than over-serve.

