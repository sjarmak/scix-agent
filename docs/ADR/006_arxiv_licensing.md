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
