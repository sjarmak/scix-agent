# ADR-017: Add a provenance-visible software-engineering metadata lane

- **Status**: Accepted (2026-08-06); implementation remains open under bead `scix_experiments-d0j`.
- **Decision owner**: Stephanie Jarmak.
- **Scope**: discovery of software-engineering literature absent from the ADS/SciX corpus.
- **Evidence**: `docs/eval/se_venue_coverage_2026-08.md`.
- **Related constraints**: ADR-006 and the OA/preprint gate for body-AI; the 15-tool MCP cap; existing OpenAlex snapshot ingestion.

## Context

SciX is broad within its scientific remit but is not a general
software-engineering index. A deterministic 40-DOI diagnostic drawn from core
SE venue candidates produced eight exact SciX matches. All eight were sampled
TSE papers; none of the sampled TOSEM, EMSE, CSCW, or ASE-journal records
matched. The sample is not a recall estimate, but it rejects the assumption
that SciX plus arXiv can stand alone for an SE secondary study.

The repository already ingests OpenAlex snapshot metadata into
`papers_openalex` and resolves DOI/arXiv identities to ADS records. That data
can expose works outside ADS. Treating it as if it were another SciX record,
however, would hide the exact coverage boundary that the audit found.

## Decision

1. Add a **federated metadata discovery lane** over the existing OpenAlex
   snapshot for works not represented in ADS/SciX. Crossref may resolve DOI,
   title, author, venue, date, and update identity.
2. Preserve `source_lane` on every candidate and result. At minimum distinguish
   `scix_ads`, `openalex_metadata`, `crossref_metadata`, and
   `publisher_native_manual` when the last is supplied by a review workflow.
3. Do not merge an OpenAlex-only work silently into the SciX lane and do not
   describe OpenAlex or Crossref discovery as an ACM DL, IEEE Xplore, or Scopus
   search.
4. Do not add a sixteenth MCP tool. Extend an existing literature-search
   response or an offline review workflow after the tool-contract change is
   designed and approved.
5. Metadata discovery grants no right to process closed full text. Abstract and
   metadata processing follow current source terms; body-AI remains restricted
   by `papers_is_oa_or_preprint(papers)` or an equivalent gate for federated
   records.
6. Evaluate the lane against a versioned DOI set stratified by venue and year.
   Report exact-identity recall separately for SciX and federated results, plus
   duplicate resolution and source-lane errors.

## Rejected alternatives

### Treat SciX plus arXiv as sufficient

Rejected by the audit. Empty retrieval can reflect corpus absence rather than
absence from the literature.

### Treat OpenAlex as a publisher-native substitute

Rejected because it erases review provenance. OpenAlex is a valuable discovery
index, not evidence that ACM, IEEE, or Scopus was searched under its own index
and query behavior.

### Ingest closed publisher full text

Rejected. It creates a licensing and terms-of-use obligation unrelated to the
metadata coverage problem and conflicts with the existing body-AI gate.

### Add a dedicated MCP search tool

Rejected under the 15-tool cap. The source distinction belongs in the data and
response contract, not in another near-duplicate tool.

## Consequences

- Literature workflows can recover relevant non-ADS records without claiming
  that SciX itself contains them.
- Every synthesis must retain source-lane provenance through deduplication and
  citation output.
- Search and ranking tests need mixed SciX/OpenAlex fixtures and explicit
  duplicate cases before serving changes.
- Publisher-native searches remain necessary when a review protocol requires
  them. SciX can preserve their imported decisions but cannot manufacture that
  search history.
- The current task remains open until a contract and implementation plan name
  the existing response surface to extend, the venue/year evaluation set is
  expanded beyond the diagnostic sample, and the federated lane passes tests.
