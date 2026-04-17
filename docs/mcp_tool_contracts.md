# MCP Tool Contracts

**Status:** PRD R3 / R16 — initial cut for `read_paper` at `schema_version=2`.
**Scope:** Response-shape contracts for SciX MCP tools that needed
additive changes in the arXiv-LaTeX + sibling-routing work (PRD Build 5,
ADR-006). Currently concerns `read_paper`; future tool additions append to
this document.
**Related:**
- `docs/ADR/006_arxiv_licensing.md` (Addendum: sibling routing +
  licensing semantics — the authority for canonical-URL, snippet budget,
  and suppression behavior)
- `docs/mcp_dual_lane_contract.md` (enrichment lane policy)
- `docs/section_schema_contract.md` (structured section payloads)

## Purpose

Nail down the exact JSON shape returned by `read_paper` when the server is
speaking `schema_version=2`, so agents and downstream callers can rely on
field presence and meaning without reverse-engineering the implementation.

This doc covers:

1. The full set of additive v2 fields.
2. Four concrete JSON examples spanning the observed response scenarios.
3. The backward-compatibility contract (v1 fields remain; v2 is additive
   only).
4. The `X-MCP-Schema-Version` header negotiation rule (R16).
5. The tool-surface invariant (13 verbs) that governs any change to this
   contract.

## Scope

- **In scope:** `read_paper` response shape, sibling-routing fields,
  licensing-derived fields (canonical URL, suppression, snippet semantics),
  schema-version negotiation.
- **Out of scope:** Request shape, other MCP verbs (tracked elsewhere),
  internal ingestion pipelines (covered by PRD Build 5 docs).

## `read_paper` at `schema_version=2`

`schema_version=2` is the current contract version emitted by `read_paper`.
A v2 response is a strict superset of v1: every v1 field remains at the
same path with the same meaning, and a v1-only client can parse a v2
response by ignoring unknown keys.

### Additive v2 fields

All fields below are new in `schema_version=2`. They are **additive only** —
no v1 field was renamed, removed, or had its type changed.

| Field                                  | Type      | Required         | Meaning                                                                                                                                                                               |
| -------------------------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                       | integer   | always           | Contract version of this response. Always `2` when this contract applies.                                                                                                             |
| `canonical_url`                        | string    | when `source="arxiv_latex"` or any LaTeX-derived snippet is returned | Public URL the reader should visit to see the authoritative paper (per ADR-006: `https://arxiv.org/abs/{arxiv_id}` for arXiv-derived bodies). Never empty when required.              |
| `served_from_sibling_bibcode`          | string    | only on sibling-routed hits | The bibcode that actually holds the body text, when the request bibcode resolved through a sibling link (e.g. journal bibcode resolved to the arXiv preprint sibling). Omitted on direct hits. |
| `source_bibcode`                       | string    | always           | The bibcode the body (or metadata, if no body) was materialised from. Equals the requested bibcode on a direct hit; equals `served_from_sibling_bibcode` when routed through a sibling. |
| `source_version`                       | string    | when body present | Version identifier of the source material (e.g. arXiv `v2`, publisher version tag, or ingest snapshot ID). Absent on abstract-only or no-body responses.                               |
| `fulltext_available_under_sibling`     | boolean   | on miss-with-hint | `true` when the requested bibcode has no servable body but a sibling bibcode does. Paired with `hint` to guide the caller.                                                             |
| `hint`                                 | string    | on miss-with-hint | Human-readable guidance describing how to reach the available full text (typically the sibling bibcode to re-query). Free-form; not a machine contract.                                |
| `suppressed_by_publisher`              | boolean   | always           | `true` when the body is withheld under publisher-side suppression (licensing, embargo, or takedown). When `true`, `source` is `"abstract"` and no body is returned.                   |

### Interaction rules

- If `served_from_sibling_bibcode` is present, then `source_bibcode` MUST
  equal `served_from_sibling_bibcode` and `canonical_url` MUST point at
  that sibling's public URL.
- If `suppressed_by_publisher` is `true`, then `source` is `"abstract"`,
  no `body` is returned, and `source_version` is absent.
- If `fulltext_available_under_sibling` is `true`, then `hint` MUST be
  present and `body` MUST be absent. `canonical_url` MAY be present
  pointing at the sibling's authoritative URL.
- `canonical_url` is mandatory on any response that includes
  LaTeX-derived body text — this is a hard contract, enforced by
  `scix.sources.licensing.enforce_snippet_budget` per ADR-006.

## Example responses

### 1. Direct hit on requested bibcode

Requested bibcode is the source of record; body is served directly, no
sibling routing involved.

```json
{
  "schema_version": 2,
  "bibcode": "2024ApJ...961...42S",
  "title": "A Direct-Hit Example Paper",
  "abstract": "We present a worked example ...",
  "source": "publisher_body",
  "source_bibcode": "2024ApJ...961...42S",
  "source_version": "published-v1",
  "canonical_url": "https://doi.org/10.3847/1538-4357/ad1234",
  "body": "Section 1. Introduction ...",
  "suppressed_by_publisher": false
}
```

### 2. LaTeX sibling hit (served from arXiv preprint)

Requested bibcode is the journal version; body is served from the arXiv
preprint sibling. `canonical_url` points to arxiv.org per ADR-006.

```json
{
  "schema_version": 2,
  "bibcode": "2024ApJ...961...42S",
  "title": "A Sibling-Routed Example Paper",
  "abstract": "We present a worked example ...",
  "source": "arxiv_latex",
  "served_from_sibling_bibcode": "2023arXiv231012345S",
  "source_bibcode": "2023arXiv231012345S",
  "source_version": "arxiv-v2",
  "canonical_url": "https://arxiv.org/abs/2310.12345",
  "body": "Section 1. Introduction ...",
  "suppressed_by_publisher": false
}
```

### 3. Non-LaTeX miss with hint

Requested bibcode has no servable body in this lane, but a sibling
bibcode does. The response carries the hint; no body is returned.

```json
{
  "schema_version": 2,
  "bibcode": "2024ApJ...961...42S",
  "title": "A Miss-With-Hint Example Paper",
  "abstract": "We present a worked example ...",
  "source": "abstract",
  "source_bibcode": "2024ApJ...961...42S",
  "fulltext_available_under_sibling": true,
  "hint": "Full text available via sibling bibcode 2023arXiv231012345S — re-query read_paper with that bibcode.",
  "canonical_url": "https://arxiv.org/abs/2310.12345",
  "suppressed_by_publisher": false
}
```

### 4. Suppressed by publisher

Body is withheld under publisher suppression (licensing, embargo, or
takedown). Abstract is still returned; no body; no `source_version`.

```json
{
  "schema_version": 2,
  "bibcode": "2024ApJ...961...42S",
  "title": "A Publisher-Suppressed Example Paper",
  "abstract": "We present a worked example ...",
  "source": "abstract",
  "source_bibcode": "2024ApJ...961...42S",
  "suppressed_by_publisher": true
}
```

## Backward-compatibility contract

- **All v1 fields remain.** `bibcode`, `title`, `abstract`, `source`,
  `body` (when present) keep their v1 paths, types, and semantics.
- **v2 is additive only.** Every new field listed above is new; no v1
  field was renamed, removed, or had its type changed.
- **Unknown-key tolerance required of clients.** A v1-only client MUST
  tolerate unknown keys in the response. A v2-aware client MUST tolerate
  absence of any non-required v2 field (see the Required column).
- **No semantic change for direct hits.** A direct-hit v2 response differs
  from a v1 response only by the additive fields (`schema_version`,
  `source_bibcode`, `source_version`, `canonical_url` where licensing
  requires it, `suppressed_by_publisher=false`). Downstream v1 readers see
  the same body and metadata they saw before.

Breaking changes (field removal, type change, semantics change) require
bumping `schema_version` to `3` and adding a new contract section here.

## `X-MCP-Schema-Version` header negotiation (R16)

Clients negotiate schema version via the `X-MCP-Schema-Version` request
header:

- **Request header:** `X-MCP-Schema-Version: 2` asks the server to emit
  `schema_version=2`. Omitting the header asks for the server default
  (currently `2`).
- **Response header:** The server echoes `X-MCP-Schema-Version: <n>` on
  every response, set to the schema version actually emitted.
- **Downgrade:** If a client requests a version the server does not
  support, the server responds with the highest version it supports and
  sets the response header accordingly. Clients MUST read the response
  header to confirm the emitted version before parsing.
- **Upgrade:** A server implementing a newer schema MUST still accept and
  honour requests for older supported versions; this preserves the
  additive contract across deploys.
- **Field-level fallback:** On a server that emits a version lower than
  the client requested, all guarantees of that lower version apply;
  v2-only fields will be absent, and the client MUST fall back to v1
  parsing.

The header is the sole negotiation channel; there is no body-level
version echo beyond the `schema_version` field itself, which exists for
in-flight and logged payload interpretation.

## Tool-surface invariant: 13 verbs

**Invariant: the SciX MCP tool surface stays at 13 verbs.** Adding a new
top-level tool to satisfy a contract change is forbidden. All schema
evolution happens by extending existing tool response shapes under a new
`schema_version`, as this document does for `read_paper`.

Rationale (tracked in `CLAUDE.md` "Tool Count Concern"): agent tool-
selection accuracy degrades past ~15 tools (A-RAG, Feb 2026). Holding the
surface at 13 keeps selection quality inside the known-good band and
forces response-shape evolution into versioned contracts rather than
verb sprawl.

Any proposal that would add a 14th verb requires an ADR update and
explicit sign-off; it is not a routine change.

## References

- `docs/ADR/006_arxiv_licensing.md` — the Addendum at the bottom of this
  ADR is the licensing-semantics authority for `canonical_url`, snippet
  budget, sibling routing, and `suppressed_by_publisher`. Read this
  before changing any of those fields.
- `docs/mcp_dual_lane_contract.md` — static vs JIT enrichment lanes.
- `docs/section_schema_contract.md` — structured section payloads for
  full-text responses.
- `CLAUDE.md` — "Tool Count Concern" section, A-RAG citation for the
  13-verb invariant.
