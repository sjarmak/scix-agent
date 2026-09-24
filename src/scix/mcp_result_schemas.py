"""Canonical successful-result schemas for the public MCP tool surface.

These schemas pin stable top-level consumer fields without constraining
additive fields.  They describe decoded JSON payloads returned in the MCP text
content; they do not introduce a new response wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

JsonSchema = dict[str, Any]


def _properties(**types: str | list[str]) -> JsonSchema:
    return {name: {"type": schema_type} for name, schema_type in types.items()}


def _object_schema(
    *,
    properties: JsonSchema,
    required: tuple[str, ...] = (),
    variants: tuple[tuple[str, ...], ...] = (),
) -> JsonSchema:
    schema: JsonSchema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = list(required)
    if variants:
        schema["anyOf"] = [{"required": list(fields)} for fields in variants]
    return schema


def _search_result_schema(*, allow_disambiguation: bool = False) -> JsonSchema:
    properties = _properties(papers="array", total="integer", timing_ms="object", metadata="object")
    if allow_disambiguation:
        properties["disambiguation"] = {"type": "array"}
        return _object_schema(
            properties=properties,
            variants=(("papers", "total", "timing_ms"), ("disambiguation",)),
        )
    return _object_schema(
        properties=properties,
        required=("papers", "total", "timing_ms"),
    )


def _citation_traverse_schema() -> JsonSchema:
    return _object_schema(
        properties={
            **_properties(papers="array", total="integer", timing_ms="object"),
            **_properties(mode="string", scope="string", bibcodes="array", by_bibcode="object"),
        },
        variants=(
            ("papers", "total", "timing_ms"),
            ("mode", "scope", "bibcodes", "by_bibcode"),
        ),
    )


def _entity_schema() -> JsonSchema:
    return _object_schema(
        properties={
            **_properties(query="string", candidates="array"),
            **_properties(entity_id=["integer", "null"], papers="array", total="integer"),
            **_properties(timing_ms="object", metadata="object"),
        },
        variants=(
            ("query", "candidates", "total"),
            ("papers", "total"),
            ("entity_id", "papers", "total"),
            ("papers", "total", "timing_ms"),
        ),
    )


def _graph_context_schema() -> JsonSchema:
    return _object_schema(
        properties={
            **_properties(papers="array", total="integer", timing_ms="object"),
            **_properties(bibcode="string", metrics="object", community="object"),
        },
        variants=(
            ("papers", "total", "timing_ms"),
            ("bibcode", "metrics", "community"),
        ),
    )


def _find_gaps_schema() -> JsonSchema:
    return _object_schema(
        properties=_properties(
            papers="array",
            total="integer",
            signal="string",
            resolution="string",
            message="string",
        ),
        required=("papers", "total", "signal"),
    )


def _claim_blame_schema() -> JsonSchema:
    return _object_schema(
        properties=_properties(
            origin="string",
            lineage="array",
            confidence="number",
            retraction_warnings="array",
            coverage="object",
        ),
        required=("origin", "lineage", "confidence", "retraction_warnings", "coverage"),
    )


def _forward_citations_schema() -> JsonSchema:
    return _object_schema(
        properties={
            **_properties(citations="array", papers="array", total="integer", coverage="object"),
            **_properties(target_bibcode="string", intent=["string", "null"]),
        },
        variants=(
            ("citations", "total", "coverage"),
            ("target_bibcode", "intent", "papers", "total", "coverage"),
        ),
    )


def _synthesize_findings_schema() -> JsonSchema:
    return _object_schema(
        properties=_properties(
            sections="array",
            unattributed_bibcodes="array",
            assignment_coverage="object",
            metadata="object",
        ),
        required=("sections", "unattributed_bibcodes", "assignment_coverage", "metadata"),
    )


_BUILDERS: dict[str, Callable[[], JsonSchema]] = {
    "search": lambda: _search_result_schema(allow_disambiguation=True),
    "lit_review": _search_result_schema,
    "concept_search": _search_result_schema,
    "get_paper": _search_result_schema,
    "read_paper": _search_result_schema,
    "citation_traverse": _citation_traverse_schema,
    "citation_similarity": _search_result_schema,
    "entity": _entity_schema,
    "graph_context": _graph_context_schema,
    "find_gaps": _find_gaps_schema,
    "temporal_evolution": _search_result_schema,
    "facet_counts": _search_result_schema,
    "claim_blame": _claim_blame_schema,
    "forward_citations": _forward_citations_schema,
    "synthesize_findings": _synthesize_findings_schema,
}


def result_schema_for_tool(tool_name: str) -> JsonSchema:
    """Return a fresh result schema for a default-visible tool."""
    try:
        return _BUILDERS[tool_name]()
    except KeyError as exc:
        raise KeyError(
            f"no successful-result schema registered for MCP tool {tool_name!r}"
        ) from exc


def registered_result_schema_names() -> frozenset[str]:
    """Return the tools covered by the successful-result schema registry."""
    return frozenset(_BUILDERS)
