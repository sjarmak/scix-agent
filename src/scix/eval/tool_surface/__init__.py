"""Tool-surface eval harness.

Compares agent tool-selection behavior across alternative MCP tool surfaces:

- ``v0`` — current 18-tool surface (baseline)
- ``v1`` — proposed 8-tool consolidated surface
- ``v2`` — v1 with terse descriptions stripped (description-quality ablation)

Used to measure how schema/description changes affect tool-selection accuracy
before committing to a real consolidation. See ``docs/eval/tool_surface_eval.md``.
"""
