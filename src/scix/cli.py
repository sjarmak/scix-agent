"""Thin CLI over the mechanical SciX tools.

A human/script front-end for the deterministic retrieval tools. It is a *thin
adapter*: subcommands are generated from the same ``_TOOL_SPECS`` the MCP server
advertises, and each call is dispatched through the same handler registry the MCP
server uses (``argv -> args dict -> handler -> JSON``). There is no logic here
beyond argv parsing and JSON printing — one core, two adapters (MCP + CLI).

Only the *mechanical* tools are exposed (see ``MECHANICAL_TOOLS``). The reasoning
tools (``lit_review``, ``synthesize_findings``) and the session/working-set tools
stay agent-only — a CLI would just re-spawn an LLM behind a worse interface.

Run:  ``python -m scix.cli <tool> [--arg ...]``  (or the ``scix`` entry point).
Output is the exact JSON the MCP tool returns; ``--pretty`` indents it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Any

# The deterministic subset (ZFC "mechanism"). Reasoning tools (lit_review,
# synthesize_findings) and session/working-set tools are intentionally excluded.
MECHANICAL_TOOLS: frozenset[str] = frozenset(
    {
        "search",
        "concept_search",
        "get_paper",
        "read_paper",
        "citation_traverse",
        "citation_similarity",
        "entity",
        "graph_context",
        "temporal_evolution",
        "facet_counts",
        "forward_citations",
    }
)

# JSON schema type -> how the CLI accepts it. object/array come in as a JSON
# string and are parsed at dispatch; boolean uses --flag/--no-flag.
_STR_TYPES = {"string"}
_INT_TYPES = {"integer"}
_NUM_TYPES = {"number"}
_BOOL_TYPES = {"boolean"}
_JSON_TYPES = {"object", "array"}


def _spec_type(prop: Mapping[str, Any]) -> str:
    """Normalize a property's JSON type (handles ``["string","null"]`` unions)."""
    t = prop.get("type", "string")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        return non_null[0] if non_null else "string"
    return t


def _primary_arg(spec: Mapping[str, Any]) -> str | None:
    """The first required string param, exposed as a convenience positional."""
    schema = spec.get("inputSchema", {})
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key in props and _spec_type(props[key]) in _STR_TYPES:
            return key
    return None


def _add_tool_subparser(sub: argparse._SubParsersAction, spec: Mapping[str, Any]) -> None:
    name = spec["name"]
    schema = spec.get("inputSchema", {})
    props: Mapping[str, Any] = schema.get("properties", {})
    primary = _primary_arg(spec)

    p = sub.add_parser(name, help=_first_line(spec.get("description", "")))
    p.set_defaults(_tool=name, _primary_key=primary)

    if primary is not None:
        p.add_argument(
            "primary",
            nargs="?",
            default=None,
            metavar=primary.upper(),
            help=f"shorthand for --{primary}",
        )

    # No argparse-level `required`: a required string is also exposed as an
    # optional positional, so requiredness is enforced uniformly at the args
    # layer (_check_required) instead — covers positional-or-flag.
    for key, prop in props.items():
        kind = _spec_type(prop)
        help_txt = _first_line(prop.get("description", ""))
        flag = f"--{key}"
        if kind in _BOOL_TYPES:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None, help=help_txt)
        elif kind in _INT_TYPES:
            p.add_argument(flag, type=int, default=None, help=help_txt)
        elif kind in _NUM_TYPES:
            p.add_argument(flag, type=float, default=None, help=help_txt)
        else:
            # string / object / array / enum — all taken as text; JSON parsed later
            choices = prop.get("enum")
            p.add_argument(
                flag,
                default=None,
                choices=choices,
                help=help_txt + (" (JSON)" if kind in _JSON_TYPES else ""),
            )


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def build_parser(specs: Iterable[Mapping[str, Any]]) -> argparse.ArgumentParser:
    """Build the CLI from tool specs, filtered to ``MECHANICAL_TOOLS``."""
    parser = argparse.ArgumentParser(prog="scix", description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    sub = parser.add_subparsers(dest="_tool", required=True, metavar="<tool>")
    for spec in specs:
        if spec.get("name") in MECHANICAL_TOOLS:
            _add_tool_subparser(sub, spec)
    return parser


def args_from_namespace(ns: argparse.Namespace, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the user-provided args into the handler's args dict.

    Only keys the user actually set are included, so each handler applies its own
    schema defaults for the rest. ``object``/``array`` values are JSON-parsed.
    """
    props: Mapping[str, Any] = spec.get("inputSchema", {}).get("properties", {})
    out: dict[str, Any] = {}
    primary = getattr(ns, "_primary_key", None)
    if primary is not None and getattr(ns, "primary", None) is not None:
        out[primary] = ns.primary
    for key, prop in props.items():
        val = getattr(ns, key, None)
        if val is None or key in out:
            continue
        if _spec_type(prop) in _JSON_TYPES:
            try:
                out[key] = json.loads(val)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"--{key} must be valid JSON: {exc}") from exc
        else:
            out[key] = val
    return out


def _check_required(spec: Mapping[str, Any], args: Mapping[str, Any]) -> None:
    """Enforce the schema's top-level required params (positional or flag)."""
    missing = [k for k in spec.get("inputSchema", {}).get("required", []) if k not in args]
    if missing:
        raise SystemExit(f"{spec['name']}: missing required arg(s): {', '.join(missing)}")


def run(
    argv: list[str],
    *,
    specs: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Callable[[Any, dict[str, Any]], str]],
    conn_factory: Callable[[], AbstractContextManager[Any]],
) -> int:
    """Parse argv, dispatch through the handler registry, print the JSON result.

    Injectable (specs/registry/conn_factory) so it is testable without a DB.
    """
    specs = list(specs)
    parser = build_parser(specs)
    ns = parser.parse_args(argv)
    tool = ns._tool
    spec = next(s for s in specs if s.get("name") == tool)
    args = args_from_namespace(ns, spec)
    _check_required(spec, args)

    handler = registry.get(tool)
    if handler is None:  # guarded by the parser, but fail explicitly
        raise SystemExit(f"no handler for tool {tool!r}")

    with conn_factory() as conn:
        result = handler(conn, args)

    if ns.pretty:
        try:
            result = json.dumps(json.loads(result), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # handler returned non-JSON; print as-is
    print(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    from scix.db import get_connection
    from scix.mcp_server import _handler_registry
    from scix.mcp_tool_specs import _TOOL_SPECS

    return run(
        sys.argv[1:] if argv is None else argv,
        specs=_TOOL_SPECS,
        registry=_handler_registry(),
        conn_factory=get_connection,
    )


if __name__ == "__main__":
    raise SystemExit(main())
