from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent

DEFAULT_READ_ONLY_CALLS = frozenset(
    {
        ("search", None),
        ("search_docs", None),
        ("get_paper", None),
        ("get_citations", None),
        ("get_references", None),
        ("get_metrics", None),
        ("export", None),
        ("health_check", None),
        ("get_libraries", None),
        ("get_library", None),
        ("get_permissions", None),
        ("get_annotation", None),
        ("library", "list"),
        ("library", "get"),
        ("library_permissions", "get"),
        ("library_annotations", "get"),
    }
)


def parse_server_command(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("server command must be a JSON array") from exc
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("server command must be a non-empty JSON array of strings")
    return value


class ProxyRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, tool: str, arguments: dict[str, Any], disposition: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tool": tool, "arguments": arguments, "disposition": disposition}
        with self.path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")


async def route_tool_call(
    name: str,
    arguments: dict[str, Any],
    read_only_calls: frozenset[tuple[str, str | None]],
    recorder: ProxyRecorder,
    call_downstream: Callable[[str, dict[str, Any]], Awaitable[CallToolResult]],
) -> CallToolResult:
    action = arguments.get("action")
    is_read_only = isinstance(action, str | type(None)) and (name, action) in read_only_calls
    if not is_read_only:
        recorder.record(name, arguments, "intercepted")
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "intercepted",
                            "tool": name,
                            "message": "Evaluation recorded this account mutation without executing it.",
                        }
                    ),
                )
            ],
            isError=False,
        )
    result = await call_downstream(name, arguments)
    recorder.record(name, arguments, "forwarded")
    return result


def downstream_environment() -> dict[str, str]:
    environment = dict(os.environ)
    ads_key = environment.get("ADS_API_KEY")
    if ads_key and not environment.get("SCIX_API_TOKEN"):
        environment["SCIX_API_TOKEN"] = ads_key
    return environment


async def run_proxy(
    server_command: list[str],
    read_only_calls: frozenset[tuple[str, str | None]],
    log_path: Path,
) -> None:
    parameters = StdioServerParameters(
        command=server_command[0],
        args=server_command[1:],
        env=downstream_environment(),
    )
    recorder = ProxyRecorder(log_path)
    async with stdio_client(parameters) as (downstream_read, downstream_write):
        async with ClientSession(downstream_read, downstream_write) as client:
            await client.initialize()
            tools = (await client.list_tools()).tools
            server = Server("mcp-tool-surface-proxy", version="1")

            @server.list_tools()
            async def list_tools():
                return tools

            @server.call_tool()
            async def call_tool(name: str, arguments: dict[str, Any]):
                return await route_tool_call(
                    name,
                    arguments,
                    read_only_calls,
                    recorder,
                    client.call_tool,
                )

            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-command-json", required=True)
    parser.add_argument("--read-only-calls-json", required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    args = parser.parse_args()
    server_command = parse_server_command(args.server_command_json)
    read_only_calls = frozenset(tuple(call) for call in json.loads(args.read_only_calls_json))
    asyncio.run(run_proxy(server_command, read_only_calls, args.log_file))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
