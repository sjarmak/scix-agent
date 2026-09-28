from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


async def run(execution_log: Path) -> None:
    server = Server("fake-eval-server", version="1")
    tools = [
        Tool(
            name="read_tool",
            description="Read a value",
            inputSchema={"type": "object", "properties": {"value": {"type": "string"}}},
        ),
        Tool(
            name="write_tool",
            description="Write a value",
            inputSchema={"type": "object", "properties": {"value": {"type": "string"}}},
        ),
    ]

    @server.list_tools()
    async def list_tools():
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        execution_log.parent.mkdir(parents=True, exist_ok=True)
        with execution_log.open("a") as handle:
            handle.write(json.dumps({"tool": name, "arguments": arguments}) + "\n")
        return [TextContent(type="text", text=f"executed:{name}")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-log", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.execution_log))


if __name__ == "__main__":
    main()
