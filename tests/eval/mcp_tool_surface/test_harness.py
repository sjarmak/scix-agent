from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

from scix.eval.mcp_tool_surface.proxy import (
    DEFAULT_READ_ONLY_CALLS,
    ProxyRecorder,
    parse_server_command,
    route_tool_call,
)
from scix.eval.mcp_tool_surface.runner import (
    RunResult,
    build_mcp_config,
    claude_environment,
    invoke_claude,
    parse_stream_json,
    run_matrix,
)
from scix.eval.mcp_tool_surface.scorer import (
    aggregate,
    score_run,
    strip_mcp_prefix,
)
from scix.eval.mcp_tool_surface.scorer import (
    main as score_main,
)
from scix.eval.mcp_tool_surface.tasks import load_tasks, validate_task


def test_parse_server_command_accepts_arbitrary_argv() -> None:
    assert parse_server_command('["node", "build/index.js", "--stdio"]') == [
        "node",
        "build/index.js",
        "--stdio",
    ]


@pytest.mark.parametrize("raw", ["{}", "[]", '["node", 1]', "not-json"])
def test_parse_server_command_rejects_invalid_argv(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_server_command(raw)


def test_default_read_only_calls_match_upstream_tools() -> None:
    assert DEFAULT_READ_ONLY_CALLS == frozenset(
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


def test_route_tool_call_intercepts_without_downstream_execution(tmp_path: Path) -> None:
    downstream_calls: list[tuple[str, dict[str, object]]] = []

    async def call_downstream(name: str, arguments: dict[str, object]) -> CallToolResult:
        downstream_calls.append((name, arguments))
        return CallToolResult(content=[TextContent(type="text", text="executed")])

    log_path = tmp_path / "calls.jsonl"
    recorder = ProxyRecorder(log_path)
    result = asyncio.run(
        route_tool_call(
            "delete_library",
            {"library_id": "library-123"},
            DEFAULT_READ_ONLY_CALLS,
            recorder,
            call_downstream,
        )
    )

    assert downstream_calls == []
    assert result.isError is False
    assert "intercepted" in result.content[0].text
    assert json.loads(log_path.read_text()) == {
        "tool": "delete_library",
        "arguments": {"library_id": "library-123"},
        "disposition": "intercepted",
    }


def test_route_tool_call_forwards_read_only_tool(tmp_path: Path) -> None:
    async def call_downstream(name: str, arguments: dict[str, object]) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text=f"{name}:{arguments['bibcode']}")]
        )

    log_path = tmp_path / "calls.jsonl"
    result = asyncio.run(
        route_tool_call(
            "get_paper",
            {"bibcode": "2024ApJ...001A...1A"},
            DEFAULT_READ_ONLY_CALLS,
            ProxyRecorder(log_path),
            call_downstream,
        )
    )

    assert result.content[0].text == "get_paper:2024ApJ...001A...1A"
    assert json.loads(log_path.read_text())["disposition"] == "forwarded"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("library", {"action": "delete", "library_id": "library-123"}),
        ("library_documents", {"action": "add", "library_id": "library-123"}),
        ("unknown_tool", {}),
        ("library", {"library_id": "library-123"}),
        ("library", {"action": ["get"]}),
    ],
)
def test_route_tool_call_intercepts_every_call_outside_read_only_allowlist(
    tmp_path: Path, name: str, arguments: dict[str, object]
) -> None:
    downstream_calls: list[tuple[str, dict[str, object]]] = []

    async def call_downstream(tool: str, values: dict[str, object]) -> CallToolResult:
        downstream_calls.append((tool, values))
        return CallToolResult(content=[TextContent(type="text", text="executed")])

    result = asyncio.run(
        route_tool_call(
            name,
            arguments,
            DEFAULT_READ_ONLY_CALLS,
            ProxyRecorder(tmp_path / "calls.jsonl"),
            call_downstream,
        )
    )

    assert downstream_calls == []
    assert "intercepted" in result.content[0].text


@pytest.mark.parametrize(
    ("name", "action"),
    [
        ("library", "list"),
        ("library", "get"),
        ("library_permissions", "get"),
        ("library_annotations", "get"),
    ],
)
def test_route_tool_call_forwards_merged_reads(tmp_path: Path, name: str, action: str) -> None:
    downstream_calls: list[tuple[str, dict[str, object]]] = []

    async def call_downstream(tool: str, values: dict[str, object]) -> CallToolResult:
        downstream_calls.append((tool, values))
        return CallToolResult(content=[TextContent(type="text", text="executed")])

    result = asyncio.run(
        route_tool_call(
            name,
            {"action": action},
            DEFAULT_READ_ONLY_CALLS,
            ProxyRecorder(tmp_path / "calls.jsonl"),
            call_downstream,
        )
    )

    assert downstream_calls == [(name, {"action": action})]
    assert result.content[0].text == "executed"


def test_build_mcp_config_wraps_arbitrary_server_command(tmp_path: Path) -> None:
    config = build_mcp_config(
        ["npx", "-y", "scix-mcp"],
        tmp_path / "calls.jsonl",
        frozenset({("search", None), ("library", "get")}),
    )

    entry = config["mcpServers"]["eval_target"]
    assert entry["args"][0:2] == ["-m", "scix.eval.mcp_tool_surface.proxy"]
    command_index = entry["args"].index("--server-command-json") + 1
    allowlist_index = entry["args"].index("--read-only-calls-json") + 1
    assert json.loads(entry["args"][command_index]) == ["npx", "-y", "scix-mcp"]
    assert json.loads(entry["args"][allowlist_index]) == [["library", "get"], ["search", None]]
    assert "ADS_API_KEY" not in json.dumps(config)


def test_proxy_stdio_forwards_reads_and_intercepts_writes(tmp_path: Path) -> None:
    execution_log = tmp_path / "executed.jsonl"
    proxy_log = tmp_path / "proxy.jsonl"
    downstream_command = [
        sys.executable,
        "-m",
        "tests.eval.mcp_tool_surface.fake_server",
        "--execution-log",
        str(execution_log),
    ]
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "scix.eval.mcp_tool_surface.proxy",
            "--server-command-json",
            json.dumps(downstream_command),
            "--read-only-calls-json",
            json.dumps([["read_tool", None]]),
            "--log-file",
            str(proxy_log),
        ],
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )

    async def exercise_proxy():
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                assert {tool.name for tool in (await client.list_tools()).tools} == {
                    "read_tool",
                    "write_tool",
                }
                read_result = await client.call_tool("read_tool", {"value": "safe"})
                write_result = await client.call_tool("write_tool", {"value": "blocked"})
                return read_result, write_result

    read_result, write_result = asyncio.run(exercise_proxy())

    assert read_result.content[0].text == "executed:read_tool"
    assert "intercepted" in write_result.content[0].text
    assert [json.loads(line)["tool"] for line in execution_log.read_text().splitlines()] == [
        "read_tool"
    ]
    assert [json.loads(line)["disposition"] for line in proxy_log.read_text().splitlines()] == [
        "forwarded",
        "intercepted",
    ]


def test_parse_stream_json_extracts_mcp_calls_and_text() -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "mcp__eval_target__search",
                                "input": {"query": "dark energy"},
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Done"}]},
                }
            ),
        ]
    )

    calls, final_text = parse_stream_json(stdout)

    assert calls == [{"name": "mcp__eval_target__search", "input": {"query": "dark energy"}}]
    assert final_text == "Done"


def test_invoke_claude_returns_process_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Process:
        returncode = 7

        async def communicate(self, prompt: bytes):
            assert prompt == b"Find papers"
            return b"stdout", b"stderr"

    async def create_process(*command, **options):
        assert command == ("claude", "-p")
        assert options["cwd"] == tmp_path
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    result = asyncio.run(invoke_claude(["claude", "-p"], "Find papers", tmp_path, 5))

    assert result == (b"stdout", b"stderr", 7)


def test_claude_environment_removes_paid_api_key_and_disables_tool_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "paid-secret")
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "true")

    environment = claude_environment()

    assert "ANTHROPIC_API_KEY" not in environment
    assert environment["ENABLE_TOOL_SEARCH"] == "false"


def test_invoke_claude_kills_timed_out_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    killed = False
    waited = False

    class Process:
        returncode = None

        async def communicate(self, prompt: bytes):
            await asyncio.Event().wait()

        def kill(self):
            nonlocal killed
            killed = True

        async def wait(self):
            nonlocal waited
            waited = True

    async def create_process(*command, **options):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    result = asyncio.run(invoke_claude(["claude"], "Find papers", tmp_path, 0.01))

    assert result == (b"", b"timeout", -1)
    assert killed is True
    assert waited is True


def test_run_matrix_writes_each_model_task_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_run_one(task, model, server_command, out_dir, timeout_seconds):
        return RunResult(
            session_id=f"{model}-{task['id']}",
            model=model,
            task_id=task["id"],
            prompt=task["prompt"],
            execution=task["execution"],
            tool_calls=[{"name": "mcp__eval_target__search", "input": {}}],
            final_text="done",
            started_at="start",
            finished_at="finish",
            exit_code=0,
            stderr_tail="",
        )

    monkeypatch.setattr("scix.eval.mcp_tool_surface.runner.run_one", fake_run_one)
    tasks = [
        {"id": "one", "prompt": "Find papers", "execution": "live"},
        {"id": "two", "prompt": "Find citations", "execution": "live"},
    ]
    (tmp_path / "runs.jsonl").write_text('{"task_id":"stale"}\n')

    path = asyncio.run(run_matrix(tasks, ["sonnet", "haiku"], ["server"], tmp_path, 2, 5))
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 4
    assert {(row["model"], row["task_id"]) for row in rows} == {
        ("sonnet", "one"),
        ("sonnet", "two"),
        ("haiku", "one"),
        ("haiku", "two"),
    }


def test_score_run_checks_tool_required_keys_and_exact_values() -> None:
    task = {
        "id": "task-1",
        "intent": "search",
        "execution": "live",
        "oracle": {
            "tool": "search",
            "required_keys": ["query"],
            "args_subset": {"rows": 5, "sort": "citation_count desc"},
        },
    }
    run = {
        "task_id": "task-1",
        "model": "sonnet",
        "tool_calls": [
            {
                "name": "mcp__eval_target__search",
                "input": {
                    "query": "JWST exoplanets",
                    "rows": 5,
                    "sort": "citation_count desc",
                },
            }
        ],
    }

    scored = score_run(run, task)

    assert scored["tool_correct"] is True
    assert scored["params_correct"] is True


@pytest.mark.parametrize(
    ("oracle", "call"),
    [
        (
            {"tool": "get_library", "args_subset": {"library_id": "lib-1"}},
            {
                "name": "mcp__eval_target__library",
                "input": {"action": "get", "library_id": "lib-1"},
            },
        ),
        (
            {
                "tool": "manage_documents",
                "args_subset": {"library_id": "lib-1", "bibcodes": ["code"], "action": "remove"},
            },
            {
                "name": "mcp__eval_target__library_documents",
                "input": {"action": "remove", "library_id": "lib-1", "bibcodes": ["code"]},
            },
        ),
        (
            {
                "tool": "manage_documents",
                "args_subset": {"library_id": "lib-1", "bibcodes": ["code"], "action": "add"},
            },
            {
                "name": "mcp__eval_target__library_documents",
                "input": {"action": "add", "library_id": "lib-1", "bibcodes": ["code"]},
            },
        ),
    ],
)
def test_score_run_maps_merged_tool_names_and_actions(
    oracle: dict[str, object], call: dict[str, object]
) -> None:
    task = {"id": "task-1", "intent": "library", "execution": "live", "oracle": oracle}
    run = {"task_id": "task-1", "model": "sonnet", "tool_calls": [call]}
    mapping = {
        "get_library": {"tool": "library", "action": "get"},
        "manage_documents": {"tool": "library_documents", "action_from": "action"},
    }

    scored = score_run(run, task, mapping)

    assert scored["tool_correct"] is True
    assert scored["params_correct"] is True


def test_score_run_marks_capability_gap_without_failure() -> None:
    task = {
        "id": "gap-1",
        "intent": "section_retrieval",
        "execution": "capability_gap",
        "gap_reason": "No section-level retrieval tool",
    }
    scored = score_run(
        {"task_id": "gap-1", "model": "haiku", "tool_calls": []},
        task,
    )

    assert scored["capability_gap"] is True
    assert scored["tool_correct"] is None
    assert scored["params_correct"] is None


def test_score_run_marks_missing_tool_call_as_failure() -> None:
    task = {
        "id": "task-1",
        "intent": "search",
        "execution": "live",
        "oracle": {"tool": "search"},
    }

    scored = score_run({"task_id": "task-1", "model": "sonnet", "tool_calls": []}, task)

    assert scored["tool_correct"] is False
    assert scored["params_correct"] is False


def test_strip_mcp_prefix_preserves_plain_and_malformed_names() -> None:
    assert strip_mcp_prefix("search") == "search"
    assert strip_mcp_prefix("mcp__search") == "mcp__search"


def test_aggregate_excludes_capability_gaps_from_accuracy() -> None:
    summary = aggregate(
        [
            {
                "model": "sonnet",
                "intent": "search",
                "capability_gap": False,
                "tool_correct": True,
                "params_correct": False,
            },
            {
                "model": "sonnet",
                "intent": "section_retrieval",
                "capability_gap": True,
                "tool_correct": None,
                "params_correct": None,
            },
        ]
    )

    assert summary["models"]["sonnet"]["scored_tasks"] == 1
    assert summary["models"]["sonnet"]["capability_gaps"] == 1
    assert summary["models"]["sonnet"]["tool_accuracy"] == 1.0
    assert summary["models"]["sonnet"]["param_accuracy"] == 0.0


def test_score_main_writes_local_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    out_dir = tmp_path / "scores"
    task = {
        "id": "one",
        "prompt": "Find papers",
        "intent": "search",
        "execution": "live",
        "oracle": {"tool": "search"},
    }
    run = {
        "session_id": "session",
        "model": "sonnet",
        "task_id": "one",
        "tool_calls": [{"name": "mcp__eval_target__search", "input": {}}],
    }
    tasks_path.write_text(json.dumps(task))
    runs_path.write_text(json.dumps(run))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scorer",
            "--runs",
            str(runs_path),
            "--tasks",
            str(tasks_path),
            "--out-dir",
            str(out_dir),
        ],
    )

    score_main()

    assert (
        json.loads((out_dir / "summary.json").read_text())["models"]["sonnet"]["tool_accuracy"]
        == 1.0
    )
    assert json.loads((out_dir / "scored.jsonl").read_text())["tool_correct"] is True
    assert '"tool_accuracy": 1.0' in capsys.readouterr().out


def test_load_tasks_validates_unique_ids_and_execution_modes(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "prompt": "Find papers",
                        "intent": "search",
                        "execution": "live",
                        "oracle": {"tool": "search", "required_keys": ["query"]},
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "prompt": "Find sections",
                        "intent": "section_retrieval",
                        "execution": "capability_gap",
                        "gap_reason": "Unsupported",
                    }
                ),
            ]
        )
    )

    assert [task["id"] for task in load_tasks(path)] == ["a", "b"]


@pytest.mark.parametrize(
    "task, message",
    [
        ({"prompt": "Missing id", "intent": "search", "execution": "live"}, "required"),
        (
            {"id": "a", "prompt": "Bad mode", "intent": "search", "execution": "wrong"},
            "invalid",
        ),
        (
            {
                "id": "a",
                "prompt": "Missing reason",
                "intent": "gap",
                "execution": "capability_gap",
            },
            "gap_reason",
        ),
        (
            {"id": "a", "prompt": "Missing oracle", "intent": "search", "execution": "live"},
            "oracle",
        ),
    ],
)
def test_validate_task_rejects_invalid_tasks(task: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_task(task)


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    task = {
        "id": "same",
        "prompt": "Find papers",
        "intent": "search",
        "execution": "live",
        "oracle": {"tool": "search"},
    }
    path.write_text(f"{json.dumps(task)}\n{json.dumps(task)}\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(path)
