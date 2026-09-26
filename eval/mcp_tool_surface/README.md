# MCP tool-surface evaluation

This harness measures whether Claude selects the expected MCP tool and required parameters for research intents. It accepts any stdio MCP server command expressed as a JSON argv array.

Set `ADS_API_KEY` for live SciX reads and authenticate `claude` with OAuth. The runner removes `ANTHROPIC_API_KEY` from the child environment and disables tool search so evaluations cannot use paid API credentials and always load the complete MCP surface.

Run a ten-task smoke test against the published SciX server:

```bash
PYTHONPATH=src python -m scix.eval.mcp_tool_surface.runner \
  --server-command-json '["npx", "-y", "scix-mcp"]' \
  --models sonnet haiku \
  --limit 10 \
  --out-dir results/mcp_tool_surface/scix_mcp_smoke
```

Score completed runs:

```bash
PYTHONPATH=src python -m scix.eval.mcp_tool_surface.scorer \
  --runs results/mcp_tool_surface/scix_mcp_smoke/runs.jsonl \
  --out-dir results/mcp_tool_surface/scix_mcp_smoke
```

Pass `--variant merged` when scoring the consolidated library-tool server.

The task set contains live reads, intercepted account mutations, and explicit capability gaps. Capability gaps are reported but excluded from accuracy denominators. The proxy forwards only explicitly allowlisted read-only tool/action pairs; unknown tools, missing actions, unknown actions, and all writes are intercepted.

All raw and scored results live under the ignored `results/` directory. Do not commit them.
