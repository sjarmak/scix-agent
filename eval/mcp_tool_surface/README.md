# MCP tool-surface evaluation

This harness measures whether Claude selects the expected MCP tool and required parameters for research intents. It accepts any stdio MCP server command expressed as a JSON argv array.

Set `ADS_API_KEY` for live SciX reads and provide whatever credentials `claude -p` normally uses. Credentials remain in the process environment and are not written to the generated MCP configuration or result records.

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

The task set contains live reads, intercepted account mutations, and explicit capability gaps. Capability gaps are reported but excluded from accuracy denominators. The proxy intercepts these tools without forwarding them:

- `create_library`
- `delete_library`
- `edit_library`
- `manage_documents`
- `add_documents_by_query`
- `library_operation`
- `update_permissions`
- `transfer_library`
- `manage_annotation`
- `delete_annotation`

All raw and scored results live under the ignored `results/` directory. Do not commit them.
