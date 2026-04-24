# MCP importer

Shadow ingests [Model Context Protocol](https://modelcontextprotocol.io/)
session logs — the JSON-RPC 2.0 protocol agents use to talk to
external tools via MCP servers. Adopted by Claude Desktop, Cursor,
Windsurf, Zed, and VS Code by early 2026.

## Usage

```bash
shadow import --format mcp session.jsonl -o imported.agentlog
shadow diff baseline.agentlog imported.agentlog
```

## Input shapes

The importer auto-detects three formats:

1. **JSONL stream** — one JSON-RPC message per line. What
   `mcp-server --log` emits.
2. **JSON array** — `[msg, msg, ...]`. MCP Inspector's export.
3. **Wrapped object** — `{"messages": [...], "metadata": {...}}`.
   Some tooling wraps messages with session-level metadata.

## What's captured

| MCP element | Shadow element |
|---|---|
| `tools/list` response | `metadata.payload.tools` |
| `tools/call` request | `chat_request` + `chat_response` (with `tool_use` block) |
| Tool result | `tool_result` content block (Anthropic-shape) |
| MCP error response | `tool_result` with `is_error: true` |
| Orphan request (no response — disconnect, crash) | `chat_response` with only the `tool_use` block |

## What's NOT captured

- **LLM completions**. MCP is the *tool* protocol, not the LLM
  protocol; the model's actual request/response lives outside the
  MCP session. Semantic / verbosity / safety axes will show zero on
  MCP-imported traces. Honest scoping.

## Real-world example

See `examples/mcp-session/` in the repo for a committed baseline +
candidate JSONL with a Customer Support agent. The candidate silently
renames `customer_id` → `cid` and drops the confirmation step.
Shadow's trajectory axis catches both.
