# MCP server

Run Shadow as a [Model Context Protocol](https://modelcontextprotocol.io/) server so any MCP-aware agentic CLI (Claude Desktop, Claude Code, Cursor, Zed, Windsurf) can invoke Shadow as a tool.

This is the reverse of [the MCP importer](mcp.md): the importer ingests MCP session logs into Shadow; this page covers Shadow exposing its analyses *as* MCP tools.

## Install

```bash
pip install 'shadow-diff[mcp]'
```

## Run

```bash
shadow mcp-serve
```

The server speaks JSON-RPC 2.0 over stdio. Wire it into your client's MCP config:

```json
{
  "mcpServers": {
    "shadow": {
      "command": "shadow",
      "args": ["mcp-serve"]
    }
  }
}
```

## Tools exposed

Seven tools, each one a thin wrapper over the equivalent CLI command:

| Tool | Purpose |
|---|---|
| `shadow_diff` | Nine-axis behavioral diff between two `.agentlog` files. Pass `policy_path` to also enforce a YAML policy. |
| `shadow_check_policy` | Check both traces against a YAML/JSON policy. Returns regressions, fixes, and per-pair violations. |
| `shadow_token_diff` | Per-dimension token distribution summary (input / output / thinking) plus top-k worst per-pair deltas. |
| `shadow_schema_watch` | Classify tool-schema changes between two configs. Tiered into breaking / risky / additive / neutral. |
| `shadow_summarise` | Plain-English summary of a saved DiffReport JSON. Deterministic, no LLM call. |
| `shadow_certify` | Generate an [Agent Behavior Certificate](certificate.md) (ABOM) for a release trace. |
| `shadow_verify_cert` | Verify a certificate's content-addressed `cert_id` matches the body. |

A typical agentic-CLI session looks like:

```
> Use shadow to compare these two trace files and tell me if it's safe to merge.

shadow_diff(
    baseline="./baseline.agentlog",
    candidate="./candidate.agentlog",
    policy_path="./shadow-policy.yaml"
)
```

The host agent gets back a JSON object with rows (one per axis), recommendations, the first-divergence record, and (when `policy_path` is supplied) policy regressions. The host can summarise, render a table, or use the signal to decide whether to proceed with a merge.

## What it doesn't do

- **No live recording.** This server exposes Shadow's analyses, not its capture surface. To record agent traces, use [the SDK's `Session`](../quickstart/record.md) or one of the [framework adapters](adapters.md) — those produce the `.agentlog` files this server then operates on.
- **No long-lived state.** Every tool call reads the trace files passed in by the client. There's no caching, no connection pooling, no shared context between calls. That keeps the server safe to drop in to any client without thinking about cleanup.

## Related

- [MCP importer](mcp.md) — the reverse direction, ingesting MCP traces into `.agentlog`.
- [`shadow certify`](certificate.md) — the ABOM workflow that the `shadow_certify` MCP tool exposes.
- [`shadow diff` policy rules](policy.md) — the policy language enforced by `shadow_check_policy`.
