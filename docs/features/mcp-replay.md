# MCP-native replay

Shadow can replay an MCP (Model Context Protocol) session deterministically by intercepting the client at the protocol layer, not the LLM layer. Useful for testing an agent that uses MCP tools without contacting the real MCP server during replay.

## How it works

`shadow.mcp_replay.RecordingIndex` is a content-addressed lookup keyed on `(method, canonical(params))`:

- `canonicalize_params` produces a deterministic JSON encoding (sorted keys, no whitespace, `ensure_ascii=False` so unicode URIs round-trip).
- Repeated calls with identical params return responses in recorded order, then fall back to the last recorded response when the recording is exhausted.
- `unconsumed_keys()` surfaces drift — keys that were recorded but never replayed (often a sign the candidate skipped a baseline step).

```python
from shadow.mcp_replay import RecordingIndex, ReplayClientSession, index_from_imported_mcp_records

# Build the index from a recorded MCP session.
records = [...]  # output of `shadow import --format mcp`
index = index_from_imported_mcp_records(records)

# Drop-in replacement for mcp.ClientSession during replay.
session = ReplayClientSession(index)
result = await session.call_tool("search", arguments={"q": "foo"})
```

## Strict vs non-strict

`ReplayClientSession(index)` defaults to `strict=True`. Strict mode treats two situations as drift and raises `MCPCallNotRecorded`:

1. The candidate calls a `(method, params)` pair that wasn't in the recording at all.
2. The candidate calls a recorded `(method, params)` pair more times than the baseline did (over-consumption).

Non-strict mode (`strict=False`) is the historical permissive behavior: case 1 returns `None` so the caller's null-check path runs, and case 2 reuses the last recorded response so chatty agents don't crash. Use it when you're recording-once-replaying-many and the call counts shouldn't matter; otherwise leave strict on.

Both modes also surface left-over recordings via `index.unconsumed_keys()` — call this after the run to detect "the candidate skipped a baseline step."

## Surface

The replay session implements:

- `call_tool(name, arguments)`
- `read_resource(uri)`
- `list_tools()`, `list_resources()`, `list_prompts()`
- `get_prompt(name, arguments)`
- `initialize()`

Sync and async variants both. The index itself is sync-only.

## Aligning with SEP-1287

The recording shape lines up with [SEP-1287](https://github.com/modelcontextprotocol/specification/issues/1287), the in-flight MCP "deterministic transport" proposal — recordings produced today should interoperate with what eventually ships there.
