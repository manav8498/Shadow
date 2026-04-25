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

`ReplayClientSession(index, strict=False)` is the default. On a missing key it returns `None` so the caller's error path runs. Set `strict=True` to raise `MCPMissingRecordingError` instead — recommended in CI when you want a candidate that calls something not in the baseline to fail loudly.

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
