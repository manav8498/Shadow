# MCP-session example

A realistic MCP (Model Context Protocol) session trace + a
regressed candidate demonstrating Shadow's MCP importer.

## The scenario

A customer-support agent uses an MCP server that exposes two tools:
`search_orders` and `refund_order`. The session log captures every
JSON-RPC 2.0 message exchanged between the client and the server.

### Baseline (`fixtures/baseline.mcp.jsonl`)

Agent behaves correctly:
- Calls `search_orders(customer_id="C42")` first
- Then `refund_order(order_id="ORD-123", amount_usd=30.0)` with confirmation

### Candidate (`fixtures/candidate.mcp.jsonl`)

A PR "cleanup" silently renames `customer_id` → `cid` and drops the
confirmation check. Our trajectory axis catches both.

## Run it

```bash
# Convert both MCP logs to.agentlog:
shadow import --format mcp fixtures/baseline.mcp.jsonl  -o baseline.agentlog
shadow import --format mcp fixtures/candidate.mcp.jsonl -o candidate.agentlog

# Diff:
shadow diff baseline.agentlog candidate.agentlog
```

You'll see:
- **trajectory** axis: severe (arg rename + call sequence change)
- **first-divergence**: structural_drift at turn 0 (renamed arg)
- Other axes blank (MCP logs don't capture LLM responses)

See `shadow.importers.mcp` for the full round-trip shape.
