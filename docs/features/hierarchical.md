# Hierarchical diff

Shadow's reports sit at four levels of granularity:

| Level | What it covers | Already in v1.0? |
|---|---|---|
| **trace** | Whole `.agentlog` — nine-axis table | ✅ |
| **session** | One user-facing conversation (between metadata records) | ✅ |
| **turn** | One chat request/response pair | ✅ (drill-down) |
| **span** | One content block (text, tool_use, tool_result) within a turn | ✅ |
| token | Sub-token deltas | deferred to v1.1+ |

## Session-level

```bash
shadow diff baseline.agentlog candidate.agentlog --hierarchical
```

Prints a worst-severity-per-session rollup after the nine-axis table:

```
Hierarchical diff — 3 session(s):
  ✓  session #0: 5 pair(s), worst severity none
  ✗  session #1: 4 pair(s), worst severity severe
  !  session #2: 3 pair(s), worst severity moderate
```

Useful when your trace contains 10 conversations and only 2 regressed
— the flat trace-level table would dilute the signal.

## Span-level

Programmatic access:

```python
from shadow.hierarchical import span_diff

spans = span_diff(baseline_response_payload, candidate_response_payload)
for s in spans:
    print(f"{s.kind}: {s.summary}")
```

Span kinds:

- `text_block_changed` — text drifted (reports char-shingle similarity)
- `tool_use_added` / `tool_use_removed`
- `tool_use_args_changed` — same tool, different argument keys/values
- `tool_result_changed` — `is_error` flip or content differ
- `stop_reason_changed`
- `block_type_changed` — type swap at same index

## Alignment

- **≤ 5 blocks per side** — greedy index alignment (optimal and cheap)
- **> 5 blocks** — Needleman-Wunsch alignment with cost model that
  prefers `add + remove` over `block_type_changed` when types
  mismatch. Catches the real case where an agent has 20+ tool calls
  per turn and the candidate inserts one in the middle — greedy
  would misreport every downstream block as `block_type_changed`.

Verified with a dedicated test that drops block #10 of a 20-block
response; NW alignment correctly reports a single `tool_use_removed`,
zero cascaded changes.
