# Hierarchical diff

Shadow's reports sit at six levels of granularity. All six ship today:

| Level | What it covers | Shipped in |
|---|---|---|
| **trace** | Whole `.agentlog`, nine-axis table | v1.0 |
| **session** | One user-facing conversation (between metadata records) | v1.0 |
| **turn** | One chat request/response pair | v1.0 (drill-down) |
| **span** | One content block (text, tool_use, tool_result) within a turn | v1.0 |
| **token** | Per-pair input / output / thinking token distribution | v1.2 |
| **policy** | Declarative YAML rules checked against the trace | v1.2 |

## Session-level

```bash
shadow diff baseline.agentlog candidate.agentlog --hierarchical
```

Prints a worst-severity-per-session rollup after the nine-axis table:

```
Hierarchical diff, 3 session(s):
  ✓  session #0: 5 pair(s), worst severity none
  ✗  session #1: 4 pair(s), worst severity severe
  !  session #2: 3 pair(s), worst severity moderate
```

Useful when your trace contains 10 conversations and only 2 regressed
- the flat trace-level table would dilute the signal.

## Span-level

Programmatic access:

```python
from shadow.hierarchical import span_diff

spans = span_diff(baseline_response_payload, candidate_response_payload)
for s in spans:
    print(f"{s.kind}: {s.summary}")
```

Span kinds:

- `text_block_changed`, text drifted (reports char-shingle similarity)
- `tool_use_added` / `tool_use_removed`
- `tool_use_args_changed`, same tool, different argument keys/values
- `tool_result_changed`, `is_error` flip or content differ
- `stop_reason_changed`
- `block_type_changed`, type swap at same index

### Alignment

- **≤ 5 blocks per side**, greedy index alignment (optimal and cheap)
- **> 5 blocks**, Needleman-Wunsch alignment with cost model that
  prefers `add + remove` over `block_type_changed` when types
  mismatch. Catches the real case where an agent has 20+ tool calls
  per turn and the candidate inserts one in the middle, greedy
  would misreport every downstream block as `block_type_changed`.

Verified with a dedicated test that drops block #10 of a 20-block
response; NW alignment correctly reports a single `tool_use_removed`,
zero cascaded changes.

## Token-level (v1.2)

```bash
shadow diff baseline.agentlog candidate.agentlog --token-diff
```

Per-dimension distribution summaries (median, p25, p75, p95, max, total)
for `input_tokens` / `output_tokens` / `thinking_tokens`, plus a ranked
list of the top-k worst per-pair deltas:

```
Token-level diff, 10 pair(s):
  input_tokens       baseline median 100.0  p95 118.0  →  candidate median 100.0  p95 118.0  shift +0.00%
  output_tokens      baseline median  40.0  p95  48.0  →  candidate median  60.0  p95  72.0  shift +50.00%
  thinking_tokens    baseline median   0.0  p95   0.0  →  candidate median   5.0  p95   5.0  shift +inf
  worst pairs (by absolute token delta):
    · turn #3: input +0, output +400, thinking +5
    · turn #7: input +0, output +150, thinking +5
```

Programmatic access: `shadow.hierarchical.token_diff(baseline, candidate)` returns
a `TokenDiff` dataclass with per-dimension `TokenDistSummary` +
per-pair `TokenPairDelta` entries.

Scale-tested to 10k pair traces in under 500ms.

## Policy-level (v1.2)

A declarative YAML overlay of rules checked against both traces:

```yaml
# policy.yaml
rules:
  - id: backup-before-migrate
    kind: must_call_before
    params: {first: backup_database, then: run_migration}
    severity: error
  - id: no-ssn-leak
    kind: forbidden_text
    params: {text: "SSN:"}
    severity: error
  - id: finish-cleanly
    kind: required_stop_reason
    params: {allowed: [end_turn, tool_use]}
    severity: error
```

```bash
shadow diff baseline.agentlog candidate.agentlog --policy policy.yaml
```

Output classifies every rule's violations as **regressions** (new in
candidate) vs **fixes** (cleared in candidate):

```
Policy diff, 0 baseline violation(s), 2 candidate violation(s)
  regressions (2):
    ✗ [critical] no-ssn-leak @ turn #2: forbidden text 'SSN:' present in response
    ✗ [critical] backup-before-migrate @ turn #0: `backup_database` must be called before `run_migration`
```

### All 8 supported rule kinds

| Kind | Params | Checks |
|---|---|---|
| `must_call_before` | `first`, `then` (tool names) | `first` must be invoked before `then` when `then` appears |
| `must_call_once` | `tool` | exactly one invocation of `tool` |
| `no_call` | `tool` | tool must never be invoked |
| `max_turns` | `limit` (int) | trace has ≤ `limit` chat_response records |
| `required_stop_reason` | `allowed` (list of strings) | final stop_reason must be in the allowed set |
| `max_total_tokens` | `limit` (int) | total input+output+thinking tokens ≤ `limit` |
| `must_include_text` | `text` | at least one response body contains `text` |
| `forbidden_text` | `text` | no response body contains `text` |

Programmatic: `shadow.hierarchical.policy_diff(baseline, candidate, rules)`.
