# Harness events

Every agent run involves harness-side activity that isn't part of the model's response — retries, rate-limit backoffs, model fallbacks, context trims, cache hits, guardrail triggers, budget cuts, stream interrupts, tool lifecycle events. These shape the production behavior of the agent and are usually invisible to standard chat traces.

The `harness_event` record kind captures them in a single line per event.

## Taxonomy

A closed taxonomy of nine categories so a diff renderer can compare apples to apples:

| Category | Meaning |
|---|---|
| `retry` | The harness retried a failed call |
| `rate_limit` | A 429 or upstream rate-limit signal |
| `model_switch` | Routed to a different model (cost, fallback, A/B) |
| `context_trim` | Tokens dropped to fit the window |
| `cache` | Prompt-cache hit or fill |
| `guardrail` | A guardrail (Bedrock / Lakera / Llama Guard / NeMo) fired |
| `budget` | A budget cap (cost, tokens, time) was hit |
| `stream_interrupt` | The streaming response was cut short |
| `tool_lifecycle` | Tool registered, deregistered, or hot-swapped |

Each event also carries a `severity` (`info` / `warning` / `error` / `fatal`) and a free-form `reason` string.

## Recording

```python
from shadow.sdk import Session
from shadow.v02_records import record_harness_event

with Session(output_path="trace.agentlog") as s:
    record_harness_event(
        s,
        category="retry",
        severity="warning",
        reason="anthropic 503, retry 1/3",
    )
```

`name` is optional and defaults to the empty string; use it to subdivide a category (e.g. `category="cache"`, `name="prompt_cache_hit"`).

## Diff

`shadow diff --harness-diff` renders a per-(category, name) diff, separating regressions (candidate has more) from fixes (candidate has fewer). Within each group, entries are sorted by severity descending then by absolute count delta descending — so the most severe new event appears first.

```text
harness events: 2 regression(s), 1 fix(es), 0 unchanged

regressions (candidate has more):
  🔴 rate_limit.: 1 → 3 (+2) first at pair 0
  🟠 retry.: 2 → 4 (+2) first at pair 0

fixes (candidate has fewer):
  ✓ context_trim.: 2 → 0 (-2)
```

A markdown variant is emitted for PR comments (two tables — regressions and fixes — with severity emoji).
