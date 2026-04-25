# Behavior policy

The diff tells you what changed. A policy tells you what is not allowed to change. A policy file is a YAML or JSON list of rules; `shadow diff --policy <file>` evaluates them against both traces and reports regressions (new violations the candidate introduced) and fixes (violations the baseline had that the candidate cleared).

## Rule kinds

Nine kinds ship today:

| Kind | What it asserts |
|---|---|
| `must_call_before` | Tool A must be called before tool B (whenever both are present in a session) |
| `must_call_once` | A specific tool must be called exactly once per session |
| `no_call` | A specific tool must never be called |
| `max_turns` | A session must not exceed N chat round-trips |
| `required_stop_reason` | Every chat response must end with one of the allowed stop reasons |
| `max_total_tokens` | Total token budget per session must stay under a cap |
| `must_include_text` | A required string must appear in at least one response |
| `forbidden_text` | A specific string must never appear in any response |
| `must_match_json_schema` | Every response's text content must parse as JSON and validate against a JSON Schema |

## Conditional rules — `when:`

Every rule can carry a `when:` clause that gates it on a list of field-path conditions. The rule fires only on the subset of pairs (request/response) where every condition holds. Multiple conditions AND together. Missing paths quietly don't match (the rule is skipped on that pair) instead of crashing the whole check.

```yaml
rules:
  - id: confirm-large-refunds
    kind: must_call_before
    params:
      first: confirm_refund_amount
      second: issue_refund
    when:
      - { path: "request.params.amount", op: ">", value: 500 }
      - { path: "request.model", op: "==", value: "gpt-4.1" }
    severity: error
```

Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `not_contains`. Paths are dotted into the per-pair context: `request.*` (model, messages, params, tools), `response.*` (content, stop_reason, latency_ms, usage), plus aliases `model` (== `request.model`) and `stop_reason` (== `response.stop_reason`).

## Structured-output assertions

`must_match_json_schema` accepts either an inline `schema:` dict or a `schema_path:` to a JSON Schema file. Mismatches name the offending dotted path so reviewers see exactly which field broke.

```yaml
rules:
  - id: structured-output
    kind: must_match_json_schema
    params:
      schema_path: schemas/refund_decision.schema.json
    severity: error
```

NaN, Infinity, and -Infinity literals are rejected because they aren't valid JSON per RFC 8259, even though Python's `json.loads` accepts them as a CPython extension. Downstream consumers (browsers, other-language parsers, strict JSON parsers) choke on them, so the rule treats them as a contract violation.

## Severity and the merge gate

Every rule carries a `severity` field: `error`, `warning`, or `info`. Combined with `shadow diff --fail-on`, you can gate a PR merge on the worst signal in the report:

```bash
shadow diff baseline.agentlog candidate.agentlog \
  --policy shadow-policy.yaml \
  --fail-on severe
```

The diff and policy summary still print first; the gate runs as a separate step. A blocked PR always has the explanation visible in the comment.

`--fail-on` levels map onto axis severity directly: `none < minor < moderate < severe`. Policy violations map as `info → minor`, `warning → moderate`, `error → severe`. The gate trips when the worst signal across both surfaces hits the threshold.

## Scope

Rules default to `scope: trace` — evaluated once per whole trace. `scope: session` evaluates each rule independently per user-initiated session, which is almost always what you want on multi-ticket production traces (ten refund conversations bundled into one .agentlog should be ten separate evaluations, not one merged one).

```yaml
rules:
  - id: per-conversation-budget
    kind: max_total_tokens
    params: { n: 4000 }
    scope: session
    severity: warning
```
