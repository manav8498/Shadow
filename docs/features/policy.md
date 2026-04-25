# Behavior policy

The diff tells you what changed. A policy tells you what is not allowed to change. A policy file is a YAML or JSON list of rules; `shadow diff --policy <file>` evaluates them against both traces and reports regressions (new violations the candidate introduced) and fixes (violations the baseline had that the candidate cleared).

## Rule kinds

Twelve kinds ship today:

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
| `must_remain_consistent` | Once a value is observed at `path`, every later pair where the path resolves must equal it (e.g. "the agent must not change the refund amount after confirming it") |
| `must_followup` | When `trigger` conditions hold in pair N, pair N+1 must satisfy `must` (a tool call by name, or a text-includes substring). A trigger on the final pair is itself a violation |
| `must_be_grounded` | Every response must overlap meaningfully with retrieved chunks at `retrieval_path`. Default threshold is `min_unigram_precision: 0.5` — the standard no-LLM-judge fallback also used by RAGAS, TruLens, DeepEval |

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

## Stateful and RAG-aware rules

Three rule kinds reason across multiple turns or compare against retrieved context.

### `must_remain_consistent`

Once a value is observed at `path` in some pair, every later pair where the same path resolves must equal that anchor. Pairs where the path is absent are skipped — absence is not change, the rule pins consistency *when observed*.

```yaml
rules:
  - id: amount-locked-after-confirmation
    kind: must_remain_consistent
    params: { path: "request.params.amount" }
    severity: error
```

### `must_followup`

When `trigger` conditions hold in pair N, pair N+1 must satisfy `must`. The `must` spec accepts two kinds: `tool_call` (the next response must include a `tool_use` block by that name) and `text_includes` (the next response text must contain the substring). A trigger on the last pair is itself a violation — the obligation could not be satisfied.

```yaml
rules:
  - id: confirm-after-quote
    kind: must_followup
    params:
      trigger:
        - { path: "response.stop_reason", op: "==", value: "tool_use" }
        - { path: "response.content", op: "contains", value: "quote_total" }
      must: { kind: tool_call, tool_name: confirm_with_user }
    severity: error
```

### `must_be_grounded`

Every response must share enough unigrams with retrieved chunks at `retrieval_path` to clear `min_unigram_precision`. Pairs without retrieval at the given path are skipped.

```yaml
rules:
  - id: rag-grounding
    kind: must_be_grounded
    params:
      retrieval_path: "request.metadata.retrieved_chunks"
      min_unigram_precision: 0.5
    severity: error
```

Tokenisation is lowercased + alphanumeric, len ≥ 2 — punctuation and stopwords-of-length-1 don't count. An attacker can't satisfy the rule by emitting only `the , .`.

**What this catches and what it doesn't.** This is *lexical overlap*, not semantic grounding or NLI-backed faithfulness. It catches the obvious failure cases — a response that talks about a totally different topic than the retrieved chunks — and it's the same no-LLM-judge baseline RAGAS / TruLens / DeepEval ship as their cheapest fallback. It does NOT catch:

- semantic-equivalent paraphrase that uses entirely different vocabulary (the response is grounded in meaning, but not in words)
- a response that quotes chunks but draws an unsupported conclusion
- factual claims a chunk *contradicts* (overlap can be high while the claim is wrong)

For deeper grounding (per-claim NLI, sentence-level entailment, LLM-judge faithfulness), pair this rule with the `Judge` axis or with an external faithfulness evaluator. Treat `must_be_grounded` as a cheap CI gate, not a hallucination guarantee.

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
