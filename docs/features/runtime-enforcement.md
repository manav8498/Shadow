# Runtime policy enforcement

`shadow diff --policy` checks rules against a fully recorded trace. Runtime enforcement runs the same rules incrementally as turns are recorded inside `shadow.sdk.Session`, so a violation can BLOCK or REPLACE the offending response before it propagates downstream.

## Quick start

```python
from shadow.policy_runtime import EnforcedSession, PolicyEnforcer

enforcer = PolicyEnforcer.from_policy_file("shadow-policy.yaml")
with EnforcedSession(enforcer=enforcer, output_path="run.agentlog") as s:
    s.record_chat(request=..., response=...)
```

If the recorded turn introduces a new policy violation, the session swaps the response for a refusal payload (`stop_reason: "policy_blocked"`) by default. The `.agentlog` flushed on context exit is structurally valid — every existing Shadow command (`diff`, `verify-cert`, `mine`, `mcp-serve`) reads it without modification.

## Three modes

`PolicyEnforcer(rules, on_violation=...)` accepts three values:

| Mode | What happens on a new violation |
|---|---|
| `replace` (default) | The chat response is swapped for a refusal payload built by the configured `replacement_builder`. Trace continues, downstream code keeps running. |
| `raise` | `EnforcedSession` raises `PolicyViolationError`. The offending request/response is rolled out of the in-memory record list so the flushed trace ends at the previous turn. |
| `warn` | A `WARNING` is logged on `shadow.policy_runtime`. The trace records the original response unchanged. Useful when you want metrics without behavior changes. |

The mode you pick depends on what your callers expect. Background pipelines that should never deliver a violating response use `replace`. Synchronous flows that already have a fallback path can use `raise`. Observability-only deployments use `warn`.

## Custom replacements

The default `default_replacement_response` builds a refusal payload that preserves structural fields (`model`, `usage`, `latency_ms`) so downstream renderers don't break. You can override it per-enforcer:

```python
def my_refusal(violations, original):
    return {
        **original,
        "content": [{"type": "text", "text": "I can't help with that."}],
        "stop_reason": "policy_blocked",
    }

enforcer = PolicyEnforcer(rules, replacement_builder=my_refusal)
```

Anything you return must be a valid `chat_response` payload (`model`, `content`, `stop_reason`, `latency_ms`, `usage`). Shadow re-canonicalises it and rebuilds the record's content-id, so the trace's content-addressing invariant holds.

## Incremental evaluation

The enforcer is stateful: it tracks violation identities as `(rule_id, pair_index)` — NOT including the human-readable `detail` text. Whole-trace rules like `max_turns` embed a running count in their detail string ("trace has 5 turns; max is 4", then "trace has 6 turns; max is 4", and so on), so a detail-keyed dedup let those rules respam a "new" violation every subsequent turn. The current key fires the rule once on the turn that crosses the threshold and stays silent on later calls — the user gets one notification, not one per recorded record. A new violation introduced on a later turn (e.g. a different pair tripping a different rule) fires fresh.

This means `PolicyEnforcer.evaluate(records)` returns a `Verdict` with only the *delta* since the previous call. Reuse one enforcer instance across the whole session.

## Scope and rule support

Every rule kind in [Behavior policy](policy.md) works at runtime. Some rules are inherently post-hoc — `must_followup` queues an obligation at trigger time and only confirms whether it was met when the next pair lands. The runtime enforcer evaluates obligations at every turn, so an unmet `must_followup` surfaces on the *following* turn that didn't satisfy it.

## Pre-tool-call enforcement (v2.1)

For dangerous tools — `issue_refund`, `send_email`, `execute_sql`, `delete_user`, `deploy_service` — post-response enforcement is too late. The side effect already happened. v2.1 adds `wrap_tools` to move the policy check BEFORE the tool function runs:

```python
from shadow.policy_runtime import EnforcedSession, PolicyEnforcer

enforcer = PolicyEnforcer.from_policy_file("policy.yaml")
with EnforcedSession(enforcer=enforcer, output_path="run.agentlog") as s:
    guarded = s.wrap_tools({
        "issue_refund": issue_refund,
        "delete_user": delete_user,
    })
    # The wrapper evaluates the policy with a synthesised candidate
    # tool_call appended to the trace-so-far. If it would violate, the
    # underlying function is NEVER called.
    result = guarded["delete_user"](user_id="u-42")
```

How it works under the hood:

1. The wrapper synthesises a candidate `tool_call` record (tool name + arguments) and appends it to a copy of `session._records`.
2. It calls `enforcer.probe(records + [candidate])`. **Probe is non-mutating** — a denied probe doesn't leave state behind, so a tool that's repeatedly blocked doesn't pollute the enforcer's `_known` set.
3. On `allow`: the underlying function runs and returns its result.
4. On deny, behaviour follows the enforcer's `on_violation`:
   - `replace`: the wrapper returns a placeholder (default: a string marker; override via `blocked_replacement=`)
   - `raise`: `PolicyViolationError` thrown; the underlying function is never called
   - `warn`: logged and the underlying function runs anyway

Rules that catch tool-sequence violations (`no_call`, `must_call_before`, `must_call_once`) work pre-dispatch automatically. Rules that depend on response text (`must_be_grounded`, `forbidden_text`, `must_match_json_schema`) are still response-side and run via `record_chat`.

The `_extract_tool_call_sequence` helper now reads BOTH `tool_use` blocks inside `chat_response` records (the LLM-emitted shape) AND standalone `tool_call` records (the explicit `Session.record_tool_call` shape, plus pre-dispatch probes). This means `Session.record_tool_call` calls are now first-class to the policy engine — historically they were invisible to `no_call` and the ordering rules.

For frameworks (LangGraph, LangChain, CrewAI) that manage their own session state, `wrap_tools` accepts a `records_provider=` callable instead of a `session=`:

```python
guarded = wrap_tools(tools, enforcer, records_provider=lambda: my_records_list)
```

That gives an integration point without forcing the host code to use Shadow's `Session`.

## Programmatic API without `EnforcedSession`

If you already integrate Shadow with another tracing layer (LangGraph, custom adapter), use `PolicyEnforcer` directly:

```python
from shadow.policy_runtime import PolicyEnforcer

enforcer = PolicyEnforcer.from_policy_file("policy.yaml", on_violation="raise")

# After each turn, hand the enforcer the records list (any list of
# Shadow .agentlog records) and act on the verdict.
verdict = enforcer.evaluate(records)
if not verdict.allow:
    if enforcer.on_violation == "raise":
        # raise yourself, or use replacement
        raise RuntimeError(verdict.reason)
    if verdict.replacement is not None:
        # swap the last record's payload for verdict.replacement
        ...
```

This is the canonical pattern for embedding Shadow into a host pipeline that already manages its own session lifecycle.

## What it doesn't do

- **Auto-intercept of framework-managed tool dispatch.** Pre-tool-call enforcement is shipped (see `wrap_tools` and `Session.wrap_tools` above), but it's opt-in: you wrap a `{name: callable}` registry. If a framework manages dispatch internally and doesn't expose that registry, you need either the matching framework adapter or a manual `evaluate(records_so_far)` call between model response and tool dispatch.
- **Network-level guardrails.** Shadow doesn't sit between your app and the LLM provider; it runs inside the session. Pair Shadow with a network-level guard (Bedrock Guardrails, Lakera, Llama Guard) if you need that.
- **Cross-process state.** Each `PolicyEnforcer` is a per-process object. Distributed enforcement needs an external coordinator.

## Related

- [Behavior policy](policy.md) — the policy YAML format and all 12 rule kinds.
- [Release certificate](certificate.md) — pair runtime enforcement with `shadow certify` to record what behavior actually shipped, post-enforcement.
