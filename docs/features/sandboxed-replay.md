# Sandboxed deterministic replay

Shadow's diff and policy machinery answers "what changed?" The
sandboxed-replay engine answers "what would the candidate do, end-to-end,
without actually doing it?" The candidate's LLM picks tools, the engine
dispatches them, results feed back, the loop runs to completion. No real
network calls. No real database writes. No real charges. The output is an
ordinary `.agentlog` so every existing Shadow command (`diff`,
`diff --policy`, `mine`, `mcp-serve`, `bisect`) reads it without changes.

This is the piece that turns "behavior diff CI" into "shadow deployment."

## When to use it

- **Confirming a bisect attribution.** `shadow bisect` says "the model
  swap is 78% of the latency regression, estimated." The hedged
  language is honest because attribution is correlational. To prove
  it, swap one variable at a time with `replace_tool_result` /
  `replace_tool_args` / `branch_at_turn` and re-diff.
- **Pre-merge sanity check on agent-loop changes.** Your PR changes
  the system prompt; the candidate might pick different tools.
  `shadow replay --agent-loop` drives the loop forward against the
  candidate's prompt and produces a candidate trace you can diff
  against baseline.
- **Reproducing production-only bugs in CI.** A candidate trace where
  the user's real tools run under sandbox: same code paths, same
  failure modes, no real side effects.

## Three backends

| Backend | What it does | When to pick |
|---|---|---|
| `ReplayToolBackend` | Indexes baseline `tool_result` records by `(tool_name, canonical_args_hash)`. Returns recorded results. | Default. Deterministic, free, no API keys. |
| `SandboxedToolBackend` | Wraps your real tool functions; blocks network / subprocess / fs writes. | When you care whether the candidate's *new* tool calls succeed. |
| `StubToolBackend` | Returns a deterministic placeholder for every call. | Tests, sanity checks. |

All three implement the same `ToolBackend` protocol:

```python
class ToolBackend(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult: ...
    @property
    def id(self) -> str: ...
```

## Novel-call policies

When the candidate calls a tool the baseline never recorded,
`ReplayToolBackend` consults a `NovelCallPolicy`. Four ship:

- `StrictPolicy` — raises `ShadowBackendError`. The default for CI:
  a tool the baseline didn't record is a behavioral regression by
  definition.
- `StubPolicy` — returns a placeholder result. Lets the loop continue
  so you can see what the candidate would do *given* a stub.
- `FuzzyMatchPolicy` — finds the nearest same-tool baseline call by
  Jaccard distance over arg keys. Useful when arg values shifted but
  intent is the same.
- `DelegatePolicy` — calls a user-supplied async function. Use this
  to bridge to a sandboxed real backend or your own mocking server.

Implementing your own is a single async method.

## CLI

```bash
shadow replay candidate.yaml \
  --baseline baseline.agentlog \
  --agent-loop \
  --tool-backend replay \
  --novel-tool-policy stub \
  --max-turns 32 \
  --output candidate.agentlog
```

The output is a normal `.agentlog`. Pipe it into `shadow diff
baseline.agentlog candidate.agentlog` like any other trace.

## Programmatic API

The CLI flag `--tool-backend sandbox` deliberately errors with
guidance, because the sandbox needs your tool function
implementations and those don't fit on a CLI flag. From Python:

```python
from shadow.replay_loop import run_agent_loop_replay
from shadow.tools.sandbox import SandboxedToolBackend
from shadow.llm import MockLLM
from shadow import _core

baseline = _core.parse_agentlog(open("baseline.agentlog", "rb").read())

async def search_kb(args: dict) -> str:
    """Your real tool function. Sandbox patches socket / subprocess /
    fs writes during execution."""
    return await my_kb_client.search(args["query"])

sandbox = SandboxedToolBackend(
    tool_registry={"search_kb": search_kb},
    block_network=True,
    block_subprocess=True,
    freeze_time=datetime(2026, 4, 25, tzinfo=UTC),
)

trace, summary = await run_agent_loop_replay(
    baseline,
    llm_backend=MockLLM.from_trace(baseline),
    tool_backend=sandbox,
)
```

## Counterfactual primitives

For surgical "what-if" questions, three helpers in
`shadow.counterfactual_loop`:

```python
from shadow.counterfactual_loop import (
    branch_at_turn,
    replace_tool_args,
    replace_tool_result,
)

# What would the agent do if this tool had returned X?
result = await replace_tool_result(
    baseline,
    tool_call_id="call_abc",
    new_output="<no results found>",
)

# What if it had been called with these args instead?
result = await replace_tool_args(
    baseline,
    tool_call_id="call_abc",
    new_arguments={"query": "python", "limit": 25},
)

# Pin turns 0..N from baseline, drive the loop forward from N+1.
result = await branch_at_turn(
    baseline,
    turn=5,
    llm_backend=live_llm,
    tool_backend=sandbox,
)
```

Each returns a `CounterfactualLoopResult` carrying the new trace,
agent-loop summary, and an `override` dict describing the
substitution so renderers can label the comparison "baseline vs.
counterfactual where ...".

## What's in the sandbox

`SandboxedToolBackend` is best-effort isolation for replay
determinism, **not a security boundary**. Patches:

- **Network**: `socket.connect`, `socket.connect_ex` (covers httpx /
  requests / aiohttp / http.client transitively).
- **Subprocess**: `subprocess.Popen`, `subprocess.run`, `os.system`,
  `os.execvp`.
- **Filesystem writes**: opens with mode containing `w`, `a`, `+`,
  or `x` are redirected into a tempdir. Read-mode opens pass through
  so config files keep working.
- **Time** (opt-in): `time.time` + `datetime.utcnow` pinned to a
  fixed instant.

A blocked operation surfaces as a `tool_result` record with
`is_error=True` and an actionable message. Patches restore after
each call so the surrounding test runner isn't affected.

## Determinism guarantees

- Same baseline + same backends + same config → byte-identical
  output. Shadow's content-addressed envelopes don't carry
  wall-clock timestamps in the hash, so re-runs match.
- The replay engine never mutates its inputs.
- A `max_turns` cap (default 32) bounds runaway agents — exceeded,
  the engine emits an `error` record with `code=loop_max_exceeded`
  and stops the session cleanly.

## Limitations

- The agent-loop engine is Python-only. The Rust core handles the
  classic copy-through replay; the agent-driving variant lives in
  Python because user tool functions live in Python.
- `SandboxedToolBackend` patches the ambient runtime, not the tool
  function source. Pure-Python computation, in-memory state, and
  arbitrary recursion all still work — those are the agent's
  business.
- Streaming responses aren't yet driven through the agent-loop
  engine; the existing classic `run_replay` handles streaming
  correctly. v1.7 will unify.
