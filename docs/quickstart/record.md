# Record your own agent

Shadow's `shadow record` wrapper captures an agent's LLM calls into an
`.agentlog` file with **zero code changes** to the agent. This page
shows how it works and when to use the explicit `Session` instead.

## Zero-config path (recommended)

```bash
shadow record -o baseline.agentlog -- python your_agent.py
```

Shadow prepends a `sitecustomize` shim to the child Python's
`PYTHONPATH`. On interpreter startup, the shim reads
`SHADOW_SESSION_OUTPUT` (set by `shadow record`), constructs a
`Session`, and registers an `atexit` handler to flush the trace when
the agent exits.

The Session already monkey-patches `anthropic.Anthropic().messages.create`
and `openai.OpenAI().chat.completions.create` (plus async variants),
so every LLM call your agent makes is captured automatically.

### Useful flags

| Flag | Effect |
|---|---|
| `-o, --output PATH` | Where to write the `.agentlog` (default `recording.agentlog`) |
| `--tags KEY=V,K=V` | Comma-separated tags attached to the trace metadata |
| `--no-auto-instrument` | Skip the sitecustomize shim (use if your agent already opens its own Session) |

### Fail-fast output-path check

`shadow record` probes the output path for writability **before**
spawning the child. If your `-o` argument points at a read-only
directory, the command exits 2 with an actionable error and the child
never runs — so no LLM tokens burn on a recording that can't be saved.

## Explicit Session path

When you need custom tags, a non-default redactor, or nested sessions,
open a `Session` yourself:

```python
from shadow.sdk import Session

with Session(output_path="trace.agentlog", tags={"env": "prod"}):
    client.messages.create(model="claude-sonnet-4-6", messages=[...])
```

The Session context manager handles:

- Redaction (secrets, PII) on request and response payloads
- Monkey-patch install on `__enter__`, restore on `__exit__`
- Streaming / async client variants

## When to use which

| Situation | Use |
|---|---|
| Running an existing agent script as-is | `shadow record` |
| Agent is a long-running service | Explicit `Session` at request boundaries |
| Need per-request tags | Explicit `Session` |
| Multi-tenant recording | Explicit `Session` with per-tenant `output_path` |
| Recording a single script, one-shot | `shadow record` |

## Replaying against a new config

Once you have a `baseline.agentlog`, produce a candidate by replaying
through a new config:

```bash
shadow replay candidate.yaml --baseline baseline.agentlog --backend anthropic
# → writes candidate.agentlog
```

Then diff:

```bash
shadow diff baseline.agentlog candidate.agentlog
```

## Next

- Wire this loop into every PR: [Wire into CI](ci.md)
