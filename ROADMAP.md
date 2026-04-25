# Roadmap

What's shipping and what we're working on. Open an issue if you want to see something added or moved up.

## Shipping today (v1.6.x)

- Content-addressed `.agentlog` trace format with sharded blob store and SQLite index
- Nine-axis behavioural diff with bootstrap 95% confidence intervals
- First-divergence detection and causal bisection (LASSO over corners with Meinshausen-Bühlmann stability selection, hedged renderer)
- Live replay backends for Anthropic and OpenAI, plus a deterministic mock and a positional mock
- Sandboxed deterministic agent-loop replay (`shadow.replay_loop`): drives the candidate's full agent loop forward against an LLM backend and a tool backend, with `max_turns` safety cap and a structured `AgentLoopSummary`. Best-effort isolation for replay determinism via `SandboxedToolBackend` (blocks `socket.connect`, `subprocess.run` / `Popen` / `os.system` / `os.execvp`, redirects write-mode `open()` to a tempdir, optional `freeze_time`). Not a security boundary.
- Tool backends: `ReplayToolBackend` (recorded-result lookup), `SandboxedToolBackend` (real tool fns under isolation), `StubToolBackend` (deterministic placeholders)
- Novel-call policies for tool calls the baseline never recorded: `StrictPolicy`, `StubPolicy`, `FuzzyMatchPolicy`, `DelegatePolicy`
- Counterfactual primitives over the agent loop: `branch_at_turn`, `replace_tool_result`, `replace_tool_args` (each preserves the baseline prefix verbatim with content-addressed ids and drives forward from the pivot)
- Behaviour-policy rules with conditional `when:` gating: `must_call_before`, `must_call_once`, `no_call`, `max_turns`, `required_stop_reason`, `max_total_tokens`, `must_include_text`, `forbidden_text`. Ten condition operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `not_contains`) over dotted paths into request / response / model / stop_reason
- `shadow diff --fail-on {minor,moderate,severe}` exits non-zero on regressions, so the GitHub Action can gate merges instead of just commenting
- Auto-instrumentation for the Anthropic and OpenAI SDKs (Python and TypeScript, including the OpenAI Responses API and streaming)
- Framework adapters: LangGraph / LangChain (`shadow-diff[langgraph]`), CrewAI (`shadow-diff[crewai]`), AG2 (`shadow-diff[ag2]`)
- Ten built-in judges, including a rubric-driven `LlmJudge`
- Eight importers: Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP (GenAI semconv v1.40), MCP, A2A, Vercel AI SDK, PydanticAI
- OTel GenAI exporter
- Hierarchical diff at six levels: trace, session, turn, span, token, policy
- Token-level distribution summaries and policy-rule overlays
- LLM-assisted prescriptive fixes grounded on deterministic recommendations
- Production trace mining (`shadow mine`) — clusters real traces by tool sequence, stop reason, response length, and latency, then surfaces representative regression cases
- MCP server (`shadow mcp-serve`) exposing diff, policy check, token diff, schema watch, and summary as MCP tools
- SOC 2-oriented access log, CycloneDX 1.5 SBOMs, cosign keyless signing
- Ubuntu, macOS, and Windows CI across Python 3.11, 3.12, 3.13
- PyPI Trusted Publisher release pipeline; published as `shadow-diff`
- Docs site at [manav8498.github.io/Shadow](https://manav8498.github.io/Shadow/)

## What's next

### Streaming replay

Record and replay inter-chunk timings so a candidate that stalls between chunks looks different from one that bursts them. Needs an `.agentlog` v0.2 chunk-record addition and parser updates.

### Multimodal traces

Images and audio in and out, with a cross-modal semantic axis. Needs blob addressing in the spec and perceptual-similarity work.

### Harness-diff instrumentation

Capture retries, tool-ordering, and context-trim events as first-class diff dimensions instead of burying them in the payload.

### MCP-native replay

We import MCP session logs today and serve diff/policy/token-diff/schema-watch/summary over MCP. Next: protocol-level interception so MCP tool invocations replay deterministically without re-running the MCP server.

### Runtime policy enforcement

`shadow diff --policy` checks a YAML policy after the fact. The runtime version: enforce the same rules at recording time, block a run that would violate one.

### Richer behaviour contracts

The `when:` gate plus eight rule kinds covers most production agents today. Next: `must_match_json_schema` for structured-output assertions, stateful rules that reason across turns (e.g. "if the agent confirmed an amount on turn N, it must not change the amount on turn N+1"), and retrieved-context assertions for RAG agents.

### Agent Behaviour Bill of Materials (ABOM)

Signed release artefact that captures the prompt version, model, tool set, memory policy, retrieval config, contract pass/fail counts, and the regression suite hash. Same idea as a software SBOM, applied to agent behaviour, so an enterprise can prove what behaviour shipped with a given release.

## Not on the roadmap

A few things we are not building:

- **Hosted Shadow.** The whole point is that Shadow lives in your repo and your CI, not in someone else's dashboard. Pair Shadow with Langfuse, Braintrust, or LangSmith if you want a UI.
- **Bundled domain rubrics** (ESI for medical, PCI-DSS for payments). Those are what the `Judge` axis is for. We ship example rubrics in `examples/judges/`.
- **A graphical trace viewer.** Text tools for text formats. `jq < trace.agentlog` is a fine viewer.
- **A security-grade sandbox.** `SandboxedToolBackend` is best-effort isolation for replay determinism, not a hardened security boundary. Run untrusted code in a real VM or container.

## How to influence this

Open an issue with a real use case: *"We hit X with Shadow on our production traffic and Y would have saved us"* is the most useful feedback. Draft PRs welcome for anything on this page. If you want to own a roadmap item, say so in an issue first so we do not duplicate work.
