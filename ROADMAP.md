# Roadmap

What's shipping and what we're working on. Open an issue if you want to see something added or moved up.

## Shipping today (v2.0.x)

- Content-addressed `.agentlog` trace format with sharded blob store and SQLite index
- Nine-axis behavioural diff with bootstrap 95% confidence intervals
- First-divergence detection and causal bisection (LASSO over corners with Meinshausen-Bühlmann stability selection, hedged renderer)
- Live replay backends for Anthropic and OpenAI, plus a deterministic mock and a positional mock
- Sandboxed deterministic agent-loop replay (`shadow.replay_loop`): drives the candidate's full agent loop forward against an LLM backend and a tool backend, with `max_turns` safety cap and a structured `AgentLoopSummary`. Best-effort isolation for replay determinism via `SandboxedToolBackend` (blocks `socket.connect`, `subprocess.run` / `Popen` / `os.system` / `os.execvp`, redirects write-mode `open()` to a tempdir, optional `freeze_time`). Not a security boundary.
- Tool backends: `ReplayToolBackend` (recorded-result lookup), `SandboxedToolBackend` (real tool fns under isolation), `StubToolBackend` (deterministic placeholders)
- Novel-call policies for tool calls the baseline never recorded: `StrictPolicy`, `StubPolicy`, `FuzzyMatchPolicy`, `DelegatePolicy`
- Counterfactual primitives over the agent loop: `branch_at_turn`, `replace_tool_result`, `replace_tool_args` (each preserves the baseline prefix verbatim with content-addressed ids and drives forward from the pivot)
- Behaviour-policy rules with conditional `when:` gating: `must_call_before`, `must_call_once`, `no_call`, `max_turns`, `required_stop_reason`, `max_total_tokens`, `must_include_text`, `forbidden_text`, `must_match_json_schema`, plus stateful and RAG-aware kinds `must_remain_consistent`, `must_followup`, `must_be_grounded`. Ten condition operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `not_contains`) over dotted paths into request / response / model / stop_reason
- Runtime policy enforcement via `shadow.policy_runtime.EnforcedSession` and `PolicyEnforcer`. Three modes: `replace` (default — swap the offending response for a refusal payload), `raise` (throw `PolicyViolationError`), `warn` (log only). The enforcer is incremental: a whole-trace rule fires once when crossed, not once per recorded record
- **Pre-tool-call enforcement** at two layers:
  - `shadow.policy_runtime.wrap_tools` and `Session.wrap_tools` for explicit tool-registry wrapping. Synthesises a candidate `tool_call` record before each tool fires, probes the enforcer non-mutatingly, and either runs the underlying function (allow), raises (`raise`), returns a placeholder (`replace`), or runs anyway with a warning (`warn`).
  - **Auto-instrument-layer pre-dispatch** for OpenAI / Anthropic-driven agents that don't wrap their own tools. When the active session is an `EnforcedSession`, the auto-instrument wrapper probes every `tool_use` block in a non-streaming response BEFORE the response is returned to user code; violating responses raise `PolicyViolationError` at the wrapped `.create` call. No code changes to the user's tool functions.
  - Both layers catch `no_call`, `must_call_before`, `must_call_once` AT the dispatch site for dangerous tools (`issue_refund`, `send_email`, `execute_sql`, `delete_user`).
- Agent Behavior Certificate (ABOM) via `shadow certify` / `shadow verify-cert`: a content-addressed JSON release artefact capturing the trace id, models, prompt hashes, tool schema hashes, policy hash, and an optional baseline-vs-candidate regression-suite rollup. Self-verifying — `verify-cert` exits non-zero on tamper, so it can gate a release pipeline.
- Cosign / sigstore keyless signing for certificates via `shadow certify --sign` and `shadow verify-cert --verify-signature` (optional `[sign]` extra). Writes a sidecar sigstore Bundle containing the signature, Fulcio-issued signing certificate, and Rekor transparency-log entry. Verification binds to a specific signer identity (workflow URL or email) so leaked Bundles signed by another identity fail.
- `shadow diff --fail-on {minor,moderate,severe}` exits non-zero on regressions, so the GitHub Action can gate merges instead of just commenting
- Auto-instrumentation for the Anthropic and OpenAI SDKs. Python: covers Anthropic Messages, OpenAI Chat Completions, OpenAI Responses API, plus streaming aggregation. TypeScript: covers non-streaming Anthropic and OpenAI calls plus streaming aggregation (each streamed call lands as a single record with the assembled response — interleaved tool-call argument deltas reassemble per index, Anthropic content blocks reassemble per `content_block_start`/`content_block_delta`/`content_block_stop` sequence)
- Framework adapters: LangGraph / LangChain (`shadow-diff[langgraph]`), CrewAI (`shadow-diff[crewai]`), AG2 (`shadow-diff[ag2]`)
- Ten built-in judges, including a rubric-driven `LlmJudge`
- Nine importers: Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP (GenAI semconv v1.40), MCP, A2A, Vercel AI SDK, PydanticAI
- OTel GenAI exporter
- Hierarchical diff at six levels: trace, session, turn, span, token, policy
- Token-level distribution summaries and policy-rule overlays
- LLM-assisted prescriptive fixes grounded on deterministic recommendations
- Production trace mining (`shadow mine`) — clusters real traces by tool sequence, stop reason, response length, and latency, then surfaces representative regression cases
- MCP server (`shadow mcp-serve`) exposing seven tools: `shadow_diff`, `shadow_check_policy`, `shadow_token_diff`, `shadow_schema_watch`, `shadow_summarise`, `shadow_certify`, `shadow_verify_cert`
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


## Not on the roadmap

A few things we are not building:

- **Hosted Shadow.** The whole point is that Shadow lives in your repo and your CI, not in someone else's dashboard. Pair Shadow with Langfuse, Braintrust, or LangSmith if you want a UI.
- **Bundled domain rubrics** (ESI for medical, PCI-DSS for payments). Those are what the `Judge` axis is for. We ship example rubrics in `examples/judges/`.
- **A graphical trace viewer.** Text tools for text formats. `jq < trace.agentlog` is a fine viewer.
- **A security-grade sandbox.** `SandboxedToolBackend` is best-effort isolation for replay determinism, not a hardened security boundary. Run untrusted code in a real VM or container.

## How to influence this

Open an issue with a real use case: *"We hit X with Shadow on our production traffic and Y would have saved us"* is the most useful feedback. Draft PRs welcome for anything on this page. If you want to own a roadmap item, say so in an issue first so we do not duplicate work.
