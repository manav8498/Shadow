# Roadmap

What's shipping and what we're working on. Open an issue if you want to see something added or moved up.

## Shipping today (v1.2.x)

- Content-addressed `.agentlog` trace format
- Nine-axis behavioural diff with bootstrap confidence intervals
- First-divergence detection and causal bisection (LASSO over corners with Meinshausen-Bühlmann stability selection)
- Live replay backends for Anthropic and OpenAI, plus a deterministic mock
- Partial replay: lock a baseline prefix, replay only the suffix
- Counterfactual replay: swap one config variable at a time
- Auto-instrumentation for the Anthropic and OpenAI SDKs (Python and TypeScript, including the OpenAI Responses API and streaming)
- Ten built-in judges, including a rubric-driven `LlmJudge`
- Eight importers: Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP, MCP, Vercel AI SDK, PydanticAI
- OTel GenAI exporter
- Hierarchical diff at six levels: trace, session, turn, span, token, policy
- Token-level distribution summaries and policy-rule overlays
- LLM-assisted prescriptive fixes grounded on deterministic recommendations
- SOC 2-oriented access log, CycloneDX 1.5 SBOMs, cosign keyless signing
- Ubuntu, macOS, and Windows CI across Python 3.11, 3.12, 3.13
- PyPI Trusted Publisher release pipeline
- Docs site at [manav8498.github.io/Shadow](https://manav8498.github.io/Shadow/)

## What's next

### Sandboxed replay

Tool calls run inside a filesystem snapshot with network isolation and a frozen clock, so replayed agents produce the same side effects they recorded. Needs a per-tool adapter system and a lightweight sandbox runtime. This is the biggest single piece of work on the roadmap.

### Streaming replay

Record and replay inter-chunk timings so a candidate that stalls between chunks looks different from one that bursts them. Needs an `.agentlog` v0.2 chunk-record addition and parser updates.

### Multimodal traces

Images and audio in and out, with a cross-modal semantic axis. Needs blob addressing in the spec and perceptual-similarity work.

### Harness-diff instrumentation

Capture retries, tool-ordering, and context-trim events as first-class diff dimensions instead of burying them in the payload.

### MCP-native replay

We import MCP session logs today. Next: protocol-level interception so MCP tool invocations replay deterministically without re-running the MCP server.

### Runtime policy enforcement

`shadow diff --policy` checks a YAML policy after the fact. The runtime version: enforce the same rules at recording time, block a run that would violate one.

## Not on the roadmap

A few things we are not building:

- **Hosted Shadow.** The whole point is that Shadow lives in your repo and your CI, not in someone else's dashboard. Pair Shadow with Langfuse, Braintrust, or LangSmith if you want a UI.
- **Bundled domain rubrics** (ESI for medical, PCI-DSS for payments). Those are what the `Judge` axis is for. We ship example rubrics in `examples/judges/`.
- **A graphical trace viewer.** Text tools for text formats. `jq < trace.agentlog` is a fine viewer.

## How to influence this

Open an issue with a real use case: *"We hit X with Shadow on our production traffic and Y would have saved us"* is the most useful feedback. Draft PRs welcome for anything on this page. If you want to own a roadmap item, say so in an issue first so we do not duplicate work.
