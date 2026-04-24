# Roadmap

Where Shadow is headed after **v1.2.x**. Items are not strictly
ordered — what lands first depends on contributor bandwidth and which
real-world use cases surface. If something below matters to you, open
an issue and we'll pull it forward.

## Shipped through v1.2

### v1.0 — core format lock-in + live backends
- `.agentlog v0.1` schema frozen; content-hash + canonicalisation
  stable forever within `1.x`.
- Live replay backends `shadow.llm.AnthropicLLM` + `OpenAILLM`
  (behind `shadow[anthropic]` / `shadow[openai]` extras).
- Auto-instrumentation of Anthropic + OpenAI Python SDKs (incl.
  Responses API, streaming, cache tiers) + TypeScript equivalents.
- LASSO-over-corners causal bisection with Meinshausen-Bühlmann
  stability selection.
- OTel GenAI exporter + importers for Langfuse, Braintrust,
  LangSmith, OpenAI Evals, OTLP.
- Enterprise scaffolding: SOC 2 access log, CycloneDX 1.5 SBOMs,
  cosign keyless signing in the release workflow.
- **10 built-in judges** including a configurable `LlmJudge` that
  takes a rubric string.
- PyPI Trusted-Publisher pipeline + docs site at
  [manav8498.github.io/Shadow](https://manav8498.github.io/Shadow/).

### v1.1 — scale + security hardening
- Scale verified to N=10k pair traces (`benchmarks/scale_drill_down.py`).
- **Banded Needleman-Wunsch** alignment for long traces — fixes the
  super-linear blow-up at N ≥ 5k.
- Path-traversal hardening on `shadow quickstart`.
- Parser resource bounds (16 MiB/1 GiB caps with tunable overrides).
- Full three-OS CI matrix: Ubuntu + macOS + Windows × Python 3.11/3.12.
- Counterfactual replay (1 of the 5 replay-as-science modes).

### v1.2 — partial-item close-out
- Vercel AI SDK + PydanticAI importers.
- Token-level + policy-level hierarchical diff
  (`--token-diff`, `--policy <yaml>`).
- Partial replay (2nd replay-as-science slice) — prefix-locked,
  suffix-live at a chosen branch point.
- LLM-assisted prescriptive fixes (`--suggest-fixes`) with
  retry/backoff and ungrounded-anchor rejection.

## Next up (v1.3+)

### Replay as first-class science

Counterfactual (v1.1) and partial (v1.2) ship. Still open:

- **Sandboxed replay** — tool calls run inside a deterministic
  sandbox (filesystem snapshot, network blackhole, frozen clock) so
  the replay produces the exact same side effects it recorded. Needs
  a tool-sandbox trait + per-tool adapters + a lightweight runtime
  (Linux namespaces + overlayfs v1). Honest estimate: 8-12 weeks,
  dedicated engineer.
- **Streaming replay** — record + replay inter-chunk timings. Needs
  a SPEC v0.2 extension (new `chat_response_chunk` record kind) and
  parser/writer updates. Honest estimate: 2-3 weeks + SPEC review.
- **Multimodal replay** — image / audio inputs and outputs survive
  record → replay → diff with a cross-modal semantic axis. Needs SPEC
  v0.2 binary-blob addressing + perceptual-similarity research.
  Honest estimate: 4-6 weeks.

### Harness-diff instrumentation

Capture retry, tool-order, and context-trim events as first-class
diff dimensions instead of burying them in trace payloads.

### MCP-native replay

MCP importer + examples ship today. Still open: protocol-level
interception + server-session snapshot infra so MCP tool invocations
can be replayed deterministically without re-running the MCP server.

### Adversarial benchmark corpus

Ongoing expansion of `examples/edge-cases/` into a labeled
regression-guard corpus covering more real-world-observed failure
modes.

### Runtime policy enforcement

v1.2 ships `shadow diff --policy` for post-hoc checking. v1.3+:
`.shadowpolicy.yaml` at the repo root + enforcement at recording
time (blocking a production run that would violate a policy rule).
Needs security review.

## Not on the roadmap

Deliberately scoped out:

- **Hosted Shadow.** The whole point is that Shadow lives in your
  repo and your CI, not in a third-party dashboard. Pair Shadow with
  Langfuse / Braintrust / LangSmith if you want a UI.
- **Domain rubrics bundled by default** (ESI adherence for medical,
  PCI-DSS for payments). That hardcoding is what the `Judge` axis is
  for; we ship example rubrics under `examples/judges/` but don't
  bundle them into the core.
- **A graphical trace viewer.** Text tools for text formats.
  `jq < trace.agentlog` is a fine viewer.

## How to influence this

- Open an issue with a real use case. *"We hit X with Shadow on our
  production traffic and Y would have saved us"* is the most useful
  feedback.
- Draft PRs are welcome for any item above — tag the roadmap item in
  the PR description.
- If you want to own a roadmap item, say so in an issue so we don't
  duplicate work.
