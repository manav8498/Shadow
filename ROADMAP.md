# Roadmap

Post-v0.1.0. Items are not strictly ordered — what lands first depends on
contributor bandwidth and which real-world use cases surface first. If
something below matters to you, open an issue and we'll pull it forward.

## v0.2 — "works against real LLMs"

**Planned headline features:**

### Live replay backends

- `shadow.llm.AnthropicLLM` and `shadow.llm.OpenAILLM` — implementations of
  the `LlmBackend` Protocol that hit the real APIs. Env-var keyed, gated
  behind `pip install shadow[anthropic,openai]` extras so the default
  install stays local-only.
- `shadow replay --backend live --provider {anthropic,openai}` wires the
  CLI through to these backends.
- Cost guardrails: `--max-calls N` and `--max-spend-usd X` options so a
  misconfigured replay doesn't burn $50 of credits.

### LASSO-over-corners bisection scorer

Replaces the v0.1 heuristic `DELTA_KIND_AFFECTS` allocator with a real
per-corner replay + sparse linear fit:

- Build the design matrix (full factorial `k ≤ 6`, Plackett-Burman
  `k ≤ 23` — both already implemented).
- For each corner, synthesize the intermediate config and run replay via
  the live backend.
- Fit `sklearn.linear_model.Lasso` on the `(runs × k)` design and
  `(runs × 9)` divergence matrix.
- Output actual per-delta attribution instead of kind-based allocation.

Depends on live replay backends being in place.

### Judge axis defaults

The Judge axis ships as a Protocol in v0.1 with no default implementation.
v0.2 ships:

- `shadow.llm.LlmJudge(model="...", rubric="...")` — an LLM-as-judge
  adapter that calls the backend with a user-supplied rubric and extracts
  a 0-1 score.
- Example Judges under `examples/judges/` for common domains
  (procedure-adherence, schema-validation, factuality).

### Per-pair drill-down in the diff report

Today the diff report gives aggregate axis metrics. v0.2 adds per-pair
rows so a reviewer can see which specific trace in the set drove the
regression. Surfaces as collapsible sections in the github-pr renderer.

## v0.3 — observability interop

Wire Shadow into the broader ecosystem:

- **OTel GenAI exporter.** `shadow export --format otel-genai` emits
  traces to an OTel collector. Field mapping is already in SPEC §7.
- **Langfuse / Braintrust / LangSmith importers.** Pull existing traces
  from those platforms into `.agentlog` format so teams already using
  them can try Shadow without re-instrumenting.
- **Auto-instrumentation** of `anthropic` / `openai` Python clients.
  Currently users call `Session.record_chat(req, resp)` manually; v0.3
  adds a `Session.autoinstrument()` context that monkey-patches the
  clients. Deferred from v0.1 because the two SDKs' streaming surfaces
  are annoyingly divergent.

## v0.4 — scale and robustness

- **Windows CI support.** v0.1 is Ubuntu + macOS only; Windows has
  POSIX-path assumptions in `.venv/bin/python` logic that need
  Windows-friendly alternatives.
- **Streaming mid-response redaction.** Today redaction happens at
  record-write time (SPEC §9.1). For very long streamed responses,
  mid-stream redaction would be more defensive.
- **Coverage expansion.** Push Rust coverage past 99% and Python past
  95% — both are currently above the 85% gate but have cheap wins.
- **Fuzz testing.** `cargo-fuzz` on the parser and canonicalizer;
  Hypothesis on the full diff pipeline.

## 1.0 — format lock-in

- `.agentlog` schema freeze at `1.0`. Once we ship a major version, the
  content-hash and canonicalization rules are stable forever. No more
  breaking changes without a 2.0 bump.
- Remote-trace support: `.shadow/remote` config, authentication, signed
  `.agentlog` files for cross-org trace sharing.
- Alternative binary encoding (msgpack or CBOR) as an opt-in — JSONL
  remains the canonical debuggable format.

## Not on the roadmap

Things we've deliberately scoped out:

- **Hosted Shadow.** The whole point of the project is that it lives in
  your repo and your CI, not in a third-party dashboard. If you want a
  dashboard, use Langfuse/Braintrust/LangSmith and (optionally) pair them
  with Shadow's local PR-gate.
- **Domain rubrics bundled by default** (ESI adherence for medical,
  PCI-DSS for payments, GDPR filters, etc.). That hardcoding is what the
  `Judge` axis is for. We'll publish example Judges but not bundle them
  into the core.
- **A graphical trace viewer.** Text tools for text formats. `jq` over
  a `.agentlog` file is a fine viewer; if you want fancy, tools like
  [`fx`](https://github.com/antonmedv/fx) work out of the box.

## How to influence this

- Open an issue with a real use case. "We hit X with Shadow on our
  production traffic and Y would have saved us" is the most useful
  feedback.
- Draft PRs are welcome for any of the above — tag the roadmap item in
  the PR description.
- If you want to own a roadmap item, say so in an issue so we don't
  duplicate work.
