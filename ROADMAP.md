# Roadmap

Post-v0.1.0. Items are not strictly ordered — what lands first depends
on contributor bandwidth and which real-world use cases surface first.
If something below matters to you, open an issue and we'll pull it
forward.

## Already landed in v0.1.0 (previously planned for v0.2)

The following items were on the v0.2 roadmap at project start but
ended up shipping in v0.1.0:

- **Live replay backends** — `shadow.llm.AnthropicLLM` and
  `shadow.llm.OpenAILLM` ship today, gated behind `shadow[anthropic]`
  and `shadow[openai]` extras.
- **LASSO-over-corners bisection** with Meinshausen-Bühlmann
  stability selection — live-replay mode fits a real sparse linear
  model per axis across corners. The heuristic kind-based allocator
  remains as a no-credentials fallback.
- **Auto-instrumentation** of the Anthropic + OpenAI Python SDKs
  (plus OpenAI Responses API, streaming, cache tiers) and their
  TypeScript equivalents.
- **OTel GenAI exporter** (`shadow.otel`) producing OTLP/JSON that
  collectors can ingest, following the v1.37+ GenAI semconv.
- **Importers** for Langfuse, Braintrust, LangSmith, OpenAI Evals,
  and OTLP.
- **Enterprise scaffolding** — SOC 2-oriented access log,
  CycloneDX 1.5 SBOM generation, cosign keyless signing in the
  release workflow.

## v0.2 — reliability and depth

### Judge axis defaults

The Judge axis ships as a Protocol with a domain-agnostic
`SanityJudge` in v0.1. v0.2 adds:

- `shadow.llm.LlmJudge(model="...", rubric="...")` — a generic LLM-
  as-judge adapter that calls the backend with a user-supplied rubric
  and extracts a 0-1 score.
- Example Judges under `examples/judges/` for common domains
  (procedure-adherence, schema-validation, factuality).

### Per-pair drill-down in the diff report

Today the diff report gives aggregate axis metrics. v0.2 adds per-pair
rows so a reviewer can see which specific trace in the set drove the
regression. Surfaces as collapsible sections in the `github-pr`
renderer.

### Embeddings-default semantic axis (behind a flag)

The v0.1 semantic axis defaults to a fast TF-IDF cosine; real
sentence-transformer embeddings are an opt-in via
`shadow[embeddings]`. v0.2 evaluates whether TF-IDF is ever wrong
enough in practice to justify making embeddings the default.

## v0.3 — scale and robustness

- **Windows CI support.** v0.1 is Ubuntu + macOS only; Windows has
  POSIX-path assumptions in `.venv/bin/python` logic that need
  Windows-friendly alternatives.
- **Streaming mid-response redaction.** Today redaction happens at
  record-write time. For very long streamed responses, mid-stream
  redaction would be more defensive.
- **Fuzz testing.** `cargo-fuzz` on the parser and canonicalizer;
  Hypothesis on the full diff pipeline.

## 1.0 — format lock-in

- `.agentlog` schema freeze at `1.0`. Once we ship a major version,
  the content-hash and canonicalization rules are stable forever. No
  more breaking changes without a 2.0 bump.
- Remote-trace support: `.shadow/remote` config, authentication,
  signed `.agentlog` files for cross-org trace sharing.
- Alternative binary encoding (msgpack or CBOR) as an opt-in — JSONL
  remains the canonical debuggable format.

## Not on the roadmap

Things we've deliberately scoped out:

- **Hosted Shadow.** The whole point of the project is that it lives
  in your repo and your CI, not in a third-party dashboard. If you
  want a dashboard, use Langfuse / Braintrust / LangSmith and
  (optionally) pair them with Shadow's local PR-gate.
- **Domain rubrics bundled by default** (ESI adherence for medical,
  PCI-DSS for payments, GDPR filters, etc.). That hardcoding is what
  the `Judge` axis is for. We publish example Judges but don't bundle
  them into the core.
- **A graphical trace viewer.** Text tools for text formats. `jq`
  over a `.agentlog` file is a fine viewer.

## Reality check

Shadow is a **v0.1.0 project with no external users and no production
deployments yet**. Everything above is contingent on real-world
feedback. If nobody uses it, the roadmap is moot. If the first users
want something totally different, the roadmap shifts.

## How to influence this

- Open an issue with a real use case. "We hit X with Shadow on our
  production traffic and Y would have saved us" is the most useful
  feedback.
- Draft PRs are welcome for any of the above — tag the roadmap item
  in the PR description.
- If you want to own a roadmap item, say so in an issue so we don't
  duplicate work.
