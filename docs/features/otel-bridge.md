# OpenTelemetry GenAI bridge

> **Phase 5 of the Causal Regression Forensics roadmap.** Status: experimental
> while the OTel GenAI semantic conventions are still maturing.

`shadow import` and `shadow export` round-trip Shadow's `.agentlog` format
to/from OpenTelemetry OTLP/JSON using the GenAI semantic conventions
(`opentelemetry-gen-ai-v1.40` and earlier majors). This means any tool that
emits OTel-instrumented agent traces — Phoenix, Langfuse, Traceloop's
OpenLLMetry, Microsoft's Semantic Kernel SDK, native OTel-instrumented apps —
can feed `shadow diagnose-pr` directly.

## Quick start

```bash
# Export a Shadow trace to OTel JSON
shadow export trace.agentlog --format otel-genai --output trace.otel.json

# Import an OTel JSON file (or any OTLP/JSON dump) back to Shadow
shadow import --format otel-genai trace.otel.json --output trace.agentlog

# Then run diagnose-pr against the imported trace
shadow diagnose-pr \
  --traces            otel_baseline_traces/ \
  --candidate-traces  otel_candidate_traces/ \
  --baseline-config   baseline.yaml \
  --candidate-config  candidate.yaml \
  --policy            policy.yaml \
  --backend           recorded
```

The `--format otel-genai` alias is identical to `--format otel`; it matches
the design spec's literal wording.

## Mapping

| OTel | `.agentlog` |
|---|---|
| `gen_ai.invoke_agent` span | metadata record |
| `gen_ai.chat` span (with input/output messages) | `chat_request` + `chat_response` records |
| `gen_ai.execute_tool` span | `tool_call` + `tool_result` records |
| OTel `traceId` | envelope `meta.trace_id` |
| OTel `spanId` | envelope `meta.otel_span_id` |
| `gen_ai.user.message` event | `messages[].content` (role=user) |
| `gen_ai.system.message` event | `messages[].content` (role=system) |
| `gen_ai.assistant.message` event | `chat_response.content[].text` + `tool_use[]` blocks (when `tool_calls` attr present) |
| `gen_ai.tool.message` event | `messages[].content` (role=tool) |
| `gen_ai.usage.input_tokens` attr | `usage.input_tokens` |
| `gen_ai.usage.output_tokens` attr | `usage.output_tokens` |
| `gen_ai.response.finish_reasons` attr | `stop_reason` |
| `shadow.latency_ms` attr | `latency_ms` (preferred over span duration) |

The exporter emits **both** the v1.37+ structured-attribute path (`gen_ai.input.messages`,
`gen_ai.output.messages`) — when applicable — and the v1.28-v1.36 deprecated
per-message events (`gen_ai.user.message` etc.). The importer accepts any of
these plus the legacy `gen_ai.prompt.N.*` / `gen_ai.completion.N.*` flat-indexed
attributes from earlier minors.

## Round-trip guarantee

Native diagnose-pr verdict on the refund demo: `STOP, 3/3 affected,
prompt.system as dominant cause`. Same corpus exported to OTel and re-imported
produces the **identical** verdict + affected count + per-axis severities.
This is pinned by `python/tests/test_otel_diagnose_pr_e2e.py`:

  * `test_otel_roundtrip_preserves_per_pair_diff_outcome` — same per-axis
    severities + first_divergence after round-trip.
  * `test_otel_imported_traces_have_unique_trace_ids` — the OTel `traceId`
    is stamped into envelope `meta.trace_id`, preventing the metadata-content-
    hash collision when multiple traces share byte-identical metadata.
  * `test_otel_imported_corpus_diagnose_pr_matches_native_verdict` — full
    diagnose-pr verdict identity native vs. round-tripped.

## Known limitations

* **No streaming**: imports are file-at-a-time JSON. For high-volume OTel
  collectors, route through a file sink first.
* **Embeddings spans are dropped**: `gen_ai.embeddings` spans have no
  natural mapping onto `chat_request` / `chat_response`; they're skipped on
  import. (Future: a dedicated `embedding` record kind.)
* **Privacy**: Shadow's exporter emits message content by default. To strip
  it, post-process the JSON to remove `gen_ai.*.message` event attributes
  before sending downstream. (Live OTel collectors typically have an
  attribute-filter processor for this.)

## See also

* [`shadow import` / `shadow export`](../reference/cli.md#shadow-import-source---format-fmt) — full flag set for both directions of the bridge.
* The OTel divergence-semantics proposal: [`docs/proposals/otel-genai-divergence.md`](../proposals/otel-genai-divergence.md) — the WG draft for adding `gen_ai.compare` / `gen_ai.divergence` semantic conventions.
* The existing `shadow.otel` module + `shadow.importers.otel` module hold
  the implementation; this page documents the user-facing contract.
