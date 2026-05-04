# Proposal: Divergence semantics in OTel GenAI semantic conventions

> **Status:** draft for the OpenTelemetry GenAI semantic-convention WG.
> **Owner:** manav8498 / Shadow project
> **Date:** 2026-05-03

## Summary

Add a small, well-defined vocabulary to the OTel GenAI semantic conventions
that lets ingestors compare two agent traces and surface a *first
divergence* — the FIRST point at which a candidate trace meaningfully
differs from a baseline. This is the missing primitive between
"observe trace" (current GenAI conventions) and "diagnose regression"
(eval / forensics tooling like Shadow, EvalView, AgentEvals).

## Motivation

Today the OTel GenAI conventions standardise the *recording* of an
agent run (`gen_ai.invoke_agent`, `gen_ai.chat`, `gen_ai.execute_tool`,
`gen_ai.user.message`, etc.). Tooling that wants to *compare* two
recorded runs has to reinvent:

1. How to pair turns across the two traces (alignment).
2. What counts as a meaningful divergence vs. acceptable drift.
3. How to communicate the result.

Each ingestor inventing its own answer means OTel-compatible traces
travel cleanly across tools but *comparisons* don't. A regression
detected by Shadow can't be re-shown by Phoenix or Langfuse without
re-running the comparison locally.

## Proposed additions

### 1. New span kind: `gen_ai.compare`

A `gen_ai.compare` span represents one comparison invocation between
a baseline trace (referenced by `gen_ai.compare.baseline.trace_id`) and
a candidate trace (`gen_ai.compare.candidate.trace_id`). Required
attributes:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.compare.baseline.trace_id` | string | OTel traceId of the baseline run |
| `gen_ai.compare.candidate.trace_id` | string | OTel traceId of the candidate run |
| `gen_ai.compare.algorithm` | string | Free-form name (e.g. `"shadow.align/v0.1"`, `"langsmith.diff/v1"`) |
| `gen_ai.compare.verdict` | enum | `equivalent` / `divergent` / `incomparable` |

### 2. New span events: `gen_ai.divergence`

Each meaningful divergence between the two traces emits a
`gen_ai.divergence` event on the `gen_ai.compare` span. Attributes
mirror the existing typed surface that Shadow + AgentEvals already
share:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.divergence.kind` | enum | `structural_drift` / `decision_drift` / `safety_flip` / `cost_drift` / `latency_drift` |
| `gen_ai.divergence.primary_axis` | enum | `trajectory` / `semantic` / `safety` / `verbosity` / `latency` / `cost` / `reasoning` / `judge` / `conformance` |
| `gen_ai.divergence.baseline_turn` | int | Pair index in the baseline |
| `gen_ai.divergence.candidate_turn` | int | Pair index in the candidate |
| `gen_ai.divergence.confidence` | double | `[0.0, 1.0]` confidence of the divergence |
| `gen_ai.divergence.explanation` | string | Human-readable one-line description |

The first event in alignment order is the "first divergence" by
convention; ingestors that want only the worst pick highest-confidence.

### 3. Optional: `gen_ai.cause` event

When the comparison includes causal attribution (which delta caused the
divergence), an optional `gen_ai.cause` event captures it:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.cause.delta_id` | string | Identifier of the candidate-config delta (file path or config-key path) |
| `gen_ai.cause.axis` | string | Axis the delta moved most strongly |
| `gen_ai.cause.ate` | double | Average treatment effect (Pearl-style) |
| `gen_ai.cause.ci_low` / `gen_ai.cause.ci_high` | double | 95% bootstrap CI on the ATE |
| `gen_ai.cause.e_value` | double | VanderWeele-Ding sensitivity to unmeasured confounding |

These are exactly the fields Shadow's `diagnose-pr` already emits;
publishing them as a convention lets other tools consume them
without translating.

## Compatibility

- Pre-v1.40 traces ignored: the `gen_ai.compare` span is *new* — any
  ingestor that doesn't recognise it can drop it without errors.
- Backwards-compatible with the existing recording conventions —
  this proposal adds a layer *above* the recording layer, not
  inside it.
- No protobuf schema changes. Everything fits in standard OTel
  span / event / attribute primitives.

## Reference implementation

Shadow's `shadow.align` library + `shadow.diagnose_pr.runner` already
produces the proposed payload shape. The `shadow export --format
otel-genai` command can be extended in v0.3 to emit a
`gen_ai.compare` span tree alongside the existing chat/tool spans
when Shadow is run in compare-two-traces mode. Reference:

* [Shadow's design spec, §5 (OTel bridge)](../superpowers/specs/2026-05-03-causal-regression-forensics-design.md)
* [`docs/features/otel-bridge.md`](../features/otel-bridge.md) for the existing import/export contract
* [`docs/features/causal-pr-diagnosis.md`](../features/causal-pr-diagnosis.md) for the divergence + cause data model

## Open questions

1. **Naming.** Should it be `gen_ai.compare` or `gen_ai.diff` or
   `gen_ai.regression`? `compare` reads more neutral; `diff` more
   familiar. Picking the most-pliable term is the WG's call.
2. **Baseline-set vs. baseline-trace.** Some tools compare a candidate
   against a *set* of baseline traces (a regression suite), not a
   single baseline. The proposal covers the 1:1 case; the 1:N case
   could be modeled as N parallel `gen_ai.compare` spans aggregated
   by a parent `gen_ai.compare_suite` span. Worth deciding now.
3. **Axis enum stability.** The 9 axes Shadow surfaces are a working
   set, not a frozen vocabulary. Whether the WG wants to standardise
   the axis names or leave them as free-form strings is the most
   contentious point.

## Why this matters

Without a shared comparison vocabulary, every eval / forensics tool
re-implements alignment + divergence detection. With it, Shadow's
diagnose-pr output, AgentEvals scores, EvalView regression flags,
and Langfuse's diff view can all reference the same shape — and a
trace can carry its own "this run regressed compared to baseline X"
annotation that survives moving between observability tools.

This proposal is intentionally small: it *adds* one span kind plus
two event kinds to a convention that's still in development, without
touching anything already standardised.

---

*Comments welcome. PRs to extend this draft go to
[github.com/manav8498/Shadow](https://github.com/manav8498/Shadow);
formal WG submission once the open questions above settle.*
