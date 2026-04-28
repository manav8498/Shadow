# TypeScript SDK feature parity vs Python

**Last reviewed:** 2026-04-28 against Python `shadow-diff==2.6.0`.

The TypeScript SDK (`@shadow/sdk`) is intentionally **trace-recording-
focused**, not feature-parity. It exists so JS / TS agents can write
`.agentlog` files that the Python toolchain analyzes. This document
makes that boundary explicit so users don't expect Python behavior
from the TS package.

## What the TS SDK ships

| Module | Function | Mirrors Python |
|---|---|---|
| `Session` | Open/close a recording context, write `.agentlog` JSONL | `shadow.sdk.Session` |
| `autoInstrument` | Monkey-patch `openai` / `@anthropic-ai/sdk` clients | `shadow.sdk.session.Session` extras |
| `Redactor` | Pre-compiled regex patterns + Luhn validation | `shadow.redact.Redactor` |
| `canonicalJson` / `contentId` | Canonical JSON serialization for content-hash IDs | `shadow._core.canonical_bytes` / `content_id` |
| `writeAgentlog` / `parseAgentlog` | Streaming JSONL parser/writer | `shadow._core.write_agentlog` / `parse_agentlog` |
| `tracing` | OTel trace-id / span-id helpers + W3C traceparent env propagation | `shadow.otel` (subset) |

All TS APIs round-trip with the Python parser: a trace recorded by
`@shadow/sdk` parses cleanly with `shadow._core.parse_agentlog`, and
the content-hash IDs match.

## What the TS SDK does NOT ship (use Python)

These features live only in the Python distribution:

| Python module | What it does | Why TS doesn't have it |
|---|---|---|
| `shadow.diff` (Rust core) | 9-axis behavioral diff | The diff engine is Rust + bootstrap CIs in scipy. Re-implementing in TS is a major undertaking with no current user demand. |
| `shadow.statistical` | Hotelling T², SPRT, mSPRT, fingerprinting | Numerical work — node has weaker stats / linear-algebra ecosystem. |
| `shadow.ltl` | Formal LTLf model checking | Workable in TS but no user pull. |
| `shadow.conformal` | Distribution-free conformal bounds | Same. |
| `shadow.causal` | Do-calculus attribution | Same. |
| `shadow.judge` | LLM-as-judge framework | TS could do this; no user pull. |
| `shadow.bisect` | LASSO config-delta attribution | Numerical work; deferred. |
| `shadow.policy_runtime` | Wrap-tools runtime enforcement | Doable in TS but no user pull. |
| `shadow.cli` | The whole `shadow` CLI | Python entry point only. TS users invoke the Python CLI on traces produced by the TS SDK. |

## Recommended workflow for TS-only teams

1. Use `@shadow/sdk` to record traces from your JS/TS agent code.
2. Commit `.agentlog` fixtures to your repo.
3. Use the **Python** `shadow` CLI in CI to diff, gate, certify,
   and bisect those traces:

   ```yaml
   # .github/workflows/shadow.yml
   - run: pip install shadow-diff==2.6.0
   - run: shadow diff fixtures/baseline.agentlog new.agentlog
   ```

4. Or invoke the Python tool from TS via `child_process.execFile`
   if you need a pure-TS entry point.

## When parity might change

Three triggers would make Python-feature-parity in TS worth the
investment:

1. A user reports they cannot use the Python CLI in their stack
   (true server-side / serverless TS-only environments where adding
   a Python runtime is operationally heavy).
2. The TS SDK installation count exceeds the Python one for a
   sustained period (telemetry-driven signal).
3. A specific feature (most likely: `shadow.policy_runtime` for
   runtime guardrails) is in demand for TS-only deployments.

Until then, the TS SDK stays focused on the recording surface.

F-2 from technical-debt plan: this document is the audit + decision.
The TS SDK is intentionally narrower than the Python one and that
should not be considered a defect. Lifting that decision requires
the user signal listed above.
