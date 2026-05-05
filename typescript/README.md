# shadow-diff (TypeScript SDK)

**Find the exact change that broke your AI agent — from a Node / TypeScript codebase.**

The TypeScript SDK is the recording side of [Shadow](https://github.com/manav8498/Shadow). It captures your agent's LLM calls into an `.agentlog` file, then the Python CLI's `shadow diagnose-pr` answers — in one PR comment — which exact prompt, model, tool-schema, or config change caused the regression. Same trace format, no translation.

The TS package keeps a tight surface (`Session`, redaction, auto-instrument, alignment primitives, gate decision). Numerical analyses (replay, nine-axis diff, bisect, certify, MCP server) live in the Python CLI — point it at the `.agentlog` your TS code produced.

## Install

```bash
npm install shadow-diff
# whichever LLM SDKs your agent uses:
npm install openai @anthropic-ai/sdk
```

## Usage

```ts
import { Session } from 'shadow-diff';
import OpenAI from 'openai';

const session = new Session({
  outputPath: 'trace.agentlog',
  tags: { env: 'dev' },
});
await session.enter();

// Auto-instrumentation patches openai + @anthropic-ai/sdk.
// Your agent code runs unchanged.
const client = new OpenAI();
await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'hi' }],
});

await session.exit();
```

Secrets (API keys, emails, credit cards) are redacted by default.

## Zero-config recording (no code change)

```bash
shadow record -o trace.agentlog -- node my-agent.js
shadow record -o trace.agentlog -- npx tsx my-langgraph-agent.ts
```

`shadow record` detects Node-family commands (`node` / `npx` / `tsx` / `ts-node` / `npm` / `pnpm` / `yarn`) and injects `NODE_OPTIONS='--import shadow-diff/auto'` plus `SHADOW_SESSION_OUTPUT=...`. Shadow opens a `Session` on Node startup, runs `autoInstrument()`, and flushes the trace on `beforeExit`.

Requirements: `shadow-diff` installed in your project, Node ≥ 20.6 (for the `--import` flag), and an agent that exits naturally — `process.exit()` skips `beforeExit` and drops trailing records (the content-addressed envelope means partial traces still parse).

You can also activate it directly:

```bash
SHADOW_SESSION_OUTPUT=trace.agentlog \
  node --import shadow-diff/auto my-agent.js
```

## CI gating from TypeScript

For repos that run their CI checks in Node, the TS SDK ships a `gate(records, { rules, ltlFormulas })` decision surface. Pass the parsed `.agentlog` records and a policy; get back a verdict and the failing rules. Byte-identical to Python's `shadow.policy_runtime` on the same fixtures (cross-validated by `python/tests/test_typescript_parity.py`). For deeper analyses (multi-axis diff, bisect, certify), run those from the Python CLI against the TS-recorded trace.

## Trace alignment primitives

The `align` submodule ships pure-TypeScript implementations of `trajectoryDistance` (Levenshtein on tool sequences) and `toolArgDelta` (structural JSON diff) that produce byte-identical results to Python's `shadow.align` and Rust's `shadow-align` on the same inputs.

```ts
import { trajectoryDistance, toolArgDelta, isNativeAvailable } from 'shadow-diff';

trajectoryDistance(['search', 'edit'], ['search']);   // 0.5
toolArgDelta({ x: 1 }, { x: '1' });                   // [{ kind: 'type_changed', ... }]
```

### Optional native acceleration (napi-rs)

For workloads with large traces, the same algorithms ship as a Rust addon (`@shadow-diff/align-native`, built from `crates/shadow-align`). When the platform-specific `.node` file is present, `trajectoryDistance` and `toolArgDelta` transparently use it — same surface, same results, substantially faster on long sequences. `isNativeAvailable()` reports whether the addon was found. The pure-TS path is the silent fallback, so consumers don't need to handle the absence.

## Distributed tracing

Multi-process agents can join a single logical trace via `SHADOW_TRACE_ID` or the W3C `traceparent` env var. The parent session emits the right env for children:

```ts
const env = session.envForChild();
spawn('node', ['worker.js'], { env: { ...process.env, ...env } });
```

## Dev

```bash
npm install
npm run typecheck
npm test
```

## Full docs

`SPEC.md`, the runnable examples, the comparison matrix against adjacent tools, and every CLI / feature page live at **https://github.com/manav8498/Shadow**.

## License

Apache-2.0. The `.agentlog` spec is independently published under Apache-2.0.
