# shadow-diff

TypeScript SDK for **Shadow**, Git-native behavioral diff for LLM agents.

Records your agent's LLM calls into a `.agentlog` file compatible with
the `.agentlog` v0.1 envelope and v0.2 record-kind extensions defined in
[SPEC.md](../SPEC.md), same content-addressed format the Python SDK
and Rust core use. Run `shadow diff` or `shadow bisect` on the output.

The TypeScript SDK has its own version line and only covers the
recording side (Session, redaction, auto-instrument). Replay, diff,
bisect, certify, and the MCP server live in the Python CLI, which
reads the same trace format without translation.

## Install

```bash
npm install shadow-diff
# optional, whichever LLM SDKs your agent uses:
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

// Auto-instrumentation patches openai + @anthropic-ai/sdk, user code
// runs unchanged.
const client = new OpenAI();
await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'hi' }],
});

await session.exit();
```

## Trace alignment primitives

The `align` submodule ships pure-TypeScript implementations of
`trajectoryDistance` (Levenshtein on tool sequences) and `toolArgDelta`
(structural JSON diff) that produce byte-identical results to Python's
`shadow.align` and Rust's `shadow-align` on the same inputs.

```ts
import { trajectoryDistance, toolArgDelta, isNativeAvailable } from 'shadow-diff';

trajectoryDistance(['search', 'edit'], ['search']);   // 0.5
toolArgDelta({ x: 1 }, { x: '1' });                   // [{ kind: 'type_changed', ... }]
```

### Optional native acceleration (napi-rs)

For workloads with large traces, the same algorithms ship as a Rust
addon (`@shadow-diff/align-napi`, built from `crates/shadow-align`).
When the platform-specific `.node` file is present, `trajectoryDistance`
and `toolArgDelta` transparently use it — same surface, same results,
substantially faster on long sequences. `isNativeAvailable()` reports
whether the addon was found. The pure-TS path is the silent fallback,
so consumers don't need to handle the absence.

## Distributed tracing

Multi-process agents can join a single logical trace via
`SHADOW_TRACE_ID` or the W3C `traceparent` env var. The parent session
emits the right env for children:

```ts
const env = session.envForChild();
spawn('node', ['worker.js'], { env: { ...process.env...env } });
```

## Dev

```bash
npm install
npm run typecheck
npm test
```

## License

MIT. `SPEC.md` is Apache-2.0.
