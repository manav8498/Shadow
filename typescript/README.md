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
