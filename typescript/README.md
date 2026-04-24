# @shadow/sdk

TypeScript SDK for **Shadow**, Git-native behavioral diff for LLM agents.

Records your agent's LLM calls into a `.agentlog` file compatible with
the [Shadow v0.1 SPEC](../SPEC.md), same content-addressed format the
Python SDK and Rust core use. Run `shadow diff` or `shadow bisect` on
the output.

## Install

```bash
npm install @shadow/sdk
# optional, whichever LLM SDKs your agent uses:
npm install openai @anthropic-ai/sdk
```

## Usage

```ts
import { Session } from '@shadow/sdk';
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
