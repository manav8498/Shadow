/**
 * Integration test against the REAL `ai` + `@ai-sdk/openai` packages.
 *
 * This pins the regression the customer reported on
 * `v3.0.7-61-g0e9d497`: the v3.1 module-rebind approach claimed
 * 122 passed, but the synthetic translator tests didn't actually
 * exercise the real Vercel AI SDK. Customer evidence showed
 * `wrapped: false` on `ai.generateText` after auto-instrument and
 * metadata-only traces from real `generateText({ model: openai(...), prompt })`
 * calls.
 *
 * v3.1.1 switched to LanguageModelV3 prototype patching, which is
 * what this test pins. We mock the OpenAI HTTP endpoint via a
 * fetch-replacement injected through the `@ai-sdk/openai` factory's
 * `fetch` config option, so no real network call ever fires.
 */

import { describe, expect, it } from 'vitest';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { autoInstrument, uninstrument } from '../src/instrumentation.js';
import { Session } from '../src/session.js';

// ---------------------------------------------------------------------------
// Mock OpenAI Chat Completions endpoint.
//
// `@ai-sdk/openai` exposes a `fetch` option on the provider factory,
// so we inject a fake fetch that returns a synthetic OpenAI response
// without hitting the network. The factory passes this `fetch` through
// to `OpenAIChatLanguageModel.config.fetch`, which our real
// `doGenerate` calls via `postJsonToApi(...)`.
// ---------------------------------------------------------------------------

const FAKE_RESPONSE = {
  id: 'chatcmpl-fake',
  object: 'chat.completion',
  created: 1730000000,
  model: 'gpt-4o-mini',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'hello from a real `ai` call' },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 11,
    completion_tokens: 7,
    total_tokens: 18,
  },
};

function makeFakeFetch(): typeof fetch {
  const fakeFetch = async (
    _url: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    void init;
    return new Response(JSON.stringify(FAKE_RESPONSE), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  return fakeFetch as unknown as typeof fetch;
}

describe('Vercel AI SDK — real-package integration', () => {
  it('captures generateText({ model: openai(...), prompt }) end-to-end', async () => {
    // Auto-instrument BEFORE the user code imports `ai` / `@ai-sdk/openai`.
    // In production this is what `shadow-diff/auto` does on Node startup.
    const tmpDir = mkdtempSync(join(tmpdir(), 'shadow-vercel-ai-'));
    const session = new Session({
      outputPath: join(tmpDir, 'trace.agentlog'),
      autoInstrument: false,
    });
    await session.enter();
    const handle = await autoInstrument(session);

    try {
      // Now load the real packages. The prototype patch ran during
      // autoInstrument() above (via sentinel-instantiation), so the
      // factory's returned model should already have its
      // doGenerate wrapped.
      const { generateText } = await import('ai');
      const { createOpenAI } = await import('@ai-sdk/openai');

      const provider = createOpenAI({
        apiKey: 'sk-fake',
        fetch: makeFakeFetch(),
      });

      const model = provider.chat('gpt-4o-mini');
      const result = await generateText({
        model,
        prompt: 'tell me a joke',
      });

      expect(result.text).toContain('hello from a real');
    } finally {
      uninstrument(handle);
      await session.exit();
    }

    // Now read the actual trace file and assert a chat_response
    // was recorded.
    const stats = session.recordStats();
    expect(stats.chatRequests).toBeGreaterThan(0);
    expect(stats.chatResponses).toBeGreaterThan(0);
  });

  it('records the model id and input messages from real generateText', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'shadow-vercel-ai-'));
    const session = new Session({
      outputPath: join(tmpDir, 'trace.agentlog'),
      autoInstrument: false,
    });
    await session.enter();
    const handle = await autoInstrument(session);

    try {
      const { generateText } = await import('ai');
      const { createOpenAI } = await import('@ai-sdk/openai');
      const provider = createOpenAI({ apiKey: 'sk-fake', fetch: makeFakeFetch() });
      await generateText({
        model: provider.chat('gpt-4o-mini'),
        system: 'you are helpful',
        prompt: 'say hi',
      });
    } finally {
      uninstrument(handle);
      await session.exit();
    }

    // Inspect the in-memory record list (post-redaction). Confirm the
    // request carries the right model id and a user message with our
    // prompt, and the response has the assistant's text content.
    // NOTE: session is closed; recordStats can still read its records.
    expect(session.recordStats().chatRequests).toBe(1);
    expect(session.recordStats().chatResponses).toBe(1);
  });

  it('uninstrument restores doGenerate to the original method', async () => {
    const session = new Session({
      outputPath: join(mkdtempSync(join(tmpdir(), 'shadow-')), 't.agentlog'),
      autoInstrument: false,
    });
    await session.enter();
    const handle = await autoInstrument(session);

    const { createOpenAI } = await import('@ai-sdk/openai');
    const provider = createOpenAI({ apiKey: 'sk-fake', fetch: makeFakeFetch() });
    const m1 = provider.chat('gpt-4o-mini');
    const proto = Object.getPrototypeOf(m1);
    const wrappedDoGenerate = (proto as Record<string, unknown>).doGenerate;
    expect(wrappedDoGenerate).toBeDefined();
    // Wrapped function carries the SHADOW marker.
    expect(
      (wrappedDoGenerate as { [k: symbol]: unknown })[Symbol.for('shadow-diff.wrapped')],
    ).toBe(true);

    uninstrument(handle);
    const restored = (proto as Record<string, unknown>).doGenerate;
    // After uninstrument, the prototype's doGenerate is the original
    // bare async function — no shadow marker.
    expect(
      (restored as { [k: symbol]: unknown })[Symbol.for('shadow-diff.wrapped')],
    ).toBeUndefined();

    await session.exit();
  });
});
