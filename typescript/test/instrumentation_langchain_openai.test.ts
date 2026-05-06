/**
 * Integration test against the real `@langchain/openai` package.
 *
 * Pins the v3.1.1 customer-reported gap: `@langchain/openai` bundles
 * its own copy of `openai` in nested `node_modules/`, so the existing
 * `tryPatchOpenAI` patch — anchored to the user's top-level `openai`
 * resolution — never reached LangChain's bundled copy. Customer
 * evidence: `mini-ts-swe-agent` recorded only metadata when calling
 * through `ChatOpenAI`, while direct `openai` calls in the same repo
 * recorded chat_request + chat_response.
 *
 * v3.1.2 fixes this by patching `_generate` and `_streamResponseChunks`
 * on `ChatOpenAI.prototype` (and the Completions / Responses variants)
 * directly. We pin that path here against a faked OpenAI HTTP endpoint
 * so no real network call fires.
 */

import { describe, expect, it } from 'vitest';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { autoInstrument, uninstrument } from '../src/instrumentation.js';
import { Session } from '../src/session.js';

const FAKE_RESPONSE = {
  id: 'chatcmpl-fake-langchain',
  object: 'chat.completion',
  created: 1730000000,
  model: 'gpt-4o-mini',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'hello from a real langchain call' },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 13,
    completion_tokens: 8,
    total_tokens: 21,
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

describe('@langchain/openai — real-package integration', () => {
  it('captures ChatOpenAI.invoke({ messages }) end-to-end', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'shadow-langchain-openai-'));
    const session = new Session({
      outputPath: join(tmpDir, 'trace.agentlog'),
      autoInstrument: false,
    });
    await session.enter();
    const handle = await autoInstrument(session);

    try {
      const { ChatOpenAI } = await import('@langchain/openai');
      const model = new ChatOpenAI({
        model: 'gpt-4o-mini',
        apiKey: 'sk-fake',
        maxRetries: 0,
        configuration: { fetch: makeFakeFetch() },
      });
      const result = await model.invoke('say hi');
      expect(typeof result.content).toBe('string');
      expect(result.content as string).toContain('hello from a real');
    } finally {
      uninstrument(handle);
      await session.exit();
    }

    const stats = session.recordStats();
    expect(stats.chatRequests).toBeGreaterThan(0);
    expect(stats.chatResponses).toBeGreaterThan(0);
  });

  it('records the model id and input messages on real invoke', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'shadow-langchain-openai-'));
    const session = new Session({
      outputPath: join(tmpDir, 'trace.agentlog'),
      autoInstrument: false,
    });
    await session.enter();
    const handle = await autoInstrument(session);
    let traceFile = '';
    try {
      const { ChatOpenAI } = await import('@langchain/openai');
      const { HumanMessage, SystemMessage } = await import('@langchain/core/messages');
      const model = new ChatOpenAI({
        model: 'gpt-4o-mini',
        apiKey: 'sk-fake',
        maxRetries: 0,
        configuration: { fetch: makeFakeFetch() },
      });
      await model.invoke([
        new SystemMessage('you are concise'),
        new HumanMessage('hi'),
      ]);
      traceFile = (session as unknown as { outputPath: string }).outputPath;
    } finally {
      uninstrument(handle);
      await session.exit();
    }

    // Read the agentlog and verify the recorded chat_request carries
    // the right model id and the user/system roles in order.
    const { readFileSync } = await import('node:fs');
    const lines = readFileSync(traceFile, 'utf8')
      .split('\n')
      .filter((l) => l.length > 0)
      .map((l) => JSON.parse(l));
    const chatReq = lines.find((r) => r.kind === 'chat_request');
    const chatResp = lines.find((r) => r.kind === 'chat_response');
    expect(chatReq).toBeDefined();
    expect(chatResp).toBeDefined();
    expect(chatReq.payload.model).toBe('gpt-4o-mini');
    const msgs = chatReq.payload.messages as Array<{ role: string }>;
    expect(msgs.length).toBe(2);
    expect(msgs[0].role).toBe('system');
    expect(msgs[1].role).toBe('user');
    expect(chatResp.payload.model).toBe('gpt-4o-mini');
    expect(chatResp.payload.usage.input_tokens).toBe(13);
    expect(chatResp.payload.usage.output_tokens).toBe(8);
  });
});
