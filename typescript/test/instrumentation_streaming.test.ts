import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parseAgentlog } from '../src/agentlog.js';
import {
  anthropicTranslators,
  openaiResponsesTranslators,
  openaiTranslators,
  type Translators,
} from '../src/instrumentation.js';
import { Session } from '../src/session.js';

/**
 * Drives the production OpenAI / Anthropic aggregators against
 * canned chunks. The aggregator factory comes from the same
 * exported translator object the autoInstrument wrapper uses, so
 * passing here means the production code path passes too.
 *
 * The wrapStream helper from instrumentation.ts is module-private;
 * we reproduce its outer shape (per-stream aggregator + record-once
 * on iteration end) to test the aggregator itself in isolation.
 */
async function drainStreaming(
  session: Session,
  chunks: unknown[],
  translators: Translators,
  kwargs: Record<string, unknown>,
  earlyBreakAfter?: number,
): Promise<void> {
  const aggregator = translators.aggregate!();
  let n = 0;
  for (const chunk of chunks) {
    aggregator.consume(chunk);
    n += 1;
    if (earlyBreakAfter !== undefined && n >= earlyBreakAfter) break;
  }
  const finalResp = aggregator.finalize();
  const req = translators.req(kwargs);
  const resp = translators.resp(finalResp, 100);
  session.recordChat(req, resp);
}

describe('streaming aggregation (TypeScript)', () => {
  it('OpenAI: text deltas are joined into a single text content block', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    const chunks: unknown[] = [
      { model: 'gpt-4o-mini', choices: [{ delta: { content: 'Hello' } }] },
      { choices: [{ delta: { content: ', ' } }] },
      { choices: [{ delta: { content: 'world!' } }] },
      {
        choices: [{ delta: {}, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 3 },
      },
    ];
    await drainStreaming(s, chunks, openaiTranslators, {
      stream: true,
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'hi' }],
    });
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const resp = records.find((r) => r.kind === 'chat_response');
    expect(resp).toBeTruthy();
    const content = (resp!.payload as { content: Array<{ type: string; text?: string }> }).content;
    expect(content).toHaveLength(1);
    expect(content[0]).toEqual({ type: 'text', text: 'Hello, world!' });
    const usage = (resp!.payload as { usage: { input_tokens: number; output_tokens: number } })
      .usage;
    expect(usage.input_tokens).toBe(5);
    expect(usage.output_tokens).toBe(3);
  });

  it('OpenAI: interleaved tool_call argument deltas reassemble per index', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    const chunks: unknown[] = [
      {
        model: 'gpt-4o-mini',
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: 'call_a',
                  type: 'function',
                  function: { name: 'lookup_order', arguments: '{"order' },
                },
              ],
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: { tool_calls: [{ index: 0, function: { arguments: '_id":' } }] },
          },
        ],
      },
      {
        choices: [
          {
            delta: { tool_calls: [{ index: 0, function: { arguments: '"ORD-1"}' } }] },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    ];
    await drainStreaming(s, chunks, openaiTranslators, {
      stream: true,
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'find it' }],
    });
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const resp = records.find((r) => r.kind === 'chat_response');
    const content = (
      resp!.payload as { content: Array<{ type: string; name?: string; input?: unknown }> }
    ).content;
    const toolUse = content.find((b) => b.type === 'tool_use');
    expect(toolUse).toBeTruthy();
    expect(toolUse!.name).toBe('lookup_order');
    expect(toolUse!.input).toEqual({ order_id: 'ORD-1' });
    const stopReason = (resp!.payload as { stop_reason: string }).stop_reason;
    expect(stopReason).toBe('tool_use'); // OpenAI 'tool_calls' → Shadow 'tool_use'
  });

  it('Anthropic: text + tool_use blocks reassemble from chunked deltas', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    const chunks: unknown[] = [
      {
        type: 'message_start',
        message: { model: 'claude-sonnet-4', usage: { input_tokens: 12, output_tokens: 0 } },
      },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Looking ' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'now.' } },
      { type: 'content_block_stop', index: 0 },
      {
        type: 'content_block_start',
        index: 1,
        content_block: { type: 'tool_use', id: 'tu_1', name: 'fetch_order', input: {} },
      },
      {
        type: 'content_block_delta',
        index: 1,
        delta: { type: 'input_json_delta', partial_json: '{"id":' },
      },
      {
        type: 'content_block_delta',
        index: 1,
        delta: { type: 'input_json_delta', partial_json: ' "X-7"}' },
      },
      { type: 'content_block_stop', index: 1 },
      {
        type: 'message_delta',
        delta: { stop_reason: 'tool_use' },
        usage: { output_tokens: 22 },
      },
      { type: 'message_stop' },
    ];
    await drainStreaming(s, chunks, anthropicTranslators, {
      stream: true,
      model: 'claude-sonnet-4',
      messages: [{ role: 'user', content: 'find X-7' }],
    });
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const resp = records.find((r) => r.kind === 'chat_response');
    expect(resp).toBeTruthy();
    const payload = resp!.payload as {
      content: Array<{ type: string; text?: string; name?: string; input?: unknown }>;
      stop_reason: string;
      usage: { input_tokens: number; output_tokens: number };
    };
    expect(payload.stop_reason).toBe('tool_use');
    expect(payload.usage.input_tokens).toBe(12);
    expect(payload.usage.output_tokens).toBe(22);
    expect(payload.content).toHaveLength(2);
    expect(payload.content[0]).toEqual({ type: 'text', text: 'Looking now.' });
    expect(payload.content[1]).toMatchObject({
      type: 'tool_use',
      id: 'tu_1',
      name: 'fetch_order',
      input: { id: 'X-7' },
    });
  });

  it('caller-side break records what was seen so far', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    const chunks: unknown[] = [
      { model: 'gpt-4o-mini', choices: [{ delta: { content: 'A' } }] },
      { choices: [{ delta: { content: 'B' } }] },
      { choices: [{ delta: { content: 'C' } }] },
      { choices: [{ delta: { content: 'D' } }] },
    ];
    await drainStreaming(
      s,
      chunks,
      openaiTranslators,
      { stream: true, model: 'gpt-4o-mini', messages: [] },
      2,
    );
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const resp = records.find((r) => r.kind === 'chat_response');
    const text = (resp!.payload as { content: Array<{ text?: string }> }).content[0].text;
    expect(text).toBe('AB');
  });

  it('OpenAI Responses: non-streaming output[] becomes content blocks', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    // Drive translators directly (no streaming) by handing the
    // aggregator a `response.completed` event carrying a full
    // Response object. This is the same path the production
    // wrapStream takes when the SDK delivers the final event.
    const fullResponse = {
      model: 'gpt-4o-mini',
      output: [
        {
          type: 'message',
          role: 'assistant',
          content: [{ type: 'output_text', text: 'Hello, Responses!' }],
        },
      ],
      usage: {
        input_tokens: 11,
        output_tokens: 4,
        output_tokens_details: { reasoning_tokens: 0 },
      },
      status: 'completed',
    };
    const aggregator = openaiResponsesTranslators.aggregate!();
    aggregator.consume({ type: 'response.completed', response: fullResponse });
    const finalResp = aggregator.finalize();
    const req = openaiResponsesTranslators.req({
      model: 'gpt-4o-mini',
      input: 'hi',
      instructions: 'Be helpful',
    });
    const resp = openaiResponsesTranslators.resp(finalResp, 100);
    s.recordChat(req, resp);
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const reqRec = records.find((r) => r.kind === 'chat_request');
    const respRec = records.find((r) => r.kind === 'chat_response');
    expect(reqRec).toBeTruthy();
    expect(respRec).toBeTruthy();

    // instructions becomes a system message prepended to messages.
    const messages = (reqRec!.payload as { messages: Array<{ role: string; content: string }> })
      .messages;
    expect(messages[0]).toEqual({ role: 'system', content: 'Be helpful' });
    expect(messages[1]).toEqual({ role: 'user', content: 'hi' });

    // output[].message.content[].output_text → text content block.
    const content = (respRec!.payload as { content: Array<{ type: string; text?: string }> })
      .content;
    expect(content).toHaveLength(1);
    expect(content[0]).toEqual({ type: 'text', text: 'Hello, Responses!' });

    // usage normalised to Shadow's input/output/thinking shape.
    const usage = (
      respRec!.payload as { usage: { input_tokens: number; output_tokens: number } }
    ).usage;
    expect(usage.input_tokens).toBe(11);
    expect(usage.output_tokens).toBe(4);
  });

  it('OpenAI Responses: function_call output becomes a tool_use block', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    const fullResponse = {
      model: 'gpt-4o-mini',
      output: [
        {
          type: 'function_call',
          id: 'fc_1',
          call_id: 'call_42',
          name: 'lookup_order',
          arguments: '{"order_id":"ORD-1"}',
        },
      ],
      usage: { input_tokens: 5, output_tokens: 7 },
      status: 'completed',
    };
    const aggregator = openaiResponsesTranslators.aggregate!();
    aggregator.consume({ type: 'response.completed', response: fullResponse });
    const finalResp = aggregator.finalize();
    const req = openaiResponsesTranslators.req({
      model: 'gpt-4o-mini',
      input: 'find ORD-1',
      tools: [
        {
          type: 'function',
          name: 'lookup_order',
          description: 'Look up an order',
          parameters: { type: 'object' },
        },
      ],
    });
    const resp = openaiResponsesTranslators.resp(finalResp, 100);
    s.recordChat(req, resp);
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const respRec = records.find((r) => r.kind === 'chat_response');
    const content = (
      respRec!.payload as {
        content: Array<{ type: string; id?: string; name?: string; input?: unknown }>;
      }
    ).content;
    const toolUse = content.find((b) => b.type === 'tool_use');
    expect(toolUse).toBeTruthy();
    expect(toolUse).toEqual({
      type: 'tool_use',
      id: 'call_42',
      name: 'lookup_order',
      input: { order_id: 'ORD-1' },
    });
    expect((respRec!.payload as { stop_reason: string }).stop_reason).toBe('tool_use');
  });

  it('OpenAI Responses: streaming deltas without response.completed assemble correctly', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-stream-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();

    // Simulate a stream where the caller broke early (no
    // response.completed). The aggregator must still produce a
    // sensible final shape from delta events.
    const events: unknown[] = [
      {
        type: 'response.created',
        response: { model: 'gpt-4o-mini', status: 'in_progress' },
      },
      { type: 'response.output_text.delta', delta: 'Hi ' },
      { type: 'response.output_text.delta', delta: 'there!' },
      {
        type: 'response.output_item.added',
        item: {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_99',
          name: 'do_thing',
        },
      },
      {
        type: 'response.function_call_arguments.delta',
        item_id: 'fc_1',
        delta: '{"x":',
      },
      {
        type: 'response.function_call_arguments.delta',
        item_id: 'fc_1',
        delta: '1}',
      },
    ];
    await drainStreaming(s, events, openaiResponsesTranslators, {
      stream: true,
      model: 'gpt-4o-mini',
      input: 'hi',
    });
    await s.exit();

    const records = parseAgentlog(await readFile(path));
    const respRec = records.find((r) => r.kind === 'chat_response');
    const content = (
      respRec!.payload as {
        content: Array<{ type: string; text?: string; name?: string; input?: unknown }>;
      }
    ).content;
    const text = content.find((b) => b.type === 'text');
    const toolUse = content.find((b) => b.type === 'tool_use');
    expect(text?.text).toBe('Hi there!');
    expect(toolUse).toEqual({
      type: 'tool_use',
      id: 'call_99',
      name: 'do_thing',
      input: { x: 1 },
    });
  });
});
