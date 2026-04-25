import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parseAgentlog } from '../src/agentlog.js';
import {
  anthropicTranslators,
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
});
