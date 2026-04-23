import { tmpdir } from 'node:os';
import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { describe, it, expect } from 'vitest';

import { Session } from '../src/session.js';
import { parseAgentlog } from '../src/agentlog.js';

describe('Session', () => {
  it('writes a metadata record on enter and a trace_id on every record', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();
    await s.exit();
    const records = parseAgentlog(await readFile(path));
    expect(records).toHaveLength(1);
    expect(records[0].kind).toBe('metadata');
    expect(records[0].meta?.trace_id).toHaveLength(32);
  });

  it('recordChat appends chat_request + chat_response with linked parents', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();
    s.recordChat(
      { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] },
      {
        model: 'gpt-4o',
        content: [{ type: 'text', text: 'hello' }],
        stop_reason: 'end_turn',
        latency_ms: 150,
        usage: { input_tokens: 4, output_tokens: 1, thinking_tokens: 0 },
      },
    );
    await s.exit();
    const records = parseAgentlog(await readFile(path));
    const kinds = records.map((r) => r.kind);
    expect(kinds).toEqual(['metadata', 'chat_request', 'chat_response']);
    expect(records[2].parent).toBe(records[1].id);
  });

  it('inherits trace_id from the env when SHADOW_TRACE_ID is set', async () => {
    const parentTrace = 'a'.repeat(32);
    process.env.SHADOW_TRACE_ID = parentTrace;
    try {
      const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-'));
      const path = join(dir, 't.agentlog');
      const s = new Session({ outputPath: path, autoInstrument: false });
      expect(s.traceId).toBe(parentTrace);
      await s.enter();
      await s.exit();
      const records = parseAgentlog(await readFile(path));
      expect(records[0].meta?.trace_id).toBe(parentTrace);
    } finally {
      delete process.env.SHADOW_TRACE_ID;
    }
  });

  it('envForChild emits valid W3C traceparent', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();
    const env = s.envForChild();
    expect(env.TRACEPARENT).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);
    await s.exit();
  });
});
