/**
 * Live integration tests against real openai + @anthropic-ai/sdk packages.
 *
 * Gated on OPENAI_API_KEY / ANTHROPIC_API_KEY being set. Each test makes
 * a tiny (~100-token) real call on the cheapest frontier model to keep
 * cost under $0.01 per run.
 */

import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, it, expect } from 'vitest';

import { Session } from '../src/session.js';
import { parseAgentlog } from '../src/agentlog.js';

const HAS_OPENAI = Boolean(process.env.OPENAI_API_KEY);
const HAS_ANTHROPIC = Boolean(process.env.ANTHROPIC_API_KEY);

describe('live TS SDK integration', () => {
  it.skipIf(!HAS_OPENAI)('records real openai.chat.completions.create', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-live-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path });
    await s.enter();
    const OpenAI = (await import('openai')).default;
    const client = new OpenAI();
    const response = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      max_completion_tokens: 20,
      messages: [{ role: 'user', content: 'Reply with one word: ping' }],
    });
    await s.exit();
    expect(response.choices[0].message.content).toBeTruthy();

    const records = parseAgentlog(await readFile(path));
    const kinds = records.map((r) => r.kind);
    expect(kinds).toContain('chat_request');
    expect(kinds).toContain('chat_response');
    const resp = records.find((r) => r.kind === 'chat_response')!;
    const payload = resp.payload as {
      model: string;
      content: Array<{ type: string; text?: string }>;
      usage: { input_tokens: number; output_tokens: number };
    };
    expect(payload.model).toContain('gpt-4o-mini');
    expect(payload.content[0].type).toBe('text');
    expect(payload.usage.output_tokens).toBeGreaterThan(0);
  }, 30_000);

  it.skipIf(!HAS_ANTHROPIC)(
    'records real anthropic.messages.create',
    async () => {
      const dir = mkdtempSync(join(tmpdir(), 'shadow-ts-live-'));
      const path = join(dir, 't.agentlog');
      const s = new Session({ outputPath: path });
      await s.enter();
      const Anthropic = (await import('@anthropic-ai/sdk')).default;
      const client = new Anthropic();
      const response = await client.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 20,
        messages: [{ role: 'user', content: 'Reply with one word: ping' }],
      });
      await s.exit();
      expect(response.content[0].type).toBe('text');

      const records = parseAgentlog(await readFile(path));
      const kinds = records.map((r) => r.kind);
      expect(kinds).toContain('chat_request');
      expect(kinds).toContain('chat_response');
      const resp = records.find((r) => r.kind === 'chat_response')!;
      const payload = resp.payload as {
        model: string;
        usage: { input_tokens: number; output_tokens: number };
      };
      expect(payload.model).toMatch(/claude/);
      expect(payload.usage.output_tokens).toBeGreaterThan(0);
    },
    30_000,
  );
});
