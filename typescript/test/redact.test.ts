import { describe, it, expect } from 'vitest';
import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Redactor, luhnValid } from '../src/redact.js';
import { Session } from '../src/session.js';
import { parseAgentlog } from '../src/agentlog.js';

describe('Redactor', () => {
  it('redacts OpenAI project keys', () => {
    const r = new Redactor();
    const out = r.redactValue('key=sk-proj-ABC123def456ghi789JKL012mno345');
    expect(out).not.toContain('sk-proj-');
    expect(out).toContain('[REDACTED:openai_api_key]');
  });

  it('redacts Anthropic keys before OpenAI pattern', () => {
    const r = new Redactor();
    const out = r.redactValue(
      'Authorization: Bearer sk-ant-api03-deadbeef' + 'a'.repeat(50),
    );
    expect(out).not.toContain('sk-ant-api03-');
    expect(out).toContain('[REDACTED:anthropic_api_key]');
  });

  it('redacts AWS access key IDs', () => {
    const r = new Redactor();
    const out = r.redactValue('creds: AKIAIOSFODNN7EXAMPLE for bucket x');
    expect(out).not.toContain('AKIAIOSFODNN7EXAMPLE');
  });

  it('redacts GitHub personal access tokens', () => {
    const r = new Redactor();
    const token = 'ghp_' + 'a'.repeat(36);
    const out = r.redactValue(`token=${token}`);
    expect(out).not.toContain(token);
  });

  it('redacts PEM private keys', () => {
    const r = new Redactor();
    const body = 'X'.repeat(60);
    const pem = `-----BEGIN RSA PRIVATE KEY-----\n${body}\n-----END RSA PRIVATE KEY-----`;
    const out = r.redactValue(`here is a key:\n${pem}\nafter`);
    expect(out).not.toContain('BEGIN RSA PRIVATE KEY');
    expect(out).not.toContain(body);
  });

  it('redacts JWTs', () => {
    const r = new Redactor();
    const jwt =
      'eyJhbGciOiJIUzI1NiJ9.' +
      'eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.' +
      'SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c';
    const out = r.redactValue(`Authorization: Bearer ${jwt}`);
    expect(out).not.toContain(jwt);
  });

  it('redacts emails', () => {
    const r = new Redactor();
    const out = r.redactValue('contact alice.smith@example.com for details');
    expect(out).not.toContain('alice.smith@example.com');
  });

  it('redacts E.164 phone numbers', () => {
    const r = new Redactor();
    const out = r.redactValue('Call +14155551234 to confirm');
    expect(out).not.toContain('+14155551234');
  });

  it('redacts Luhn-valid credit card PANs (contiguous and hyphenated)', () => {
    const r = new Redactor();
    expect(r.redactValue('card 4111-1111-1111-1111 on file')).not.toContain('4111-1111');
    expect(r.redactValue('pan 4111111111111111')).not.toContain('4111111111111111');
  });

  it('redacts Amex 4-6-5 format', () => {
    const r = new Redactor();
    // 371449635398431 = valid Amex test card in 4-6-5 grouping.
    const out = r.redactValue('amex: 3714-496353-98431 today');
    expect(out).not.toContain('3714-496353-98431');
  });

  it('does NOT redact 20-digit Luhn-valid strings (out of PAN range)', () => {
    const r = new Redactor();
    const out = r.redactValue('identifier 0000 4111 1111 1111 1111 end');
    expect(out).toContain('0000 4111 1111 1111 1111');
  });

  it('does NOT redact Luhn-failing 16-digit strings', () => {
    const r = new Redactor();
    const out = r.redactValue('tracking 1234567890123456 shipment');
    expect(out).toContain('1234567890123456');
  });

  it('recurses into nested objects and arrays', () => {
    const r = new Redactor();
    const payload = {
      user: { email: 'bob@example.com', phone: '+447911123456' },
      cards: [{ pan: '4111-1111-1111-1111' }],
    };
    const out = r.redactValue(payload);
    const flat = JSON.stringify(out);
    expect(flat).not.toContain('bob@example.com');
    expect(flat).not.toContain('+447911123456');
    expect(flat).not.toContain('4111-1111-1111-1111');
  });

  it('is idempotent', () => {
    const r = new Redactor();
    const once = r.redactValue({ email: 'a@b.com' });
    const twice = r.redactValue(once);
    expect(twice).toEqual(once);
  });

  it('bypasses redaction for allowlisted keys', () => {
    const payload = {
      internal_email: 'ops@company.com',
      user_email: 'alice@x.com',
    };
    const bothRed = new Redactor().redactValue(payload);
    expect(JSON.stringify(bothRed)).not.toContain('ops@company.com');
    expect(JSON.stringify(bothRed)).not.toContain('alice@x.com');
    const oneRed = new Redactor({
      allowlistKeys: new Set(['internal_email']),
    }).redactValue(payload);
    expect(JSON.stringify(oneRed)).toContain('ops@company.com');
    expect(JSON.stringify(oneRed)).not.toContain('alice@x.com');
  });

  it('luhnValid agrees with Python on canonical test cards', () => {
    expect(luhnValid('4111111111111111')).toBe(true); // Visa test
    expect(luhnValid('371449635398431')).toBe(true); // Amex test
    expect(luhnValid('378282246310005')).toBe(true); // Amex 2
    expect(luhnValid('1234567890123456')).toBe(false);
  });
});

describe('Session applies redaction by default', () => {
  it('redacts API keys before writing to disk', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-redact-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({ outputPath: path, autoInstrument: false });
    await s.enter();
    s.recordChat(
      {
        model: 'gpt-4o',
        messages: [
          { role: 'user', content: 'my key is sk-proj-ABC123def456ghi789JKL012mno345' },
        ],
      },
      {
        model: 'gpt-4o',
        content: [{ type: 'text', text: 'Paris is the capital of France.' }],
        stop_reason: 'end_turn',
        latency_ms: 150,
        usage: { input_tokens: 4, output_tokens: 1, thinking_tokens: 0 },
      },
    );
    await s.exit();
    const raw = await readFile(path, 'utf8');
    expect(raw).not.toContain('sk-proj-ABC123');
    const records = parseAgentlog(await readFile(path));
    const req = records.find((r) => r.kind === 'chat_request')!;
    const payload = req.payload as { messages: Array<{ content: string }> };
    expect(payload.messages[0].content).toContain('[REDACTED:openai_api_key]');
  });

  it('can be disabled by passing redactor: null', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-noredact-'));
    const path = join(dir, 't.agentlog');
    const s = new Session({
      outputPath: path,
      autoInstrument: false,
      redactor: null,
    });
    await s.enter();
    s.recordChat(
      { model: 'x', messages: [{ role: 'user', content: 'alice@example.com' }] },
      {
        model: 'x',
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        latency_ms: 1,
        usage: { input_tokens: 1, output_tokens: 1, thinking_tokens: 0 },
      },
    );
    await s.exit();
    const raw = await readFile(path, 'utf8');
    expect(raw).toContain('alice@example.com');
  });
});
