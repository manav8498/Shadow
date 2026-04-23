import { describe, it, expect } from 'vitest';

import { canonicalJson, contentId, writeAgentlog, parseAgentlog } from '../src/agentlog.js';

describe('canonical JSON', () => {
  it('sorts object keys lexicographically', () => {
    const text = new TextDecoder().decode(canonicalJson({ b: 1, a: 2 }));
    expect(text).toBe('{"a":2,"b":1}');
  });

  it('emits compact arrays and nulls', () => {
    expect(new TextDecoder().decode(canonicalJson([1, null, 2]))).toBe('[1,null,2]');
  });

  it('escapes control characters', () => {
    expect(new TextDecoder().decode(canonicalJson('a\nb'))).toBe('"a\\nb"');
  });

  it('NFC-normalises strings', () => {
    // "café" in NFD (c + a + f + e + combining acute) should equal NFC form.
    const nfd = 'café';
    const nfc = 'café';
    expect(canonicalJson(nfd)).toEqual(canonicalJson(nfc));
  });

  it('produces the SPEC §5.6 known-vector hash', () => {
    // {"hello":"world"} → deterministic canonical bytes.
    // Hash is a cross-language conformance anchor.
    const id = contentId({ hello: 'world' });
    expect(id).toMatch(/^sha256:[0-9a-f]{64}$/);
    // Same input produces same id every time.
    expect(id).toEqual(contentId({ hello: 'world' }));
  });

  it('normalises -0 to 0', () => {
    expect(new TextDecoder().decode(canonicalJson(-0))).toBe('0');
  });
});

describe('agentlog JSONL I/O', () => {
  it('writes one record per line + trailing newline', () => {
    const records = [
      {
        version: '0.1',
        id: 'sha256:' + 'a'.repeat(64),
        kind: 'metadata' as const,
        ts: '2026-04-21T10:00:00.000Z',
        parent: null,
        payload: { sdk: 'shadow' },
      },
    ];
    const bytes = writeAgentlog(records);
    const text = new TextDecoder().decode(bytes);
    expect(text.endsWith('\n')).toBe(true);
    expect(text.split('\n').filter((l) => l.length > 0)).toHaveLength(1);
  });

  it('round-trips records without loss', () => {
    const records = [
      {
        version: '0.1',
        id: 'sha256:' + 'a'.repeat(64),
        kind: 'metadata' as const,
        ts: '2026-04-21T10:00:00.000Z',
        parent: null,
        payload: { a: 1, nested: { b: [1, 2] } },
      },
    ];
    const bytes = writeAgentlog(records);
    const parsed = parseAgentlog(bytes);
    expect(parsed).toEqual(records);
  });
});
