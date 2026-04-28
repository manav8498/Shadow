/**
 * Tests for the TypeScript policy evaluator.
 *
 * Covers each rule kind on representative fixtures. The cross-language
 * conformance suite (`python/tests/test_typescript_parity.py`) verifies
 * decisions match the Python implementation byte-for-byte.
 */

import { describe, it, expect } from 'vitest';

import { checkPolicy } from '../src/policy/check.js';
import type { PolicyRule } from '../src/policy/rules.js';
import type { AgentlogRecord } from '../src/agentlog.js';

let counter = 0;

function makeId(): string {
  counter += 1;
  const hex = counter.toString(16).padStart(64, '0');
  return `sha256:${hex}`;
}

function response(opts: { tools?: string[]; text?: string; stopReason?: string }): AgentlogRecord {
  const content: Array<Record<string, unknown>> = [];
  if (opts.text) content.push({ type: 'text', text: opts.text });
  for (const t of opts.tools ?? []) {
    content.push({ type: 'tool_use', id: makeId(), name: t, input: {} });
  }
  return {
    version: '0.1',
    id: makeId(),
    kind: 'chat_response',
    ts: '2026-04-28T00:00:00.000Z',
    parent: 'sha256:0000000000000000000000000000000000000000000000000000000000000000',
    meta: {},
    payload: {
      model: 't',
      content,
      stop_reason: opts.stopReason ?? 'tool_use',
      latency_ms: 0,
      usage: { input_tokens: 0, output_tokens: 0, thinking_tokens: 0 },
    },
  } as AgentlogRecord;
}

describe('no_call', () => {
  it('flags every occurrence of the forbidden tool', () => {
    const records = [
      response({ tools: ['safe'] }),
      response({ tools: ['delete_user'] }),
      response({ tools: ['delete_user'] }),
    ];
    const rules: PolicyRule[] = [
      { kind: 'no_call', ruleId: 'no-delete', tool: 'delete_user' },
    ];
    const v = checkPolicy(records, rules);
    expect(v).toHaveLength(2);
    expect(v[0]!.pairIndex).toBe(1);
    expect(v[1]!.pairIndex).toBe(2);
  });

  it('ignores tools other than the named one', () => {
    const records = [response({ tools: ['lookup'] })];
    expect(
      checkPolicy(records, [{ kind: 'no_call', ruleId: 'r', tool: 'execute_sql' }]),
    ).toEqual([]);
  });
});

describe('must_call_before', () => {
  it('passes when first precedes then', () => {
    const records = [
      response({ tools: ['verify_user'] }),
      response({ tools: ['issue_refund'] }),
    ];
    const rules: PolicyRule[] = [
      {
        kind: 'must_call_before',
        ruleId: 'verify-then-refund',
        first: 'verify_user',
        then: 'issue_refund',
      },
    ];
    expect(checkPolicy(records, rules)).toEqual([]);
  });

  it('fails when then precedes first', () => {
    const records = [
      response({ tools: ['issue_refund'] }),
      response({ tools: ['verify_user'] }),
    ];
    const rules: PolicyRule[] = [
      {
        kind: 'must_call_before',
        ruleId: 'verify-then-refund',
        first: 'verify_user',
        then: 'issue_refund',
      },
    ];
    const v = checkPolicy(records, rules);
    expect(v).toHaveLength(1);
    expect(v[0]!.pairIndex).toBe(0);
  });

  it('fails when then fires without first', () => {
    const records = [response({ tools: ['issue_refund'] })];
    const rules: PolicyRule[] = [
      {
        kind: 'must_call_before',
        ruleId: 'r',
        first: 'verify_user',
        then: 'issue_refund',
      },
    ];
    expect(checkPolicy(records, rules)).toHaveLength(1);
  });

  it('vacuous when then never fires', () => {
    const records = [response({ tools: ['verify_user'] })];
    const rules: PolicyRule[] = [
      {
        kind: 'must_call_before',
        ruleId: 'r',
        first: 'verify_user',
        then: 'issue_refund',
      },
    ];
    expect(checkPolicy(records, rules)).toEqual([]);
  });
});

describe('must_call_once', () => {
  it('flags duplicate calls beyond the first', () => {
    const records = [
      response({ tools: ['lookup'] }),
      response({ tools: ['lookup'] }),
      response({ tools: ['lookup'] }),
    ];
    const rules: PolicyRule[] = [
      { kind: 'must_call_once', ruleId: 'lookup-once', tool: 'lookup' },
    ];
    const v = checkPolicy(records, rules);
    expect(v).toHaveLength(2);
    expect(v[0]!.pairIndex).toBe(1);
    expect(v[1]!.pairIndex).toBe(2);
  });

  it('vacuous when tool appears 0 or 1 times', () => {
    const rules: PolicyRule[] = [
      { kind: 'must_call_once', ruleId: 'r', tool: 'lookup' },
    ];
    expect(checkPolicy([], rules)).toEqual([]);
    expect(checkPolicy([response({ tools: ['lookup'] })], rules)).toEqual([]);
  });
});

describe('forbidden_text', () => {
  it('flags responses containing the substring', () => {
    const records = [response({ text: 'sure, here is your refund' })];
    const rules: PolicyRule[] = [
      { kind: 'forbidden_text', ruleId: 'no-refund-promise', substring: 'refund' },
    ];
    const v = checkPolicy(records, rules);
    expect(v).toHaveLength(1);
    expect(v[0]!.pairIndex).toBe(0);
  });
});

describe('must_include_text', () => {
  it('passes when at least one response contains the substring', () => {
    const records = [
      response({ text: 'consult a clinician' }),
      response({ text: 'consult your doctor' }),
    ];
    const rules: PolicyRule[] = [
      { kind: 'must_include_text', ruleId: 'must-disclaim', substring: 'consult' },
    ];
    expect(checkPolicy(records, rules)).toEqual([]);
  });

  it('fails with pairIndex=null when no response contains the substring', () => {
    const records = [response({ text: 'no disclaimer here' })];
    const rules: PolicyRule[] = [
      { kind: 'must_include_text', ruleId: 'must-disclaim', substring: 'consult' },
    ];
    const v = checkPolicy(records, rules);
    expect(v).toHaveLength(1);
    expect(v[0]!.pairIndex).toBe(null);
  });
});

describe('output ordering', () => {
  it('sorts by pairIndex then ruleId', () => {
    const records = [
      response({ tools: ['z'] }),
      response({ tools: ['a'] }),
    ];
    const rules: PolicyRule[] = [
      { kind: 'no_call', ruleId: 'rule-z', tool: 'z' },
      { kind: 'no_call', ruleId: 'rule-a', tool: 'a' },
    ];
    const v = checkPolicy(records, rules);
    expect(v.map((x) => [x.pairIndex, x.ruleId])).toEqual([
      [0, 'rule-z'],
      [1, 'rule-a'],
    ]);
  });
});
