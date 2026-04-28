/**
 * Tests for `gate()` — the public CI-gating composition.
 */

import { describe, it, expect } from 'vitest';

import { gate, renderGateSummary } from '../src/gate/index.js';
import { atom, globallyOp, not } from '../src/ltl/formula.js';
import type { PolicyRule } from '../src/policy/rules.js';
import type { AgentlogRecord } from '../src/agentlog.js';

let counter = 0;

function makeId(): string {
  counter += 1;
  const hex = counter.toString(16).padStart(64, '0');
  return `sha256:${hex}`;
}

function response(opts: { tools?: string[]; text?: string }): AgentlogRecord {
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
      stop_reason: 'end_turn',
      latency_ms: 0,
      usage: { input_tokens: 0, output_tokens: 0, thinking_tokens: 0 },
    },
  } as AgentlogRecord;
}

describe('gate', () => {
  it('passes a clean trace with no checks', () => {
    const result = gate([]);
    expect(result.passed).toBe(true);
    expect(result.violations).toEqual([]);
    expect(result.ltlResults).toEqual([]);
  });

  it('passes when policy and LTL both clear', () => {
    const records = [
      response({ tools: ['safe_tool'], text: 'consult a doctor' }),
    ];
    const rules: PolicyRule[] = [
      { kind: 'no_call', ruleId: 'no-delete', tool: 'delete_user' },
      { kind: 'must_include_text', ruleId: 'disclaim', substring: 'consult' },
    ];
    const result = gate(records, {
      rules,
      ltlFormulas: [globallyOp(not(atom('tool_call:delete_user')))],
    });
    expect(result.passed).toBe(true);
    expect(result.violations).toEqual([]);
    expect(result.ltlResults[0]!.passed).toBe(true);
  });

  it('fails when a policy rule fires', () => {
    const records = [response({ tools: ['delete_user'] })];
    const rules: PolicyRule[] = [
      { kind: 'no_call', ruleId: 'no-delete', tool: 'delete_user' },
    ];
    const result = gate(records, { rules });
    expect(result.passed).toBe(false);
    expect(result.violations).toHaveLength(1);
  });

  it('fails when an LTL formula fails', () => {
    const records = [response({ tools: ['delete_user'] })];
    const result = gate(records, {
      ltlFormulas: [globallyOp(not(atom('tool_call:delete_user')))],
    });
    expect(result.passed).toBe(false);
    expect(result.ltlResults[0]!.passed).toBe(false);
  });
});

describe('renderGateSummary', () => {
  it('summarises the pass case', () => {
    const text = renderGateSummary({
      passed: true,
      violations: [],
      ltlResults: [{ formula: 'G(¬tool_call:x)', passed: true }],
    });
    expect(text).toMatch(/PASSED/);
  });

  it('summarises the fail case with violations and LTL failures', () => {
    const text = renderGateSummary({
      passed: false,
      violations: [
        {
          ruleId: 'no-delete',
          kind: 'no_call',
          severity: 'high',
          detail: 'tool delete_user was called',
          pairIndex: 1,
        },
      ],
      ltlResults: [{ formula: 'G(¬tool_call:delete_user)', passed: false }],
    });
    expect(text).toContain('FAILED');
    expect(text).toContain('no-delete');
    expect(text).toContain('G(¬tool_call:delete_user)');
  });
});
