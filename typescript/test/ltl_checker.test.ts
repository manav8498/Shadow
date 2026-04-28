/**
 * Tests for the TypeScript LTLf checker.
 *
 * Mirror the headline cases from `python/tests/test_ltl_checker.py` so
 * a reviewer can read the two side-by-side. The cross-language
 * conformance test (`python/tests/test_typescript_parity.py`) loads
 * the same fixtures into both implementations and asserts identical
 * boolean outputs.
 */

import { describe, it, expect } from 'vitest';

import {
  atom,
  not,
  and,
  or,
  implies,
  next,
  globallyOp,
  finallyOp,
  until,
  weakUntil,
  TRUE,
  FALSE,
} from '../src/ltl/formula.js';
import {
  check,
  defaultEval,
  evalAllPositions,
  traceFromRecords,
  type TraceState,
} from '../src/ltl/checker.js';

function state(opts: Partial<TraceState> & { pairIndex: number }): TraceState {
  return {
    pairIndex: opts.pairIndex,
    toolCalls: opts.toolCalls ?? [],
    stopReason: opts.stopReason ?? '',
    textContent: opts.textContent ?? '',
    extra: opts.extra ?? {},
  };
}

describe('atomic predicates', () => {
  it('matches tool_call exactly', () => {
    const trace = [state({ pairIndex: 0, toolCalls: ['lookup_order'] })];
    expect(check(atom('tool_call:lookup_order'), trace)).toBe(true);
    expect(check(atom('tool_call:refund_order'), trace)).toBe(false);
  });

  it('matches stop_reason exactly', () => {
    const trace = [state({ pairIndex: 0, stopReason: 'tool_use' })];
    expect(check(atom('stop_reason:tool_use'), trace)).toBe(true);
    expect(check(atom('stop_reason:end_turn'), trace)).toBe(false);
  });

  it('matches text_contains substring', () => {
    const trace = [state({ pairIndex: 0, textContent: 'I will not provide that' })];
    expect(check(atom('text_contains:not provide'), trace)).toBe(true);
    expect(check(atom('text_contains:absolutely will'), trace)).toBe(false);
  });

  it('honours TRUE / FALSE constants', () => {
    expect(check(TRUE, [])).toBe(true);
    expect(check(FALSE, [])).toBe(false);
  });

  it('handles extra:key=value', () => {
    const trace = [state({ pairIndex: 0, extra: { user_id: 'abc' } })];
    expect(defaultEval('extra:user_id=abc', trace[0]!)).toBe(true);
    expect(defaultEval('extra:user_id=xyz', trace[0]!)).toBe(false);
  });
});

describe('boolean operators', () => {
  it('Not flips truth', () => {
    const trace = [state({ pairIndex: 0, toolCalls: ['x'] })];
    expect(check(not(atom('tool_call:x')), trace)).toBe(false);
    expect(check(not(atom('tool_call:y')), trace)).toBe(true);
  });

  it('And + Or + Implies follow classical truth tables', () => {
    const trace = [state({ pairIndex: 0, toolCalls: ['a'], stopReason: 'end_turn' })];
    expect(check(and(atom('tool_call:a'), atom('stop_reason:end_turn')), trace)).toBe(true);
    expect(check(and(atom('tool_call:a'), atom('stop_reason:tool_use')), trace)).toBe(false);
    expect(check(or(atom('tool_call:b'), atom('stop_reason:end_turn')), trace)).toBe(true);
    expect(check(implies(atom('tool_call:b'), atom('stop_reason:tool_use')), trace)).toBe(true);
  });
});

describe('temporal operators', () => {
  it('Next looks one position ahead', () => {
    const trace = [
      state({ pairIndex: 0, toolCalls: ['a'] }),
      state({ pairIndex: 1, toolCalls: ['b'] }),
    ];
    // Python semantics: X(φ)[i] = φ[i+1] for i+1 < n; otherwise false.
    // n=2, so the only valid position with i+1<n is i=0 → φ[1].
    expect(check(next(atom('tool_call:b')), trace, 0)).toBe(true);
    expect(check(next(atom('tool_call:a')), trace, 0)).toBe(false);
  });

  it('Globally requires φ at every future position', () => {
    const trace = [
      state({ pairIndex: 0, toolCalls: ['safe'] }),
      state({ pairIndex: 1, toolCalls: ['safe'] }),
    ];
    expect(check(globallyOp(atom('tool_call:safe')), trace)).toBe(true);
  });

  it('Globally over an empty trace is vacuously true', () => {
    expect(check(globallyOp(atom('tool_call:anything')), [])).toBe(true);
  });

  it('Finally requires φ at some future position', () => {
    const trace = [
      state({ pairIndex: 0, toolCalls: ['x'] }),
      state({ pairIndex: 1, toolCalls: ['target'] }),
    ];
    expect(check(finallyOp(atom('tool_call:target')), trace)).toBe(true);
    expect(check(finallyOp(atom('tool_call:nope')), trace)).toBe(false);
  });

  it('Until: φ holds until ψ; ψ must occur', () => {
    const trace = [
      state({ pairIndex: 0, toolCalls: ['safe'] }),
      state({ pairIndex: 1, toolCalls: ['safe'] }),
      state({ pairIndex: 2, toolCalls: ['done'] }),
    ];
    expect(check(until(atom('tool_call:safe'), atom('tool_call:done')), trace)).toBe(true);
  });

  it('WeakUntil: φ holds forever, no ψ required', () => {
    const trace = [
      state({ pairIndex: 0, toolCalls: ['safe'] }),
      state({ pairIndex: 1, toolCalls: ['safe'] }),
    ];
    expect(check(weakUntil(atom('tool_call:safe'), atom('tool_call:never')), trace)).toBe(true);
  });

  it('Strong Until fails when ψ never fires', () => {
    const trace = [
      state({ pairIndex: 0, toolCalls: ['safe'] }),
      state({ pairIndex: 1, toolCalls: ['safe'] }),
    ];
    expect(check(until(atom('tool_call:safe'), atom('tool_call:never')), trace)).toBe(false);
  });
});

describe('eval_all_positions DP', () => {
  it('returns length n+1 vector', () => {
    const trace = [
      state({ pairIndex: 0 }),
      state({ pairIndex: 1 }),
    ];
    const arr = evalAllPositions(globallyOp(TRUE), trace);
    expect(arr).toHaveLength(3); // n+1 = 3
    expect(arr.every((v) => v === true)).toBe(true);
  });

  it('shares subformula caches across siblings', () => {
    // No direct way to assert cache hits without instrumentation; just
    // ensure the result is consistent for a formula that references
    // the same atom multiple times.
    const phi = and(atom('tool_call:a'), or(atom('tool_call:a'), atom('tool_call:b')));
    const trace = [state({ pairIndex: 0, toolCalls: ['a'] })];
    expect(check(phi, trace)).toBe(true);
  });
});

describe('traceFromRecords', () => {
  it('extracts tool_use names and text content', () => {
    const records = [
      {
        version: '0.1',
        id: 'sha256:0',
        kind: 'chat_request',
        ts: '2026-04-28T00:00:00.000Z',
        parent: 'sha256:none',
        meta: {},
        payload: {},
      },
      {
        version: '0.1',
        id: 'sha256:1',
        kind: 'chat_response',
        ts: '2026-04-28T00:00:01.000Z',
        parent: 'sha256:0',
        meta: {},
        payload: {
          content: [
            { type: 'text', text: 'doing the thing' },
            { type: 'tool_use', id: 't1', name: 'lookup_order', input: {} },
          ],
          stop_reason: 'tool_use',
        },
      },
    ];
    const trace = traceFromRecords(records as never);
    expect(trace).toHaveLength(1);
    expect(trace[0]!.toolCalls).toEqual(['lookup_order']);
    expect(trace[0]!.stopReason).toBe('tool_use');
    expect(trace[0]!.textContent).toBe('doing the thing');
  });
});
