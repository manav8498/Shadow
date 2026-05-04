/**
 * Tests for the TS port of `shadow.align`.
 *
 * Cross-language parity: each test mirrors a Python test in
 * `python/tests/test_align_library.py`. Outputs must match
 * byte-for-byte (modulo Python's snake_case vs TS camelCase
 * dataclass field names) on the same inputs.
 */

import { describe, expect, it } from 'vitest';

import {
  alignTraces,
  firstDivergence,
  toolArgDelta,
  topKDivergences,
  trajectoryDistance,
} from '../src/align/index.js';

// ---------------------------------------------------------------------------
// trajectoryDistance — pure TS, matches Python
// ---------------------------------------------------------------------------

describe('trajectoryDistance', () => {
  it('equal sequences return 0', () => {
    expect(trajectoryDistance(['a', 'b', 'c'], ['a', 'b', 'c'])).toBe(0.0);
  });

  it('completely disjoint sequences return 1', () => {
    expect(trajectoryDistance(['a', 'b'], ['x', 'y'])).toBe(1.0);
  });

  it('one substitution in three is one third', () => {
    const d = trajectoryDistance(['a', 'b', 'c'], ['a', 'x', 'c']);
    expect(d).toBeCloseTo(1 / 3, 2);
  });

  it('extra tool at end counted as one insertion', () => {
    const d = trajectoryDistance(['a', 'b'], ['a', 'b', 'c']);
    expect(d).toBeCloseTo(1 / 3, 2);
  });

  it('both empty returns 0', () => {
    expect(trajectoryDistance([], [])).toBe(0.0);
  });

  it('one empty returns 1', () => {
    expect(trajectoryDistance(['a'], [])).toBe(1.0);
  });

  it('handles non-string elements (numbers) like Python', () => {
    expect(trajectoryDistance([1, 2, 3], [1, 2, 3])).toBe(0.0);
    const d = trajectoryDistance([1, 2, 3], [1, 9, 3]);
    expect(d).toBeCloseTo(1 / 3, 2);
  });
});

// ---------------------------------------------------------------------------
// toolArgDelta — pure TS, matches Python
// ---------------------------------------------------------------------------

describe('toolArgDelta', () => {
  it('equal objects produce no deltas', () => {
    expect(toolArgDelta({ a: 1, b: 2 }, { a: 1, b: 2 })).toEqual([]);
  });

  it('added key', () => {
    const deltas = toolArgDelta({ a: 1 }, { a: 1, b: 2 });
    expect(deltas).toHaveLength(1);
    expect(deltas[0].path).toBe('/b');
    expect(deltas[0].kind).toBe('added');
    expect(deltas[0].new).toBe(2);
  });

  it('removed key', () => {
    const deltas = toolArgDelta({ a: 1, b: 2 }, { a: 1 });
    expect(deltas[0].kind).toBe('removed');
    expect(deltas[0].old).toBe(2);
  });

  it('changed value', () => {
    const deltas = toolArgDelta({ x: 1 }, { x: 2 });
    expect(deltas[0].kind).toBe('changed');
    expect(deltas[0].old).toBe(1);
    expect(deltas[0].new).toBe(2);
  });

  it('type changed', () => {
    const deltas = toolArgDelta({ x: 1 }, { x: '1' });
    expect(deltas[0].kind).toBe('type_changed');
  });

  it('nested dict changes get nested paths', () => {
    const deltas = toolArgDelta(
      { outer: { inner: 'old' } },
      { outer: { inner: 'new' } },
    );
    expect(deltas).toHaveLength(1);
    expect(deltas[0].path).toBe('/outer/inner');
    expect(deltas[0].kind).toBe('changed');
  });

  it('list with appended item', () => {
    const deltas = toolArgDelta({ items: [1, 2] }, { items: [1, 2, 3] });
    expect(deltas).toHaveLength(1);
    expect(deltas[0].path).toBe('/items/2');
    expect(deltas[0].kind).toBe('added');
    expect(deltas[0].new).toBe(3);
  });

  it('list index change', () => {
    const deltas = toolArgDelta([1, 2, 3], [1, 9, 3]);
    expect(deltas).toHaveLength(1);
    expect(deltas[0].path).toBe('/1');
    expect(deltas[0].old).toBe(2);
    expect(deltas[0].new).toBe(9);
  });

  it('null/undefined inputs handled', () => {
    expect(toolArgDelta(null, null)).toEqual([]);
    expect(toolArgDelta({ x: 1 }, null)).toHaveLength(1);
    expect(toolArgDelta(null, { x: 1 })).toHaveLength(1);
  });

  it('100-deep nested structure stack-safe', () => {
    let a: unknown = 'old';
    let b: unknown = 'new';
    for (let i = 0; i < 100; i++) {
      a = { inner: a };
      b = { inner: b };
    }
    const deltas = toolArgDelta(a, b);
    expect(deltas).toHaveLength(1);
    // Path is "/inner/inner/.../inner" with 100 segments.
    expect(deltas[0].path.split('/').filter(Boolean)).toHaveLength(100);
  });
});

// ---------------------------------------------------------------------------
// Pure-TS alignTraces / firstDivergence / topKDivergences (v0.2)
// ---------------------------------------------------------------------------

function makePair(toolName: string | null, text: string = '') {
  const content = toolName
    ? [{ type: 'tool_use', name: toolName, input: {}, id: '1' }]
    : [{ type: 'text', text }];
  return [
    { kind: 'chat_request', payload: { messages: [{ role: 'user', content: 'q' }] } },
    { kind: 'chat_response', payload: { content } },
  ];
}

describe('alignTraces (pure TS)', () => {
  it('returns empty alignment for two empty inputs', () => {
    const aln = alignTraces([], []);
    expect(aln.turns).toEqual([]);
    expect(aln.totalCost).toBe(0);
  });

  it('pairs index-by-index for equal-length symmetric pairs', () => {
    const records = [...makePair('search'), ...makePair('summarize')];
    const aln = alignTraces(records, records);
    expect(aln.turns).toHaveLength(2);
    expect(aln.totalCost).toBe(0);
  });

  it('asymmetric pair counts emit gap turns with cost 1', () => {
    const a = makePair('search');
    const b = [...makePair('search'), ...makePair('extra')];
    const aln = alignTraces(a, b);
    expect(aln.turns).toHaveLength(2);
    expect(aln.turns[1].baselineIndex).toBeNull();
    expect(aln.turns[1].cost).toBe(1.0);
  });
});

describe('firstDivergence (pure TS)', () => {
  it('returns null for identical traces', () => {
    const records = makePair('search');
    expect(firstDivergence(records, records)).toBeNull();
  });

  it('detects asymmetric corpus as structural_drift_full', () => {
    const fd = firstDivergence(makePair('search'), []);
    expect(fd).not.toBeNull();
    expect(fd!.kind).toBe('structural_drift_full');
    expect(fd!.primaryAxis).toBe('trajectory');
  });

  it('detects tool-sequence drift as structural_drift', () => {
    const fd = firstDivergence(makePair('search'), makePair('summarize'));
    expect(fd).not.toBeNull();
    expect(fd!.kind).toBe('structural_drift');
    expect(fd!.primaryAxis).toBe('trajectory');
  });

  it('detects text drift as decision_drift', () => {
    const fd = firstDivergence(makePair(null, 'hello'), makePair(null, 'hi there'));
    expect(fd).not.toBeNull();
    expect(fd!.kind).toBe('decision_drift');
    expect(fd!.primaryAxis).toBe('semantic');
  });
});

describe('topKDivergences (pure TS)', () => {
  it('returns empty for identical', () => {
    expect(topKDivergences(makePair('search'), makePair('search'))).toEqual([]);
  });

  it('caps at k', () => {
    const a = [...makePair('a'), ...makePair('b'), ...makePair('c')];
    const b = [...makePair('x'), ...makePair('y'), ...makePair('z')];
    const out = topKDivergences(a, b, 2);
    expect(out.length).toBeLessThanOrEqual(2);
  });

  it('rejects k<1', () => {
    expect(() => topKDivergences([], [], 0)).toThrow(/k must be >= 1/);
  });
});

// ---------------------------------------------------------------------------
// Cross-language parity assertions
// ---------------------------------------------------------------------------

describe('cross-language parity (vs Python shadow.align)', () => {
  it('trajectoryDistance produces the same numbers as Python', () => {
    // These match the Python tests one-for-one. If a result here
    // changes, update Python too — drift between the two ports is
    // the bug.
    const cases: [string[], string[], number][] = [
      [['a', 'b', 'c'], ['a', 'b', 'c'], 0.0],
      [['a', 'b'], ['x', 'y'], 1.0],
      [['a', 'b', 'c'], ['a', 'x', 'c'], 1 / 3],
      [[], [], 0.0],
      [['a'], [], 1.0],
    ];
    for (const [a, b, expected] of cases) {
      expect(trajectoryDistance(a, b)).toBeCloseTo(expected, 5);
    }
  });

  it('toolArgDelta path format matches Python (slash-separated)', () => {
    const deltas = toolArgDelta({ a: { b: 1 } }, { a: { b: 2 } });
    // Same path Python emits: "/a/b"
    expect(deltas[0].path).toBe('/a/b');
  });
});
