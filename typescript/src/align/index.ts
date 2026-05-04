/**
 * `shadow-diff/align` — TypeScript port of `shadow.align` (Python).
 *
 * Reusable trace-comparison primitives for tools in the JS/TS
 * ecosystem (LangChain.js, AI SDK, etc.) that want Shadow's
 * alignment math without depending on the Python or Rust stack.
 *
 * Two functions ship as fully-native TypeScript:
 *   * `trajectoryDistance(a, b)` — Levenshtein on flat sequences.
 *   * `toolArgDelta(a, b)` — structural diff of two JSON values.
 *
 * Three functions (`alignTraces`, `firstDivergence`,
 * `topKDivergences`) require Shadow's Rust 9-axis differ. v0.1
 * exposes them with a clear `unavailable` stub that throws and
 * points callers at `pip install shadow-diff` for the Python
 * package, or at the v0.2 napi-rs binding once landed. Same
 * surface as Python so callers can write code today that works
 * across both.
 *
 * Cross-language parity: byte-identical results to
 * `shadow.align.trajectory_distance` and
 * `shadow.align.tool_arg_delta` on the same inputs (verified by
 * `typescript/test/align.test.ts`).
 */

// ---------------------------------------------------------------------------
// Public types — mirror Python dataclasses one-to-one.
// ---------------------------------------------------------------------------

export interface AlignedTurn {
  baselineIndex: number | null;
  candidateIndex: number | null;
  cost: number;
}

export interface Alignment {
  turns: AlignedTurn[];
  totalCost: number;
}

export interface Divergence {
  baselineTurn: number;
  candidateTurn: number;
  kind: string;
  primaryAxis: string;
  explanation: string;
  confidence: number;
}

export type ArgDeltaKind = 'added' | 'removed' | 'changed' | 'type_changed';

export interface ArgDelta {
  path: string;
  kind: ArgDeltaKind;
  old?: unknown;
  new?: unknown;
}

// ---------------------------------------------------------------------------
// trajectoryDistance — pure TypeScript, no native deps.
// ---------------------------------------------------------------------------

/**
 * Levenshtein edit distance between two flat sequences,
 * normalised to `[0.0, 1.0]` by the longer length.
 *
 * Returns `0.0` for identical sequences, `1.0` for fully
 * disjoint, and `0.0` for `(empty, empty)`. Equality is
 * `===`-based so int/string/object sequences all work.
 *
 * Cross-language parity: identical to
 * `shadow.align.trajectory_distance` in Python on the same
 * inputs.
 */
export function trajectoryDistance<T>(a: readonly T[], b: readonly T[]): number {
  if (a.length === 0 && b.length === 0) return 0.0;
  const n = a.length;
  const m = b.length;
  // Standard DP Levenshtein. O(n*m) time, O(n*m) memory.
  // For very long sequences we'd switch to two-row DP; v0.1
  // matches Python's implementation exactly for parity.
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost,
      );
    }
  }
  return dp[n][m] / Math.max(n, m);
}

// ---------------------------------------------------------------------------
// toolArgDelta — pure TypeScript, no native deps.
// ---------------------------------------------------------------------------

/**
 * Structural diff between two JSON values. Walks objects,
 * arrays, and scalars; produces typed deltas keyed by
 * slash-separated JSON-pointer paths.
 *
 * Cross-language parity: byte-identical output (modulo path
 * formatting) to `shadow.align.tool_arg_delta` in Python on
 * the same inputs.
 */
export function toolArgDelta(a: unknown, b: unknown, prefix: string = ''): ArgDelta[] {
  const out: ArgDelta[] = [];
  walkArgDelta(a, b, prefix, out);
  return out;
}

function walkArgDelta(a: unknown, b: unknown, path: string, out: ArgDelta[]): void {
  if (a === null && b === null) return;
  if (a === undefined && b === undefined) return;
  if (a == null) {
    out.push({ path: path || '/', kind: 'added', new: b });
    return;
  }
  if (b == null) {
    out.push({ path: path || '/', kind: 'removed', old: a });
    return;
  }
  // Match Python's `type(a) is not type(b)` semantics: number vs
  // string vs boolean must register as `type_changed`, not just
  // `changed`. Lumping all scalars under one `jsonType` would
  // miss this and break cross-language parity.
  const ta = preciseType(a);
  const tb = preciseType(b);
  if (ta !== tb) {
    out.push({ path: path || '/', kind: 'type_changed', old: a, new: b });
    return;
  }
  if (ta === 'object' && tb === 'object') {
    const aObj = a as Record<string, unknown>;
    const bObj = b as Record<string, unknown>;
    const keys = Array.from(new Set([...Object.keys(aObj), ...Object.keys(bObj)])).sort();
    for (const k of keys) {
      const sub = `${path}/${k}`;
      if (!(k in aObj)) {
        out.push({ path: sub, kind: 'added', new: bObj[k] });
      } else if (!(k in bObj)) {
        out.push({ path: sub, kind: 'removed', old: aObj[k] });
      } else {
        walkArgDelta(aObj[k], bObj[k], sub, out);
      }
    }
    return;
  }
  if (ta === 'array' && tb === 'array') {
    const aArr = a as unknown[];
    const bArr = b as unknown[];
    const minLen = Math.min(aArr.length, bArr.length);
    for (let i = 0; i < minLen; i++) {
      walkArgDelta(aArr[i], bArr[i], `${path}/${i}`, out);
    }
    for (let i = aArr.length; i < bArr.length; i++) {
      out.push({ path: `${path}/${i}`, kind: 'added', new: bArr[i] });
    }
    for (let i = bArr.length; i < aArr.length; i++) {
      out.push({ path: `${path}/${i}`, kind: 'removed', old: aArr[i] });
    }
    return;
  }
  if (a !== b) {
    out.push({ path: path || '/', kind: 'changed', old: a, new: b });
  }
}

function jsonType(v: unknown): 'object' | 'array' | 'scalar' {
  if (Array.isArray(v)) return 'array';
  if (v !== null && typeof v === 'object') return 'object';
  return 'scalar';
}

/**
 * Per-scalar type discriminator that matches Python's `type()`
 * semantics: number vs string vs boolean must register as
 * `type_changed` to maintain cross-language parity with
 * `shadow.align.tool_arg_delta`.
 */
function preciseType(v: unknown): string {
  if (Array.isArray(v)) return 'array';
  if (v === null) return 'null';
  return typeof v; // 'object', 'string', 'number', 'boolean', etc.
}

// ---------------------------------------------------------------------------
// alignTraces / firstDivergence / topKDivergences — Rust-backed.
//
// v0.1 stubs. The actual Rust 9-axis differ that does the
// alignment lives in the Python wheel (`shadow._core`); a
// future v0.2 of this package adds a napi-rs binding so JS
// callers can invoke it without spawning Python.
// ---------------------------------------------------------------------------

/**
 * v0.1 stub. Returns an Alignment whose only signal is the
 * baseline trace's pair count; throws when the rich pairing
 * is needed. Use the Python package for full alignment until
 * the v0.2 napi-rs binding lands.
 */
export function alignTraces(_baseline: unknown[], _candidate: unknown[]): Alignment {
  throw new Error(
    'alignTraces requires the Rust 9-axis differ. v0.1 of @shadow-diff/align ' +
      'ships pure-TS trajectoryDistance + toolArgDelta only. For full alignment, ' +
      'use the Python package: `pip install shadow-diff`. Native TS binding is ' +
      'tracked for v0.2.',
  );
}

/**
 * v0.1 stub — see `alignTraces` for migration path.
 */
export function firstDivergence(
  _baseline: unknown[],
  _candidate: unknown[],
): Divergence | null {
  throw new Error(
    'firstDivergence requires the Rust 9-axis differ. v0.1 of @shadow-diff/align ' +
      'ships pure-TS trajectoryDistance + toolArgDelta only. For full alignment, ' +
      'use the Python package: `pip install shadow-diff`. Native TS binding is ' +
      'tracked for v0.2.',
  );
}

/**
 * v0.1 stub — see `alignTraces` for migration path.
 */
export function topKDivergences(
  _baseline: unknown[],
  _candidate: unknown[],
  _k: number = 5,
): Divergence[] {
  throw new Error(
    'topKDivergences requires the Rust 9-axis differ. v0.1 of @shadow-diff/align ' +
      'ships pure-TS trajectoryDistance + toolArgDelta only. For full alignment, ' +
      'use the Python package: `pip install shadow-diff`. Native TS binding is ' +
      'tracked for v0.2.',
  );
}
