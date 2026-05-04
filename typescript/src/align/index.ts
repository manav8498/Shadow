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
// alignTraces / firstDivergence / topKDivergences — pure TS.
//
// Records-array shape matches the `.agentlog` envelope: each item
// is `{ kind, payload, ... }`. This pure-TS implementation walks
// chat_request/chat_response pairs and detects divergences via the
// existing trajectoryDistance + toolArgDelta primitives. It is NOT
// byte-identical to Shadow's Rust 9-axis differ (which factors
// embedding similarity, latency CDFs, etc.) — but for the wedge
// use cases (regression detection, tool-trajectory drift,
// argument shape changes) the TS implementation produces useful
// results without requiring Python or native bindings.
//
// For Rust-grade results, use `shadow.align` (Python) or the
// future v0.3 napi-rs binding.
// ---------------------------------------------------------------------------

interface ChatPair {
  request: Record<string, unknown>;
  response: Record<string, unknown>;
  index: number; // index in the alignment (0-based)
}

function extractChatPairs(records: readonly Record<string, unknown>[]): ChatPair[] {
  const pairs: ChatPair[] = [];
  let pendingReq: Record<string, unknown> | null = null;
  for (const rec of records) {
    const kind = rec.kind;
    if (kind === 'chat_request') {
      pendingReq = rec;
    } else if (kind === 'chat_response' && pendingReq != null) {
      pairs.push({ request: pendingReq, response: rec, index: pairs.length });
      pendingReq = null;
    }
  }
  return pairs;
}

function toolNamesFromContent(content: unknown): string[] {
  if (!Array.isArray(content)) return [];
  return content
    .filter((b): b is Record<string, unknown> => typeof b === 'object' && b !== null)
    .filter((b) => b.type === 'tool_use')
    .map((b) => String(b.name ?? ''));
}

function textFromContent(content: unknown): string {
  if (!Array.isArray(content)) return '';
  return content
    .filter((b): b is Record<string, unknown> => typeof b === 'object' && b !== null)
    .filter((b) => b.type === 'text')
    .map((b) => String(b.text ?? ''))
    .join('\n');
}

function payloadOf(rec: Record<string, unknown>): Record<string, unknown> {
  const p = rec.payload;
  return typeof p === 'object' && p !== null ? (p as Record<string, unknown>) : {};
}

/**
 * Pair every baseline chat turn to its best-match candidate turn.
 * v0.1: index-based pairing (turn N to turn N) — the fast path
 * Shadow's Rust differ uses for most real-world traces. Future
 * versions add Needleman-Wunsch with insert/delete gap costs.
 */
export function alignTraces(
  baseline: readonly Record<string, unknown>[],
  candidate: readonly Record<string, unknown>[],
): Alignment {
  const basePairs = extractChatPairs(baseline);
  const candPairs = extractChatPairs(candidate);
  const turns: AlignedTurn[] = [];
  let total = 0;
  const minLen = Math.min(basePairs.length, candPairs.length);
  for (let i = 0; i < minLen; i++) {
    const cost = pairCost(basePairs[i], candPairs[i]);
    turns.push({ baselineIndex: i, candidateIndex: i, cost });
    total += cost;
  }
  // Insertions / deletions count as cost 1.0 each (asymmetric pair
  // counts).
  for (let i = minLen; i < basePairs.length; i++) {
    turns.push({ baselineIndex: i, candidateIndex: null, cost: 1.0 });
    total += 1.0;
  }
  for (let i = minLen; i < candPairs.length; i++) {
    turns.push({ baselineIndex: null, candidateIndex: i, cost: 1.0 });
    total += 1.0;
  }
  return { turns, totalCost: total };
}

function pairCost(b: ChatPair, c: ChatPair): number {
  // Cost = max(toolNameDrift, textDrift). Bounded [0, 1].
  const bTools = toolNamesFromContent(payloadOf(b.response).content);
  const cTools = toolNamesFromContent(payloadOf(c.response).content);
  const toolCost = trajectoryDistance(bTools, cTools);
  const bText = textFromContent(payloadOf(b.response).content);
  const cText = textFromContent(payloadOf(c.response).content);
  const textCost = bText === cText ? 0.0 : bText.length === 0 || cText.length === 0 ? 1.0 : 0.5;
  return Math.max(toolCost, textCost);
}

/**
 * Find the FIRST point at which the two traces meaningfully
 * differ in alignment order, or `null` when they agree end-to-end.
 *
 * v0.1: walks chat-pairs in alignment order; emits a divergence
 * on the first non-zero pair cost or on asymmetric pair counts
 * (`structural_drift_full`).
 */
export function firstDivergence(
  baseline: readonly Record<string, unknown>[],
  candidate: readonly Record<string, unknown>[],
): Divergence | null {
  const basePairs = extractChatPairs(baseline);
  const candPairs = extractChatPairs(candidate);
  if (basePairs.length === 0 && candPairs.length === 0) return null;
  if (basePairs.length === 0 || candPairs.length === 0) {
    return {
      baselineTurn: 0,
      candidateTurn: 0,
      kind: 'structural_drift_full',
      primaryAxis: 'trajectory',
      explanation: `asymmetric corpus: baseline has ${basePairs.length} chat pair(s), candidate has ${candPairs.length}`,
      confidence: 1.0,
    };
  }
  const minLen = Math.min(basePairs.length, candPairs.length);
  for (let i = 0; i < minLen; i++) {
    const bTools = toolNamesFromContent(payloadOf(basePairs[i].response).content);
    const cTools = toolNamesFromContent(payloadOf(candPairs[i].response).content);
    const toolDist = trajectoryDistance(bTools, cTools);
    if (toolDist > 0) {
      return {
        baselineTurn: i,
        candidateTurn: i,
        kind: 'structural_drift',
        primaryAxis: 'trajectory',
        explanation: `tool sequence diverged at turn ${i} (drift=${toolDist.toFixed(2)})`,
        confidence: Math.min(1.0, toolDist),
      };
    }
    const bText = textFromContent(payloadOf(basePairs[i].response).content);
    const cText = textFromContent(payloadOf(candPairs[i].response).content);
    if (bText !== cText) {
      return {
        baselineTurn: i,
        candidateTurn: i,
        kind: 'decision_drift',
        primaryAxis: 'semantic',
        explanation: `response text diverged at turn ${i}`,
        confidence: 0.7,
      };
    }
  }
  if (basePairs.length !== candPairs.length) {
    return {
      baselineTurn: minLen,
      candidateTurn: minLen,
      kind: 'structural_drift',
      primaryAxis: 'trajectory',
      explanation: `pair-count drift: baseline=${basePairs.length}, candidate=${candPairs.length}`,
      confidence: 1.0,
    };
  }
  return null;
}

/**
 * Top-K ranked divergences. v0.1 walks all turns and emits one
 * Divergence per non-zero per-pair cost; sorts by cost descending.
 */
export function topKDivergences(
  baseline: readonly Record<string, unknown>[],
  candidate: readonly Record<string, unknown>[],
  k: number = 5,
): Divergence[] {
  if (k < 1) {
    throw new Error(`k must be >= 1, got ${k}`);
  }
  const basePairs = extractChatPairs(baseline);
  const candPairs = extractChatPairs(candidate);
  const out: Divergence[] = [];
  if (basePairs.length === 0 && candPairs.length === 0) return out;
  if (basePairs.length === 0 || candPairs.length === 0) {
    return [
      {
        baselineTurn: 0,
        candidateTurn: 0,
        kind: 'structural_drift_full',
        primaryAxis: 'trajectory',
        explanation: `asymmetric corpus: baseline=${basePairs.length} candidate=${candPairs.length}`,
        confidence: 1.0,
      },
    ].slice(0, k);
  }
  const minLen = Math.min(basePairs.length, candPairs.length);
  for (let i = 0; i < minLen; i++) {
    const bResp = payloadOf(basePairs[i].response);
    const cResp = payloadOf(candPairs[i].response);
    const bTools = toolNamesFromContent(bResp.content);
    const cTools = toolNamesFromContent(cResp.content);
    const toolDist = trajectoryDistance(bTools, cTools);
    if (toolDist > 0) {
      out.push({
        baselineTurn: i,
        candidateTurn: i,
        kind: 'structural_drift',
        primaryAxis: 'trajectory',
        explanation: `tool sequence diverged at turn ${i} (drift=${toolDist.toFixed(2)})`,
        confidence: Math.min(1.0, toolDist),
      });
      continue;
    }
    const bText = textFromContent(bResp.content);
    const cText = textFromContent(cResp.content);
    if (bText !== cText) {
      out.push({
        baselineTurn: i,
        candidateTurn: i,
        kind: 'decision_drift',
        primaryAxis: 'semantic',
        explanation: `response text diverged at turn ${i}`,
        confidence: 0.7,
      });
    }
  }
  out.sort((a, b) => b.confidence - a.confidence);
  return out.slice(0, k);
}
