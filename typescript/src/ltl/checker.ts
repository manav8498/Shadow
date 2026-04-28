/**
 * Finite-trace LTL (LTLf) model checker.
 *
 * Bottom-up dynamic programming, byte-for-byte mirror of the Python
 * implementation in `shadow.ltl.checker`. For each subformula φ' we
 * compute a length-(n+1) boolean vector `holds[i]` once, then look up
 * the result for any starting position in O(1).
 *
 * Recurrences (n = trace length):
 *   G(φ)[i]    = φ[i] ∧ G(φ)[i+1]   ; G(φ)[n] = true
 *   F(φ)[i]    = φ[i] ∨ F(φ)[i+1]   ; F(φ)[n] = false
 *   (φ U ψ)[i] = ψ[i] ∨ (φ[i] ∧ U[i+1])    ; U[n]  = false
 *   (φ W ψ)[i] = ψ[i] ∨ (φ[i] ∧ W[i+1])    ; W[n]  = true
 *   X(φ)[i]    = φ[i+1] for i < n; X(φ)[n] = false; X(φ)[n-1] = false
 *
 * Total work: O(|π| × |φ|) where |φ| is the number of distinct AST nodes.
 */

import type { Formula } from './formula.js';
import type { AgentlogRecord } from '../agentlog.js';

export interface TraceState {
  pairIndex: number;
  toolCalls: string[];
  stopReason: string;
  textContent: string;
  extra?: Record<string, unknown>;
}

export type EvalFn = (pred: string, state: TraceState) => boolean;

/**
 * Built-in atomic-predicate evaluator. Mirrors `shadow.ltl.checker.default_eval`
 * exactly:
 *   "true" / "false"
 *   "tool_call:<name>"
 *   "stop_reason:<value>"
 *   "text_contains:<substr>"
 *   "extra:<key>=<value>"
 */
export function defaultEval(pred: string, state: TraceState): boolean {
  if (pred === 'true') return true;
  if (pred === 'false') return false;
  if (pred.startsWith('tool_call:')) {
    const name = pred.slice('tool_call:'.length);
    return state.toolCalls.includes(name);
  }
  if (pred.startsWith('stop_reason:')) {
    const value = pred.slice('stop_reason:'.length);
    return state.stopReason === value;
  }
  if (pred.startsWith('text_contains:')) {
    const substr = pred.slice('text_contains:'.length);
    return state.textContent.includes(substr);
  }
  if (pred.startsWith('extra:')) {
    const rest = pred.slice('extra:'.length);
    const eq = rest.indexOf('=');
    if (eq >= 0) {
      const k = rest.slice(0, eq);
      const v = rest.slice(eq + 1);
      const have = state.extra?.[k];
      return String(have ?? '') === v;
    }
  }
  return false;
}

/**
 * Compute the truth value of `formula` at every position 0..n, returning
 * a length-(n+1) array. `out[n]` is the end-of-trace value (used for
 * vacuous Globally / WeakUntil semantics).
 */
export function evalAllPositions(
  formula: Formula,
  trace: TraceState[],
  evalFn: EvalFn = defaultEval,
): boolean[] {
  const n = trace.length;
  const cache = new Map<Formula, boolean[]>();
  return holds(formula, trace, n, evalFn, cache);
}

/**
 * Evaluate `formula` on `trace` starting at position `i` (default 0).
 */
export function check(
  formula: Formula,
  trace: TraceState[],
  i: number = 0,
  evalFn: EvalFn = defaultEval,
): boolean {
  const arr = evalAllPositions(formula, trace, evalFn);
  if (i < 0 || i > trace.length) {
    throw new RangeError(`position ${i} out of range for trace of length ${trace.length}`);
  }
  return arr[i]!;
}

function holds(
  phi: Formula,
  trace: TraceState[],
  n: number,
  evalFn: EvalFn,
  cache: Map<Formula, boolean[]>,
): boolean[] {
  const cached = cache.get(phi);
  if (cached !== undefined) return cached;

  let out: boolean[];

  switch (phi.kind) {
    case 'atom': {
      // End-of-trace: only the literal "true" holds.
      out = new Array<boolean>(n + 1);
      for (let i = 0; i < n; i++) out[i] = evalFn(phi.pred, trace[i]!);
      out[n] = phi.pred === 'true';
      break;
    }
    case 'not': {
      const child = holds(phi.child, trace, n, evalFn, cache);
      out = child.map((v) => !v);
      break;
    }
    case 'and': {
      const a = holds(phi.left, trace, n, evalFn, cache);
      const b = holds(phi.right, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1);
      for (let i = 0; i <= n; i++) out[i] = a[i]! && b[i]!;
      break;
    }
    case 'or': {
      const a = holds(phi.left, trace, n, evalFn, cache);
      const b = holds(phi.right, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1);
      for (let i = 0; i <= n; i++) out[i] = a[i]! || b[i]!;
      break;
    }
    case 'implies': {
      const a = holds(phi.left, trace, n, evalFn, cache);
      const b = holds(phi.right, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1);
      for (let i = 0; i <= n; i++) out[i] = !a[i]! || b[i]!;
      break;
    }
    case 'next': {
      const child = holds(phi.child, trace, n, evalFn, cache);
      // X(φ)[i] = φ[i+1] for i+1 < n; otherwise false. Matches the
      // Python rule out[i] = child[i+1] if (i+1) < n else false.
      out = new Array<boolean>(n + 1).fill(false);
      for (let i = 0; i < n; i++) {
        out[i] = i + 1 < n ? child[i + 1]! : false;
      }
      out[n] = false;
      break;
    }
    case 'globally': {
      // G(φ)[i] = φ[i] ∧ G(φ)[i+1]; G(φ)[n] = true
      const child = holds(phi.child, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1).fill(true);
      for (let i = n - 1; i >= 0; i--) out[i] = child[i]! && out[i + 1]!;
      break;
    }
    case 'finally': {
      // F(φ)[i] = φ[i] ∨ F(φ)[i+1]; F(φ)[n] = false
      const child = holds(phi.child, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1).fill(false);
      for (let i = n - 1; i >= 0; i--) out[i] = child[i]! || out[i + 1]!;
      break;
    }
    case 'until': {
      // (φ U ψ)[i] = ψ[i] ∨ (φ[i] ∧ U[i+1]); U[n] = false
      const a = holds(phi.left, trace, n, evalFn, cache);
      const b = holds(phi.right, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1).fill(false);
      for (let i = n - 1; i >= 0; i--) out[i] = b[i]! || (a[i]! && out[i + 1]!);
      break;
    }
    case 'weakUntil': {
      // (φ W ψ)[i] = ψ[i] ∨ (φ[i] ∧ W[i+1]); W[n] = true
      const a = holds(phi.left, trace, n, evalFn, cache);
      const b = holds(phi.right, trace, n, evalFn, cache);
      out = new Array<boolean>(n + 1).fill(true);
      for (let i = n - 1; i >= 0; i--) out[i] = b[i]! || (a[i]! && out[i + 1]!);
      break;
    }
  }

  cache.set(phi, out);
  return out;
}

/**
 * Build a list of TraceState from agentlog records. One TraceState
 * per `chat_response` record. Mirrors `shadow.ltl.checker.trace_from_records`.
 */
export function traceFromRecords(records: AgentlogRecord[]): TraceState[] {
  const states: TraceState[] = [];
  let pairIdx = 0;
  for (const rec of records) {
    if (rec.kind !== 'chat_response') continue;
    const payload = (rec.payload ?? {}) as Record<string, unknown>;
    const content = (payload.content ?? []) as Array<Record<string, unknown>>;
    const toolCalls: string[] = [];
    const textParts: string[] = [];
    for (const block of content) {
      if (typeof block !== 'object' || block === null) continue;
      const type = block.type;
      if (type === 'tool_use') {
        toolCalls.push(String(block.name ?? ''));
      } else if (type === 'text') {
        const t = block.text;
        if (typeof t === 'string') textParts.push(t);
      }
    }
    states.push({
      pairIndex: pairIdx,
      toolCalls,
      stopReason: String(payload.stop_reason ?? ''),
      textContent: textParts.join('\n'),
    });
    pairIdx += 1;
  }
  return states;
}
