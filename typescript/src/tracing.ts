/** Distributed trace-context propagation for Shadow Sessions. */

import { randomBytes } from 'node:crypto';

export const TRACE_ID_ENV = 'SHADOW_TRACE_ID';
export const PARENT_SPAN_ENV = 'SHADOW_PARENT_SPAN_ID';
export const W3C_TRACEPARENT_ENV = 'TRACEPARENT';

/** Active trace_id from env, or null if not in a distributed trace. */
export function currentTraceId(): string | null {
  const direct = process.env[TRACE_ID_ENV];
  if (direct) return direct;
  const tp = process.env[W3C_TRACEPARENT_ENV];
  if (tp) {
    const parsed = parseTraceparent(tp);
    if (parsed) return parsed.traceId;
  }
  return null;
}

export function currentParentSpanId(): string | null {
  const direct = process.env[PARENT_SPAN_ENV];
  if (direct) return direct;
  const tp = process.env[W3C_TRACEPARENT_ENV];
  if (tp) {
    const parsed = parseTraceparent(tp);
    if (parsed) return parsed.spanId;
  }
  return null;
}

/** Fresh 128-bit hex trace_id. */
export function newTraceId(): string {
  return randomBytes(16).toString('hex');
}

/** Fresh 64-bit hex span_id. */
export function newSpanId(): string {
  return randomBytes(8).toString('hex');
}

/** Env object a child process should inherit to continue the trace. */
export function envForChild(traceId: string, spanId: string): Record<string, string> {
  const traceparent = `00-${traceId}-${spanId}-01`;
  return {
    [TRACE_ID_ENV]: traceId,
    [PARENT_SPAN_ENV]: spanId,
    [W3C_TRACEPARENT_ENV]: traceparent,
  };
}

function parseTraceparent(header: string): { traceId: string; spanId: string } | null {
  const parts = header.trim().split('-');
  if (parts.length !== 4) return null;
  const [version, traceId, spanId] = parts;
  if (version !== '00') return null;
  if (traceId.length !== 32 || spanId.length !== 16) return null;
  return { traceId, spanId };
}
