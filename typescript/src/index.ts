/**
 * @shadow/sdk — TypeScript SDK for Shadow (behavioral diff for LLM agents).
 *
 * Mirrors the Python SDK surface: a `Session` context that records
 * chat request/response pairs into a `.agentlog` file using the
 * Shadow v0.1 SPEC (content-addressed SHA-256 over canonical JSON,
 * NFC Unicode normalization, JSON Lines envelope).
 *
 * Usage:
 *
 *     import { Session, autoInstrument } from '@shadow/sdk';
 *     import OpenAI from 'openai';
 *
 *     const session = new Session({ outputPath: 'trace.agentlog' });
 *     await session.enter();
 *     autoInstrument(session);  // monkey-patch openai / anthropic
 *     const client = new OpenAI();
 *     await client.chat.completions.create({ ... });
 *     await session.exit();
 */

export { Session, type SessionOptions } from './session.js';
export {
  autoInstrument,
  uninstrument,
  type InstrumentorHandle,
} from './instrumentation.js';
export {
  Redactor,
  luhnValid,
  DEFAULT_PATTERNS,
  type RedactionPattern,
  type RedactorOptions,
} from './redact.js';
export {
  canonicalJson,
  contentId,
  writeAgentlog,
  parseAgentlog,
  type AgentlogRecord,
  type AgentlogMeta,
  type RecordKind,
} from './agentlog.js';
export {
  currentTraceId,
  currentParentSpanId,
  newTraceId,
  newSpanId,
  envForChild,
  TRACE_ID_ENV,
  PARENT_SPAN_ENV,
  W3C_TRACEPARENT_ENV,
} from './tracing.js';
