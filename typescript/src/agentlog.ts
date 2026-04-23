/**
 * Shadow `.agentlog` v0.1 — canonical JSON + SHA-256 content addressing.
 *
 * Implements SPEC §5 (canonical JSON rules) and §6 (content addressing)
 * in TypeScript for wire compatibility with the Rust/Python impls.
 *
 * Rules:
 * - UTF-8, Unicode NFC normalization before hashing
 * - Object keys sorted lexicographically (Unicode codepoint order)
 * - Numbers: integers → fixed; floats → shortest-roundtrip
 * - Strings: escape control chars + backslash + quote per RFC 8259
 * - No whitespace between tokens
 * - Record id: `sha256:<hex>` over the canonical payload bytes
 */

import { createHash } from 'node:crypto';

export type RecordKind =
  | 'metadata'
  | 'chat_request'
  | 'chat_response'
  | 'tool_call'
  | 'tool_result'
  | 'error';

export interface AgentlogRecord {
  version: string;
  id: string;
  kind: RecordKind;
  ts: string;
  parent: string | null;
  payload: unknown;
  meta?: AgentlogMeta;
}

export interface AgentlogMeta {
  session_tag?: string;
  redacted?: boolean;
  trace_id?: string;
  parent_span_id?: string;
  [key: string]: unknown;
}

/** Canonicalise a JSON value to bytes per SPEC §5. */
export function canonicalJson(value: unknown): Uint8Array {
  const text = canonicalize(value);
  return new TextEncoder().encode(text);
}

/** SHA-256 content id over the canonical payload: `sha256:<64-hex>`. */
export function contentId(payload: unknown): string {
  const bytes = canonicalJson(payload);
  const hash = createHash('sha256').update(bytes).digest('hex');
  return `sha256:${hash}`;
}

/** Write a list of records as a `.agentlog` (JSON Lines of canonical JSON). */
export function writeAgentlog(records: AgentlogRecord[]): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const r of records) {
    parts.push(canonicalJson(r));
    parts.push(new Uint8Array([0x0a])); // '\n'
  }
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

/** Parse a `.agentlog` byte buffer into a record list. */
export function parseAgentlog(bytes: Uint8Array): AgentlogRecord[] {
  const text = new TextDecoder().decode(bytes);
  const out: AgentlogRecord[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    out.push(JSON.parse(trimmed) as AgentlogRecord);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Internal canonical-JSON walker.
// ---------------------------------------------------------------------------

function canonicalize(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return canonicalNumber(value);
  if (typeof value === 'string') return canonicalString(value);
  if (Array.isArray(value)) {
    return '[' + value.map(canonicalize).join(',') + ']';
  }
  if (typeof value === 'object') {
    const obj = value as { [k: string]: unknown };
    const keys = Object.keys(obj).sort();
    const parts: string[] = [];
    for (const k of keys) {
      parts.push(canonicalString(k) + ':' + canonicalize(obj[k]));
    }
    return '{' + parts.join(',') + '}';
  }
  throw new Error(`unsupported JSON value: ${typeof value}`);
}

function canonicalNumber(n: number): string {
  if (!Number.isFinite(n)) {
    throw new Error(`non-finite number: ${n}`);
  }
  if (Object.is(n, -0)) return '0';
  if (Number.isInteger(n)) return n.toString(10);
  // JavaScript's toString is already shortest-roundtrip.
  return n.toString();
}

function canonicalString(s: string): string {
  // NFC normalize per SPEC §5.4.
  const norm = s.normalize('NFC');
  let out = '"';
  for (const ch of norm) {
    const code = ch.codePointAt(0)!;
    if (ch === '"') out += '\\"';
    else if (ch === '\\') out += '\\\\';
    else if (code === 0x08) out += '\\b';
    else if (code === 0x09) out += '\\t';
    else if (code === 0x0a) out += '\\n';
    else if (code === 0x0c) out += '\\f';
    else if (code === 0x0d) out += '\\r';
    else if (code < 0x20) {
      out += '\\u' + code.toString(16).padStart(4, '0');
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}
