/** Shadow Session — records chat pairs into a `.agentlog` file. */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { canonicalJson, contentId, type AgentlogRecord } from './agentlog.js';
import {
  autoInstrument,
  uninstrument,
  type InstrumentorHandle,
} from './instrumentation.js';
import { Redactor } from './redact.js';
import {
  currentParentSpanId,
  currentTraceId,
  envForChild as envForChildImpl,
  newTraceId,
} from './tracing.js';

export interface SessionOptions {
  outputPath: string;
  tags?: Record<string, string>;
  sessionTag?: string;
  autoInstrument?: boolean;
  traceId?: string;
  parentSpanId?: string;
  /** Default Redactor is installed if omitted. Pass `null` to DISABLE redaction
   *  (not recommended — every record's payload is hashed as-is, so any PII or
   *  API keys land in the content-addressed store and persist). */
  redactor?: Redactor | null;
}

const SPEC_VERSION = '0.1';

// Package version — read from the shipped package.json so SDK-provenance
// records carry the actual installed version, not a stale literal. Reading
// at module-load time (cheap, once per process) is standard practice for
// Node SDKs that emit version headers.
function _readSdkVersion(): string {
  try {
    // Walk up from the compiled file location (dist/session.js) to the
    // package root, which contains package.json for this package.
    const here = dirname(fileURLToPath(import.meta.url));
    for (const candidate of [
      `${here}/../package.json`,
      `${here}/../../package.json`,
    ]) {
      try {
        const raw = readFileSync(candidate, 'utf-8');
        const parsed = JSON.parse(raw) as { version?: string; name?: string };
        if (parsed.name === 'shadow-diff' && typeof parsed.version === 'string') {
          return parsed.version;
        }
      } catch {
        continue;
      }
    }
  } catch {
    // fall through
  }
  return '0.0.0';
}
const SDK_VERSION: string = _readSdkVersion();

export class Session {
  private readonly outputPath: string;
  private readonly tags: Record<string, string>;
  private readonly sessionTag: string | undefined;
  private readonly autoInstrumentFlag: boolean;
  private readonly traceIdValue: string;
  private readonly parentSpanId: string | null;
  private readonly redactor: Redactor | null;

  private records: AgentlogRecord[] = [];
  private rootId: string | null = null;
  private instrumentor: InstrumentorHandle | null = null;

  constructor(opts: SessionOptions) {
    this.outputPath = opts.outputPath;
    this.tags = { ...(opts.tags ?? {}) };
    this.sessionTag = opts.sessionTag;
    this.autoInstrumentFlag = opts.autoInstrument ?? true;
    this.traceIdValue = opts.traceId ?? currentTraceId() ?? newTraceId();
    this.parentSpanId = opts.parentSpanId ?? currentParentSpanId();
    // `undefined` → default Redactor; explicit `null` → no redaction.
    this.redactor =
      opts.redactor === undefined ? new Redactor() : opts.redactor;
  }

  async enter(): Promise<this> {
    const metaPayload: Record<string, unknown> = {
      sdk: { name: 'shadow', version: SDK_VERSION },
      runtime: {
        node: process.version,
        platform: `${process.platform}-${process.arch}`,
      },
    };
    if (Object.keys(this.tags).length > 0) {
      metaPayload.tags = { ...this.tags };
    }
    const redactedMeta = this.redact(metaPayload);
    const id = contentId(redactedMeta);
    this.rootId = id;
    // Initialise the on-disk file synchronously and write the metadata
    // record immediately. Doing this in `enter()` (rather than buffering
    // until `exit()`) means a process that crashes mid-run still leaves
    // a parseable .agentlog containing every record produced before the
    // crash, instead of writing nothing at all. JSONL appends are
    // self-terminating so a partial trace parses cleanly.
    mkdirSync(dirname(this.outputPath), { recursive: true });
    const env = this.envelope('metadata', redactedMeta, id, null);
    this.records.push(env);
    writeFileSync(this.outputPath, this._encodeRecord(env));
    if (this.autoInstrumentFlag) {
      this.instrumentor = await autoInstrument(this);
    }
    return this;
  }

  async exit(): Promise<void> {
    if (this.instrumentor) {
      uninstrument(this.instrumentor);
      this.instrumentor = null;
    }
    // No final flush needed — every record was appended synchronously
    // as it was captured (see `enter()` and `recordChat()`). exit() is
    // now solely about un-monkey-patching the openai/anthropic SDKs.
  }

  /** Append a chat_request + chat_response pair. */
  recordChat(
    request: Record<string, unknown>,
    response: Record<string, unknown>,
  ): { requestId: string; responseId: string } {
    if (this.rootId === null) {
      throw new Error('Session not entered; await session.enter() first.');
    }
    const parent = this.lastId();
    // Redact BEFORE content-id so the hash is over the redacted payload,
    // not the original (otherwise PII leaks into the content-address).
    const redactedReq = this.redact(request);
    const redactedResp = this.redact(response);
    const reqId = contentId(redactedReq);
    const reqEnv = this.envelope('chat_request', redactedReq, reqId, parent);
    this.records.push(reqEnv);
    const respId = contentId(redactedResp);
    const respEnv = this.envelope('chat_response', redactedResp, respId, reqId);
    this.records.push(respEnv);
    // Append both records synchronously so a crash AFTER this call
    // returns still leaves them on disk. appendFileSync is one syscall
    // per record and is durable in the same way `>>` from a shell
    // would be — fine for the < 100 records/sec rate a typical agent
    // produces. If recording were on the hot path of every token we'd
    // batch, but chat_request + chat_response is one pair per LLM call.
    appendFileSync(this.outputPath, this._encodeRecord(reqEnv));
    appendFileSync(this.outputPath, this._encodeRecord(respEnv));
    return { requestId: reqId, responseId: respId };
  }

  /**
   * Encode a single record as canonical JSONL bytes (one line + LF).
   * Same wire format as `writeAgentlog([record])` but allocates one
   * record at a time so per-record appends don't quadratically rebuild
   * the whole file.
   */
  private _encodeRecord(env: AgentlogRecord): Uint8Array {
    const line = canonicalJson(env);
    const out = new Uint8Array(line.length + 1);
    out.set(line, 0);
    out[line.length] = 0x0a; // '\n'
    return out;
  }

  get traceId(): string {
    return this.traceIdValue;
  }

  /** Env object a child process should inherit to join this trace. */
  envForChild(): Record<string, string> {
    const spanId = (this.rootId ?? '').replace('sha256:', '').slice(0, 16);
    return envForChildImpl(this.traceIdValue, spanId || '0'.repeat(16));
  }

  private envelope(
    kind: AgentlogRecord['kind'],
    payload: unknown,
    id: string,
    parent: string | null,
  ): AgentlogRecord {
    const meta: AgentlogRecord['meta'] = {};
    if (this.sessionTag !== undefined) meta.session_tag = this.sessionTag;
    meta.trace_id = this.traceIdValue;
    if (this.parentSpanId !== null) meta.parent_span_id = this.parentSpanId;
    return {
      version: SPEC_VERSION,
      id,
      kind,
      ts: nowIso(),
      parent,
      payload,
      meta,
    };
  }

  private lastId(): string | null {
    return this.records.length > 0
      ? this.records[this.records.length - 1].id
      : null;
  }

  /**
   * Counts of captured records by kind. Used by the auto entrypoint
   * to emit a loud zero-capture warning at session exit.
   */
  recordStats(): { total: number; chatRequests: number; chatResponses: number } {
    let chatRequests = 0;
    let chatResponses = 0;
    for (const r of this.records) {
      if (r.kind === 'chat_request') chatRequests++;
      else if (r.kind === 'chat_response') chatResponses++;
    }
    return { total: this.records.length, chatRequests, chatResponses };
  }

  private redact<T>(value: T): T {
    if (this.redactor === null) return value;
    return this.redactor.redactValue(value);
  }
}

function nowIso(): string {
  const d = new Date();
  const pad = (n: number, w: number = 2): string => n.toString().padStart(w, '0');
  return (
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T` +
    `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}.` +
    `${pad(d.getUTCMilliseconds(), 3)}Z`
  );
}
