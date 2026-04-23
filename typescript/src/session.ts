/** Shadow Session — records chat pairs into a `.agentlog` file. */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { contentId, writeAgentlog, type AgentlogRecord } from './agentlog.js';
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
      sdk: { name: 'shadow', version: '0.1.0' },
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
    this.records.push(this.envelope('metadata', redactedMeta, id, null));
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
    await mkdir(dirname(this.outputPath), { recursive: true });
    await writeFile(this.outputPath, writeAgentlog(this.records));
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
    return { requestId: reqId, responseId: respId };
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
