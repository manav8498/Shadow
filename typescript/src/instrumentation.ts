/**
 * Auto-instrumentation for `openai` and `@anthropic-ai/sdk`.
 *
 * In Node's ESM+CJS dual-package world, `require('openai/...')` and
 * `await import('openai/...')` return DIFFERENT class prototypes. User
 * code typically imports via ESM, so we must patch the ESM variant via
 * dynamic import. We ALSO patch the CJS variant so projects mixing
 * module systems (rare but real) are covered.
 *
 * Streaming (`stream: true`) is intercepted via an async-iterator
 * proxy that aggregates chunks into a single chat_response record
 * after the stream completes. Aggregator implementations live below
 * (one per provider). Recording happens once on stream end (or on
 * caller-side break, via the iterator's return path).
 */

import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

import type { Session } from './session.js';

type AnyFn = (...args: unknown[]) => unknown;

export interface InstrumentorHandle {
  patches: Array<{ target: object; attr: string; original: AnyFn }>;
}

/**
 * Resolve `openai` / `@anthropic-ai/sdk` from the USER APP's location,
 * not shadow-diff's. When shadow-diff is installed via symlink (`npm
 * link` or `npm install ../path/to/shadow-diff`), Node's default
 * resolution walks up from the symlink TARGET — i.e. shadow-diff's
 * own dev tree, which has its own peer copies of those SDKs in
 * `node_modules/`. Patching THOSE copies has no effect on the user
 * agent's actual SDK calls, and `shadow record` produces a metadata-
 * only trace.
 *
 * Anchoring on `process.argv[1]` (the script Node was launched with —
 * the user's entrypoint) makes both `require()` and `import()` look
 * up modules from the user's project root regardless of how
 * shadow-diff itself was installed. Falls back to `import.meta.url`
 * when argv[1] is empty (REPL, `node -e`, etc.) so non-script callers
 * still work.
 */
function _resolutionAnchor(): string {
  const script = process.argv[1];
  if (script) {
    try {
      return pathToFileURL(script).href;
    } catch {
      /* fall through to the module-anchored default */
    }
  }
  return import.meta.url;
}

const _ANCHOR_URL = _resolutionAnchor();
const nodeRequire = createRequire(_ANCHOR_URL);

/**
 * Module-level guard against double-wrapping the same prototype.
 *
 * The handle-level `patches.some()` dedup only catches the case of one
 * `autoInstrument()` call seeing the same proto twice. It does NOT
 * protect against the more common pitfall: a caller that runs
 * `Session.enter()` (which already calls `autoInstrument(this)` by
 * default) and then runs `autoInstrument(session)` again. Each call
 * builds a fresh handle, so the second call's `existingCreate` is
 * the FIRST wrapper — the second `wrapped` then chains it as
 * `original`, and every chat call records twice.
 *
 * Tracking wrapped functions globally fixes this regardless of how
 * many handles or sessions exist. We mark a function as wrapped via
 * a non-enumerable symbol property; WeakSet would also work but the
 * symbol is observable in a debugger and makes the intent explicit.
 */
const _SHADOW_WRAPPED = Symbol.for('shadow-diff.wrapped');

interface ShadowWrapped extends AnyFn {
  [_SHADOW_WRAPPED]?: true;
}

export async function autoInstrument(session: Session): Promise<InstrumentorHandle> {
  const handle: InstrumentorHandle = { patches: [] };
  await tryPatchOpenAI(session, handle);
  await tryPatchOpenAIResponses(session, handle);
  await tryPatchAnthropic(session, handle);
  await tryPatchVercelAi(session, handle);
  return handle;
}

export function uninstrument(handle: InstrumentorHandle): void {
  for (const { target, attr, original } of handle.patches) {
    try {
      (target as Record<string, unknown>)[attr] = original;
    } catch {
      /* best-effort */
    }
  }
  handle.patches = [];
}

// ---------------------------------------------------------------------------

async function tryPatchOpenAI(
  session: Session,
  handle: InstrumentorHandle,
): Promise<void> {
  const candidates = await loadModuleBothWays('openai/resources/chat/completions');
  for (const mod of candidates) {
    const Completions = (mod as { Completions?: { prototype?: object } }).Completions;
    const AsyncCompletions = (mod as { AsyncCompletions?: { prototype?: object } })
      .AsyncCompletions;
    patchCreate(session, Completions?.prototype, handle, openaiTranslators);
    patchCreate(session, AsyncCompletions?.prototype, handle, openaiTranslators);
  }
}

async function tryPatchOpenAIResponses(
  session: Session,
  handle: InstrumentorHandle,
): Promise<void> {
  const candidates = await loadModuleBothWays('openai/resources/responses');
  for (const mod of candidates) {
    const Responses = (mod as { Responses?: { prototype?: object } }).Responses;
    patchCreate(session, Responses?.prototype, handle, openaiResponsesTranslators);
  }
}

/**
 * Patch the Vercel AI SDK's top-level functions: `generateText`,
 * `streamText`, `generateObject`, `streamObject`.
 *
 * Vercel AI is a higher-level wrapper that abstracts over OpenAI /
 * Anthropic / Bedrock / xAI and several others — used by BrowserOS
 * and many Next.js / serverless agents. Direct OpenAI patching
 * doesn't catch it because the calls go through `@ai-sdk/openai`'s
 * internal HTTP client, not through the `openai` package.
 *
 * We patch each function as a module export. Since the Vercel AI SDK
 * exports these as `const`, we can re-bind them on the module
 * namespace object (live binding for ESM, plain assignment for CJS).
 *
 * Result shape (from `await generateText({...})`):
 *   { text, usage: { promptTokens, completionTokens, totalTokens },
 *     finishReason, toolCalls, response: { messages, modelId, ... } }
 */
async function tryPatchVercelAi(
  session: Session,
  handle: InstrumentorHandle,
): Promise<void> {
  const candidates = await loadModuleBothWays('ai');
  for (const mod of candidates) {
    if (mod === null || mod === undefined || typeof mod !== 'object') continue;
    for (const fnName of ['generateText', 'streamText', 'generateObject', 'streamObject']) {
      patchModuleFn(session, mod, fnName, handle, vercelAiTranslators);
    }
  }
}

/**
 * Re-bind a top-level module function (ESM live binding or CJS export)
 * to a wrapped variant that records the underlying call. Skips when
 * the module entry isn't a function or has already been wrapped.
 */
function patchModuleFn(
  session: Session,
  mod: object,
  attr: string,
  handle: InstrumentorHandle,
  translators: Translators,
): void {
  const target = mod as Record<string, unknown>;
  const original = target[attr] as ShadowWrapped | undefined;
  if (typeof original !== 'function') return;
  if (original[_SHADOW_WRAPPED]) return;
  if (handle.patches.some((p) => p.target === mod && p.attr === attr)) return;

  handle.patches.push({ target: mod, attr, original: original as AnyFn });

  const wrapped: ShadowWrapped = async function (
    this: unknown,
    ...args: unknown[]
  ): Promise<unknown> {
    const kwargs = (args[0] as Record<string, unknown> | undefined) ?? {};
    const start = Date.now();
    const result = await (original as AnyFn).apply(this, args);
    const latencyMs = Date.now() - start;
    try {
      const req = translators.req(kwargs);
      const resp = translators.resp(result, latencyMs);
      session.recordChat(req, resp);
    } catch {
      /* never break the caller */
    }
    return result;
  };
  Object.defineProperty(wrapped, _SHADOW_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });

  // ESM live bindings are typically read-only via `Object.defineProperty`
  // with `writable: false`. Try plain assignment first; on failure (ESM
  // namespace is sealed), fall back to redefining the property
  // descriptor with `configurable: true`.
  try {
    target[attr] = wrapped;
  } catch {
    try {
      Object.defineProperty(target, attr, {
        value: wrapped,
        writable: true,
        configurable: true,
        enumerable: true,
      });
    } catch {
      /* the binding is genuinely sealed; can't patch this module */
    }
  }
}

async function tryPatchAnthropic(
  session: Session,
  handle: InstrumentorHandle,
): Promise<void> {
  // The anthropic SDK relocated the Messages class in v0.90: it now lives
  // at the top-level resources barrel, not a sub-path.
  const paths = [
    '@anthropic-ai/sdk/resources/messages',
    '@anthropic-ai/sdk/resources/messages/messages',
    '@anthropic-ai/sdk/resources',
  ];
  for (const path of paths) {
    const candidates = await loadModuleBothWays(path);
    for (const mod of candidates) {
      const Messages = (mod as { Messages?: { prototype?: object } }).Messages;
      patchCreate(session, Messages?.prototype, handle, anthropicTranslators);
    }
  }
}

async function loadModuleBothWays(path: string): Promise<unknown[]> {
  const out: unknown[] = [];
  // CJS path — `nodeRequire` is anchored to the user's script (see
  // `_resolutionAnchor()`), so symlinked shadow-diff installs don't
  // accidentally pick up shadow-diff's own dev dependency.
  try {
    out.push(nodeRequire(path));
  } catch {
    /* no CJS variant */
  }

  // ESM path — TWO entries to cover both how user code might import:
  //
  //   1. User-anchored absolute file URL of the CJS-resolved path.
  //      Some packages ship a single `.js` that Node treats as ESM
  //      via `package.json "type": "module"`. This branch covers
  //      those.
  //   2. The same path with `.js` rewritten to `.mjs` (when that
  //      file exists). Modern dual-package layouts (openai v6+,
  //      anthropic-ai/sdk) use a wildcard `exports` map of
  //      `"./resources/*": { "import": "./resources/*.mjs",
  //      "require": "./resources/*.js" }`. `userRequire.resolve`
  //      uses the `require` condition and returns `.js`; the
  //      user's `import OpenAI from 'openai'` instead uses the
  //      `import` condition and gets a DIFFERENT class object from
  //      `.mjs`. Patching only the .js class leaves the user's
  //      actual class unwrapped and `shadow record` produces a
  //      metadata-only trace. Patching BOTH covers both code paths.
  let resolvedAbs: string | null = null;
  try {
    resolvedAbs = nodeRequire.resolve(path);
  } catch {
    /* exports field may not surface this subpath to CJS */
  }

  // Variant A: import the `userRequire.resolve()`-returned path.
  if (resolvedAbs !== null) {
    try {
      const url = pathToFileURL(resolvedAbs).href;
      const esm = await import(/* @vite-ignore */ url);
      out.push(esm);
      const maybeDefault = (esm as { default?: unknown }).default;
      if (maybeDefault && maybeDefault !== esm) out.push(maybeDefault);
    } catch {
      /* not loadable as ESM */
    }
  }

  // Variant B: try the .mjs sibling when the resolver returned .js.
  if (resolvedAbs !== null && resolvedAbs.endsWith('.js')) {
    const mjsPath = `${resolvedAbs.slice(0, -'.js'.length)}.mjs`;
    if (existsSync(mjsPath)) {
      try {
        const url = pathToFileURL(mjsPath).href;
        const esm = await import(/* @vite-ignore */ url);
        out.push(esm);
        const maybeDefault = (esm as { default?: unknown }).default;
        if (maybeDefault && maybeDefault !== esm) out.push(maybeDefault);
      } catch {
        /* .mjs sibling exists but isn't a valid ESM module */
      }
    }
  }

  // Variant C (fallback): bare specifier, anchored to THIS module
  // URL. Used only when CJS resolution failed entirely (ESM-only
  // packages with no `require` condition). Under a symlinked
  // install this can find shadow-diff's nested copy first; that's
  // why it's last — Variants A and B already covered the user's
  // copy via user-anchored resolve.
  if (resolvedAbs === null) {
    try {
      const esm = await import(/* @vite-ignore */ path);
      out.push(esm);
      const maybeDefault = (esm as { default?: unknown }).default;
      if (maybeDefault && maybeDefault !== esm) out.push(maybeDefault);
    } catch {
      /* no ESM variant either */
    }
  }
  return out;
}

// ---------------------------------------------------------------------------

// Exported so streaming-aggregation tests can drive the production
// aggregator + translator pair without mocking node:module.
export interface Translators {
  req: (kwargs: Record<string, unknown>) => Record<string, unknown>;
  resp: (result: unknown, latencyMs: number) => Record<string, unknown>;
  /**
   * Optional per-provider streaming aggregator factory. Returns a
   * fresh aggregator instance per stream — must not be shared across
   * concurrent calls.
   */
  aggregate?: () => StreamAggregator;
}

function patchCreate(
  session: Session,
  proto: object | undefined,
  handle: InstrumentorHandle,
  translators: Translators,
): void {
  if (!proto) return;
  const existingCreate = (proto as Record<string, unknown>).create as ShadowWrapped | undefined;
  if (typeof existingCreate !== 'function') return;
  // Don't double-patch within this handle.
  if (handle.patches.some((p) => p.target === proto && p.attr === 'create')) return;
  // Cross-handle / cross-call dedup: if `existingCreate` is itself a
  // wrapper we installed earlier (perhaps via Session.enter()'s built-in
  // autoInstrument), wrapping it again would chain wrappers and cause
  // each chat call to record once per layer. Skip — the proto is
  // already instrumented and recording.
  if (existingCreate[_SHADOW_WRAPPED]) return;

  const original = existingCreate as AnyFn;
  handle.patches.push({ target: proto, attr: 'create', original });

  const wrapped: ShadowWrapped = async function (
    this: unknown,
    ...args: unknown[]
  ): Promise<unknown> {
    const kwargs = (args[0] as Record<string, unknown> | undefined) ?? {};
    const start = Date.now();
    const result = await original.apply(this, args);
    if (kwargs.stream === true) {
      return wrapStream(result, session, translators, kwargs, start);
    }
    const latencyMs = Date.now() - start;
    try {
      const req = translators.req(kwargs);
      const resp = translators.resp(result, latencyMs);
      session.recordChat(req, resp);
    } catch {
      /* never break the caller */
    }
    return result;
  };
  // Tag the wrapper so future `patchCreate` calls — across other
  // sessions, other handles, or after Session.exit() unwraps — can
  // detect "already-wrapped" and skip.
  Object.defineProperty(wrapped, _SHADOW_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });

  (proto as Record<string, unknown>).create = wrapped;
}


// ---------------------------------------------------------------------------
// Streaming aggregation.
//
// OpenAI and Anthropic emit chunks in different shapes. Each
// translator carries an `aggregate` factory that builds a per-stream
// state machine; we feed every chunk to it and call `finalize()` on
// stream end to produce a synthetic response object that the
// translator's `resp()` can consume — same as a non-streaming call.
//
// We yield each chunk through to the caller unchanged so the user's
// own code observes the raw provider stream. Recording is a side
// effect that happens once when iteration completes (or on
// caller-side break / throw).
// ---------------------------------------------------------------------------

export interface StreamAggregator {
  consume(chunk: unknown): void;
  finalize(): unknown;
}

function wrapStream(
  stream: unknown,
  session: Session,
  translators: Translators,
  kwargs: Record<string, unknown>,
  start: number,
): AsyncIterable<unknown> {
  if (!isAsyncIterable(stream)) return stream as AsyncIterable<unknown>;
  const aggregator = translators.aggregate?.();
  if (!aggregator) return stream as AsyncIterable<unknown>;

  return {
    async *[Symbol.asyncIterator](): AsyncGenerator<unknown> {
      let recorded = false;
      const recordOnce = (): void => {
        if (recorded) return;
        recorded = true;
        const latencyMs = Date.now() - start;
        try {
          const finalResp = aggregator.finalize();
          const req = translators.req(kwargs);
          const resp = translators.resp(finalResp, latencyMs);
          session.recordChat(req, resp);
        } catch {
          /* never break the caller */
        }
      };
      try {
        for await (const chunk of stream as AsyncIterable<unknown>) {
          try {
            aggregator.consume(chunk);
          } catch {
            /* aggregation failure must not stop the stream */
          }
          yield chunk;
        }
      } finally {
        recordOnce();
      }
    },
  };
}

function isAsyncIterable(x: unknown): boolean {
  return x !== null && typeof x === 'object' && Symbol.asyncIterator in (x as object);
}

// ---------------------------------------------------------------------------

export const openaiTranslators: Translators = {
  req: (kwargs) => {
    const messages = Array.isArray(kwargs.messages) ? [...kwargs.messages] : [];
    const params: Record<string, unknown> = {};
    const copy = [
      ['max_tokens', 'max_tokens'],
      ['max_completion_tokens', 'max_tokens'],
      ['temperature', 'temperature'],
      ['top_p', 'top_p'],
      ['stop', 'stop'],
    ] as const;
    for (const [src, dst] of copy) {
      if (src in kwargs && kwargs[src] !== undefined) params[dst] = kwargs[src];
    }
    const out: Record<string, unknown> = {
      model: kwargs.model ?? '',
      messages,
      params,
    };
    if (Array.isArray(kwargs.tools) && kwargs.tools.length > 0) {
      out.tools = (kwargs.tools as Array<Record<string, unknown>>).map((t) => {
        if (t && t.type === 'function' && t.function) {
          const fn = t.function as Record<string, unknown>;
          return {
            name: fn.name ?? '',
            description: fn.description ?? '',
            input_schema: fn.parameters ?? {},
          };
        }
        return { ...t };
      });
    }
    return out;
  },
  resp: (result, latencyMs) => {
    const r = result as {
      model?: string;
      choices?: Array<{
        message?: {
          content?: string | null;
          tool_calls?: Array<{
            id?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };
        finish_reason?: string;
      }>;
      usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        completion_tokens_details?: { reasoning_tokens?: number };
        prompt_tokens_details?: { cached_tokens?: number };
      };
    };
    const choice = r.choices?.[0];
    const message = choice?.message;
    const content: Array<Record<string, unknown>> = [];
    // Newer OpenAI APIs (vision, structured output) return an array of
    // content parts. Preserve each part; plain strings still work.
    const rawContent = (message as { content?: string | Array<Record<string, unknown>> | null } | undefined)?.content;
    if (typeof rawContent === 'string' && rawContent.length > 0) {
      content.push({ type: 'text', text: rawContent });
    } else if (Array.isArray(rawContent)) {
      for (const part of rawContent) {
        if (part && typeof part === 'object') content.push({ ...part });
      }
    }
    const refusal = (message as { refusal?: string } | undefined)?.refusal;
    if (typeof refusal === 'string' && refusal.length > 0) {
      content.push({ type: 'refusal', text: refusal });
    }
    for (const tc of message?.tool_calls ?? []) {
      let parsed: unknown = {};
      const rawArgs = tc.function?.arguments;
      if (typeof rawArgs === 'string') {
        try {
          parsed = JSON.parse(rawArgs);
        } catch {
          parsed = { _raw: rawArgs };
        }
      }
      content.push({
        type: 'tool_use',
        id: tc.id ?? '',
        name: tc.function?.name ?? '',
        input: parsed,
      });
    }
    const finish = choice?.finish_reason ?? 'stop';
    const stopReason =
      ({ stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use' } as Record<
        string,
        string
      >)[finish] ?? finish;
    const reasoning = r.usage?.completion_tokens_details?.reasoning_tokens ?? 0;
    const cachedTokens = r.usage?.prompt_tokens_details?.cached_tokens ?? 0;
    const usageOutOpenAI: Record<string, unknown> = {
      input_tokens: r.usage?.prompt_tokens ?? 0,
      output_tokens: r.usage?.completion_tokens ?? 0,
      thinking_tokens: reasoning,
    };
    if (cachedTokens > 0) usageOutOpenAI.cached_input_tokens = cachedTokens;
    return {
      model: r.model ?? '',
      content,
      stop_reason: stopReason,
      latency_ms: latencyMs,
      usage: usageOutOpenAI,
    };
  },
  aggregate: () => new OpenAIStreamAggregator(),
};

/**
 * Aggregates an OpenAI chat-completions stream into a synthetic
 * non-streaming response object. Each chunk has the shape:
 *
 *   { choices: [{ delta: { content?, tool_calls?[] }, finish_reason? }],
 *     model?, usage? }
 *
 * `usage` only appears in the final chunk when the caller passed
 * `stream_options: { include_usage: true }`. When absent, we leave
 * usage zeroed and downstream cost axes degrade gracefully.
 *
 * Tool-call deltas come in by index — multiple tool_calls in one
 * response interleave their argument-string deltas. We rebuild each
 * tool's id/name/arguments by index then hand the whole thing back
 * in OpenAI's response shape so `openaiTranslators.resp()` can
 * consume it without modification.
 */
class OpenAIStreamAggregator implements StreamAggregator {
  private model = '';
  private contentParts: string[] = [];
  private toolCallsByIndex: Record<
    number,
    { id?: string; type?: string; function?: { name?: string; arguments?: string } }
  > = {};
  private finishReason = 'stop';
  private usage: Record<string, unknown> | undefined;

  consume(chunk: unknown): void {
    const c = chunk as {
      model?: string;
      choices?: Array<{
        delta?: {
          content?: string | null;
          tool_calls?: Array<{
            index?: number;
            id?: string;
            type?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };
        finish_reason?: string | null;
      }>;
      usage?: Record<string, unknown>;
    };
    if (c.model) this.model = c.model;
    if (c.usage) this.usage = c.usage;
    const choice = c.choices?.[0];
    if (!choice) return;
    if (choice.finish_reason) this.finishReason = choice.finish_reason;
    const delta = choice.delta;
    if (!delta) return;
    if (typeof delta.content === 'string' && delta.content.length > 0) {
      this.contentParts.push(delta.content);
    }
    for (const tc of delta.tool_calls ?? []) {
      const idx = tc.index ?? 0;
      const slot = (this.toolCallsByIndex[idx] ??= { function: { name: '', arguments: '' } });
      if (tc.id) slot.id = tc.id;
      if (tc.type) slot.type = tc.type;
      if (tc.function) {
        slot.function ??= { name: '', arguments: '' };
        if (tc.function.name) slot.function.name = (slot.function.name ?? '') + tc.function.name;
        if (tc.function.arguments)
          slot.function.arguments = (slot.function.arguments ?? '') + tc.function.arguments;
      }
    }
  }

  finalize(): unknown {
    const orderedToolCalls = Object.keys(this.toolCallsByIndex)
      .map((k) => Number(k))
      .sort((a, b) => a - b)
      .map((i) => this.toolCallsByIndex[i]);
    return {
      model: this.model,
      choices: [
        {
          message: {
            content: this.contentParts.join(''),
            tool_calls: orderedToolCalls.length > 0 ? orderedToolCalls : undefined,
            refusal: null,
          },
          finish_reason: this.finishReason,
        },
      ],
      usage: this.usage,
    };
  }
}

// ---------------------------------------------------------------------------
// OpenAI Responses API.
//
// Different surface from Chat Completions — `input` instead of `messages`,
// `instructions` instead of system role, `output[]` of typed items instead
// of `choices[].message`, and a different streaming event taxonomy
// (response.completed, response.output_text.delta, etc.). Translators
// below normalise both shapes onto the same envelope Shadow records.
// ---------------------------------------------------------------------------

export const openaiResponsesTranslators: Translators = {
  req: (kwargs) => {
    const input = kwargs.input;
    let messages: Array<Record<string, unknown>> = [];
    if (typeof input === 'string') {
      messages = [{ role: 'user', content: input }];
    } else if (Array.isArray(input)) {
      messages = (input as Array<Record<string, unknown>>).map((m) => ({ ...m }));
    }
    const instructions = kwargs.instructions;
    if (typeof instructions === 'string' && instructions.length > 0) {
      messages = [{ role: 'system', content: instructions }, ...messages];
    }
    const params: Record<string, unknown> = {};
    const copy = [
      ['max_output_tokens', 'max_tokens'],
      ['max_tokens', 'max_tokens'],
      ['temperature', 'temperature'],
      ['top_p', 'top_p'],
      ['stop', 'stop'],
    ] as const;
    for (const [src, dst] of copy) {
      if (src in kwargs && kwargs[src] !== undefined) params[dst] = kwargs[src];
    }
    const out: Record<string, unknown> = {
      model: kwargs.model ?? '',
      messages,
      params,
    };
    if (Array.isArray(kwargs.tools) && kwargs.tools.length > 0) {
      out.tools = (kwargs.tools as Array<Record<string, unknown>>).map((t) => {
        if (t && t.type === 'function') {
          return {
            name: t.name ?? '',
            description: t.description ?? '',
            input_schema: t.parameters ?? {},
          };
        }
        return { ...t };
      });
    }
    return out;
  },
  resp: (result, latencyMs) => {
    const r = result as {
      model?: string;
      output?: Array<{
        type: string;
        role?: string;
        content?: Array<{ type: string; text?: string }>;
        call_id?: string;
        id?: string;
        name?: string;
        arguments?: string;
      }>;
      usage?: {
        input_tokens?: number;
        output_tokens?: number;
        output_tokens_details?: { reasoning_tokens?: number };
        input_tokens_details?: { cached_tokens?: number };
      };
      status?: string;
      incomplete_details?: { reason?: string };
    };
    const content: Array<Record<string, unknown>> = [];
    for (const item of r.output ?? []) {
      if (item.type === 'message') {
        for (const part of item.content ?? []) {
          if (
            part.type === 'output_text' &&
            typeof part.text === 'string' &&
            part.text.length > 0
          ) {
            content.push({ type: 'text', text: part.text });
          } else if (
            part.type === 'refusal' &&
            typeof part.text === 'string' &&
            part.text.length > 0
          ) {
            content.push({ type: 'refusal', text: part.text });
          }
        }
      } else if (item.type === 'function_call') {
        let parsed: unknown = {};
        if (typeof item.arguments === 'string') {
          try {
            parsed = JSON.parse(item.arguments);
          } catch {
            parsed = { _raw: item.arguments };
          }
        }
        content.push({
          type: 'tool_use',
          id: item.call_id ?? item.id ?? '',
          name: item.name ?? '',
          input: parsed,
        });
      }
    }
    let stopReason = 'end_turn';
    if (r.status === 'incomplete') {
      const reason = r.incomplete_details?.reason ?? '';
      stopReason = reason === 'max_output_tokens' ? 'max_tokens' : reason || 'incomplete';
    } else if (content.some((b) => b.type === 'tool_use')) {
      stopReason = 'tool_use';
    }
    const reasoning = r.usage?.output_tokens_details?.reasoning_tokens ?? 0;
    const cachedTokens = r.usage?.input_tokens_details?.cached_tokens ?? 0;
    const usage: Record<string, unknown> = {
      input_tokens: r.usage?.input_tokens ?? 0,
      output_tokens: r.usage?.output_tokens ?? 0,
      thinking_tokens: reasoning,
    };
    if (cachedTokens > 0) usage.cached_input_tokens = cachedTokens;
    return {
      model: r.model ?? '',
      content,
      stop_reason: stopReason,
      latency_ms: latencyMs,
      usage,
    };
  },
  aggregate: () => new OpenAIResponsesStreamAggregator(),
};

/**
 * Aggregates an OpenAI Responses streaming sequence into a final
 * Response-shaped object. Two paths:
 *
 *   1. Best path — `response.completed` event arrives carrying the
 *      full Response object. We pass it straight through to `resp()`.
 *   2. Fallback — caller broke early or the stream ended without
 *      `response.completed`. We rebuild from delta events:
 *        - response.output_text.delta → accumulate text
 *        - response.output_item.added (function_call) → register tool
 *        - response.function_call_arguments.delta → accumulate args
 */
class OpenAIResponsesStreamAggregator implements StreamAggregator {
  private completedResponse: unknown | undefined;
  private model = '';
  private status: string | undefined;
  private textParts: string[] = [];
  private toolCallsByItemId: Record<
    string,
    { call_id?: string; name?: string; arguments?: string }
  > = {};
  private orderedItemIds: string[] = [];
  private usage: Record<string, unknown> | undefined;

  consume(chunk: unknown): void {
    const c = chunk as {
      type?: string;
      response?: unknown;
      delta?: unknown;
      item?: unknown;
      item_id?: string;
    };
    const type = c.type ?? '';
    if (type === 'response.completed' && c.response) {
      this.completedResponse = c.response;
      return;
    }
    if ((type === 'response.created' || type === 'response.in_progress') && c.response) {
      const r = c.response as { model?: string; status?: string };
      if (r.model) this.model = r.model;
      if (r.status) this.status = r.status;
    }
    if (type === 'response.output_text.delta' && typeof c.delta === 'string') {
      this.textParts.push(c.delta);
    }
    if (type === 'response.output_item.added' && c.item) {
      const item = c.item as { id?: string; type?: string; call_id?: string; name?: string };
      if (item.type === 'function_call' && item.id) {
        if (!(item.id in this.toolCallsByItemId)) this.orderedItemIds.push(item.id);
        this.toolCallsByItemId[item.id] = {
          call_id: item.call_id,
          name: item.name,
          arguments: '',
        };
      }
    }
    if (
      type === 'response.function_call_arguments.delta' &&
      c.item_id &&
      typeof c.delta === 'string'
    ) {
      if (!(c.item_id in this.toolCallsByItemId)) {
        this.orderedItemIds.push(c.item_id);
        this.toolCallsByItemId[c.item_id] = { arguments: '' };
      }
      const slot = this.toolCallsByItemId[c.item_id];
      slot.arguments = (slot.arguments ?? '') + c.delta;
    }
  }

  finalize(): unknown {
    if (this.completedResponse) return this.completedResponse;
    const output: Array<Record<string, unknown>> = [];
    if (this.textParts.length > 0) {
      output.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: this.textParts.join('') }],
      });
    }
    for (const id of this.orderedItemIds) {
      const slot = this.toolCallsByItemId[id];
      output.push({
        type: 'function_call',
        call_id: slot.call_id ?? '',
        name: slot.name ?? '',
        arguments: slot.arguments ?? '',
      });
    }
    return {
      model: this.model,
      output,
      usage: this.usage,
      status: this.status,
    };
  }
}

export const anthropicTranslators: Translators = {
  req: (kwargs) => {
    let messages = Array.isArray(kwargs.messages)
      ? [...(kwargs.messages as Array<Record<string, unknown>>)]
      : [];
    const system = kwargs.system;
    if (typeof system === 'string') {
      messages = [{ role: 'system', content: system }, ...messages];
    }
    const params: Record<string, unknown> = {};
    const copy = [
      ['max_tokens', 'max_tokens'],
      ['temperature', 'temperature'],
      ['top_p', 'top_p'],
      ['stop_sequences', 'stop'],
    ] as const;
    for (const [src, dst] of copy) {
      if (src in kwargs && kwargs[src] !== undefined) params[dst] = kwargs[src];
    }
    const out: Record<string, unknown> = {
      model: kwargs.model ?? '',
      messages,
      params,
    };
    if (Array.isArray(kwargs.tools) && kwargs.tools.length > 0) {
      out.tools = (kwargs.tools as Array<Record<string, unknown>>).map((t) => ({ ...t }));
    }
    return out;
  },
  resp: (result, latencyMs) => {
    const r = result as {
      model?: string;
      content?: Array<{
        type?: string;
        text?: string;
        id?: string;
        name?: string;
        input?: unknown;
      }>;
      stop_reason?: string;
      usage?: {
        input_tokens?: number;
        output_tokens?: number;
        cache_read_input_tokens?: number;
      };
    };
    const content: Array<Record<string, unknown>> = [];
    for (const part of r.content ?? []) {
      if (part.type === 'text' && typeof part.text === 'string') {
        content.push({ type: 'text', text: part.text });
      } else if (part.type === 'tool_use') {
        content.push({
          type: 'tool_use',
          id: part.id ?? '',
          name: part.name ?? '',
          input: part.input ?? {},
        });
      } else if (part.type === 'thinking') {
        content.push({ type: 'thinking', text: part.text ?? '' });
      }
    }
    // `cache_read_input_tokens` is prompt-cache READS — bill at cache-read
    // rate via `cached_input_tokens`, NOT at reasoning rate via `thinking_tokens`.
    const cachedInput = r.usage?.cache_read_input_tokens ?? 0;
    const usageOut: Record<string, unknown> = {
      input_tokens: r.usage?.input_tokens ?? 0,
      output_tokens: r.usage?.output_tokens ?? 0,
      thinking_tokens: 0,
    };
    if (cachedInput > 0) usageOut.cached_input_tokens = cachedInput;
    return {
      model: r.model ?? '',
      content,
      stop_reason: r.stop_reason ?? 'end_turn',
      latency_ms: latencyMs,
      usage: usageOut,
    };
  },
  aggregate: () => new AnthropicStreamAggregator(),
};

/**
 * Aggregates an Anthropic Messages stream into a synthetic
 * non-streaming Message object. Anthropic's stream events are:
 *
 *   - message_start          { message: { model, usage } }
 *   - content_block_start    { index, content_block }
 *   - content_block_delta    { index, delta: { text_delta | input_json_delta | thinking_delta } }
 *   - content_block_stop     { index }
 *   - message_delta          { delta: { stop_reason }, usage }
 *   - message_stop
 *
 * We track content blocks by index, accumulate text/JSON/thinking
 * deltas, capture the final stop_reason from message_delta, and
 * finalize into the same shape `anthropicTranslators.resp()` expects.
 */
class AnthropicStreamAggregator implements StreamAggregator {
  private model = '';
  private blocksByIndex: Record<
    number,
    {
      type?: string;
      text?: string;
      id?: string;
      name?: string;
      inputJson?: string;
      input?: unknown;
    }
  > = {};
  private stopReason = 'end_turn';
  private usageInputTokens = 0;
  private usageOutputTokens = 0;
  private usageCacheReadInputTokens = 0;

  consume(chunk: unknown): void {
    const c = chunk as {
      type?: string;
      index?: number;
      content_block?: {
        type?: string;
        text?: string;
        id?: string;
        name?: string;
        input?: unknown;
      };
      delta?: {
        type?: string;
        text?: string;
        partial_json?: string;
        stop_reason?: string;
        thinking?: string;
      };
      message?: {
        model?: string;
        usage?: {
          input_tokens?: number;
          output_tokens?: number;
          cache_read_input_tokens?: number;
        };
      };
      usage?: { output_tokens?: number; input_tokens?: number; cache_read_input_tokens?: number };
    };
    switch (c.type) {
      case 'message_start':
        if (c.message?.model) this.model = c.message.model;
        if (c.message?.usage) {
          this.usageInputTokens = c.message.usage.input_tokens ?? this.usageInputTokens;
          this.usageOutputTokens = c.message.usage.output_tokens ?? this.usageOutputTokens;
          this.usageCacheReadInputTokens =
            c.message.usage.cache_read_input_tokens ?? this.usageCacheReadInputTokens;
        }
        break;
      case 'content_block_start': {
        const idx = c.index ?? 0;
        const cb = c.content_block ?? {};
        this.blocksByIndex[idx] = {
          type: cb.type,
          text: cb.type === 'text' ? '' : undefined,
          id: cb.id,
          name: cb.name,
          inputJson: cb.type === 'tool_use' ? '' : undefined,
        };
        break;
      }
      case 'content_block_delta': {
        const idx = c.index ?? 0;
        const slot = (this.blocksByIndex[idx] ??= {});
        const d = c.delta;
        if (!d) break;
        if (d.type === 'text_delta' && typeof d.text === 'string') {
          slot.text = (slot.text ?? '') + d.text;
        } else if (d.type === 'input_json_delta' && typeof d.partial_json === 'string') {
          slot.inputJson = (slot.inputJson ?? '') + d.partial_json;
        } else if (d.type === 'thinking_delta' && typeof d.thinking === 'string') {
          slot.text = (slot.text ?? '') + d.thinking;
        }
        break;
      }
      case 'content_block_stop': {
        const idx = c.index ?? 0;
        const slot = this.blocksByIndex[idx];
        if (slot && slot.type === 'tool_use' && slot.inputJson !== undefined) {
          try {
            slot.input = JSON.parse(slot.inputJson || '{}');
          } catch {
            slot.input = { _raw: slot.inputJson };
          }
        }
        break;
      }
      case 'message_delta':
        if (c.delta?.stop_reason) this.stopReason = c.delta.stop_reason;
        if (c.usage) {
          this.usageOutputTokens = c.usage.output_tokens ?? this.usageOutputTokens;
        }
        break;
    }
  }

  finalize(): unknown {
    const ordered = Object.keys(this.blocksByIndex)
      .map((k) => Number(k))
      .sort((a, b) => a - b)
      .map((i) => this.blocksByIndex[i]);
    const content = ordered.map((b) => {
      if (b.type === 'text') return { type: 'text', text: b.text ?? '' };
      if (b.type === 'tool_use')
        return { type: 'tool_use', id: b.id ?? '', name: b.name ?? '', input: b.input ?? {} };
      if (b.type === 'thinking') return { type: 'thinking', text: b.text ?? '' };
      return { type: b.type ?? 'unknown' };
    });
    return {
      model: this.model,
      content,
      stop_reason: this.stopReason,
      usage: {
        input_tokens: this.usageInputTokens,
        output_tokens: this.usageOutputTokens,
        cache_read_input_tokens: this.usageCacheReadInputTokens,
      },
    };
  }
}


// ---------------------------------------------------------------------------
// Vercel AI SDK translators (`ai` package: generateText / streamText / etc.)
//
// The Vercel AI SDK abstracts over OpenAI / Anthropic / Bedrock / xAI etc.
// It accepts:
//   { model: <provider model object>, messages: [...], prompt: "...",
//     system: "...", tools: { name: { description, parameters } }, ... }
// and returns:
//   { text: "...", usage: { promptTokens, completionTokens, totalTokens },
//     finishReason: "stop"|"tool-calls"|..., toolCalls: [...],
//     response: { messages, modelId, ... } }
//
// streamText returns an object with `.textStream`, `.toUIMessageStream()`,
// `.finishReason`, `.usage`, etc. — accessed AFTER the consumer has
// awaited those promises. The simplest robust capture is post-hoc on
// the returned object's `.text` (a Promise) + `.usage` (a Promise),
// which we wait for in the wrapper before recording. For non-streaming
// generateText / generateObject we read the resolved fields directly.
// ---------------------------------------------------------------------------

export const vercelAiTranslators: Translators = {
  req: (kwargs) => {
    const messages: Array<Record<string, unknown>> = [];
    if (typeof kwargs.system === 'string' && kwargs.system) {
      messages.push({ role: 'system', content: kwargs.system });
    }
    if (Array.isArray(kwargs.messages)) {
      for (const m of kwargs.messages as Array<Record<string, unknown>>) {
        messages.push({ ...m });
      }
    } else if (typeof kwargs.prompt === 'string' && kwargs.prompt) {
      messages.push({ role: 'user', content: kwargs.prompt });
    }
    const params: Record<string, unknown> = {};
    const copy = [
      ['maxTokens', 'max_tokens'],
      ['maxCompletionTokens', 'max_tokens'],
      ['temperature', 'temperature'],
      ['topP', 'top_p'],
      ['stopSequences', 'stop'],
    ] as const;
    for (const [src, dst] of copy) {
      if (src in kwargs && kwargs[src] !== undefined) params[dst] = kwargs[src];
    }
    // Model is a provider object — extract its id when present, else
    // fall back to a string repr; never serialise the whole provider
    // object (it carries client state and would blow up canonical-JSON).
    let modelId = '';
    const modelArg = kwargs.model;
    if (typeof modelArg === 'string') {
      modelId = modelArg;
    } else if (modelArg && typeof modelArg === 'object') {
      const candidate = (modelArg as Record<string, unknown>).modelId
        ?? (modelArg as Record<string, unknown>).model
        ?? (modelArg as Record<string, unknown>).id;
      if (typeof candidate === 'string') modelId = candidate;
    }
    const out: Record<string, unknown> = {
      model: modelId,
      messages,
      params,
    };
    // Vercel AI exposes tools as a record { name: { description, parameters } }.
    // Translate to the OpenAI-style `[{ name, description, input_schema }]`
    // shape for downstream uniformity.
    const tools = kwargs.tools;
    if (tools && typeof tools === 'object' && !Array.isArray(tools)) {
      const shadowTools: Array<Record<string, unknown>> = [];
      for (const [name, def] of Object.entries(tools as Record<string, Record<string, unknown>>)) {
        shadowTools.push({
          name,
          description: def.description ?? '',
          input_schema: def.parameters ?? def.inputSchema ?? {},
        });
      }
      if (shadowTools.length > 0) out.tools = shadowTools;
    }
    return out;
  },

  resp: (result, latencyMs) => {
    const r = (result ?? {}) as Record<string, unknown>;
    const text = (r.text as string | undefined) ?? '';
    const content: Array<Record<string, unknown>> = [];
    if (text) content.push({ type: 'text', text });
    const toolCalls = r.toolCalls as
      | Array<{ toolCallId?: string; toolName?: string; args?: unknown; input?: unknown }>
      | undefined;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        content.push({
          type: 'tool_use',
          id: tc.toolCallId ?? '',
          name: tc.toolName ?? '',
          input: tc.args ?? tc.input ?? {},
        });
      }
    }
    const usage = (r.usage ?? {}) as Record<string, unknown>;
    const stopReasonMap: Record<string, string> = {
      stop: 'end_turn',
      length: 'max_tokens',
      'tool-calls': 'tool_use',
      'content-filter': 'content_filter',
      error: 'error',
    };
    const finish = (r.finishReason as string | undefined) ?? 'stop';
    const stop_reason = stopReasonMap[finish] ?? finish;
    const response = (r.response as Record<string, unknown> | undefined) ?? {};
    const modelId = (response.modelId as string | undefined)
      ?? (r.model as string | undefined)
      ?? '';
    return {
      model: modelId,
      content,
      stop_reason,
      latency_ms: latencyMs,
      usage: {
        input_tokens: Number(usage.promptTokens ?? usage.inputTokens ?? 0),
        output_tokens: Number(usage.completionTokens ?? usage.outputTokens ?? 0),
        thinking_tokens: Number(usage.reasoningTokens ?? 0),
      },
    };
  },
};
