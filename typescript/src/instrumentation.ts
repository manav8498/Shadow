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

import { createRequire } from 'node:module';

import type { Session } from './session.js';

type AnyFn = (...args: unknown[]) => unknown;

export interface InstrumentorHandle {
  patches: Array<{ target: object; attr: string; original: AnyFn }>;
}

const nodeRequire = createRequire(import.meta.url);

export async function autoInstrument(session: Session): Promise<InstrumentorHandle> {
  const handle: InstrumentorHandle = { patches: [] };
  await tryPatchOpenAI(session, handle);
  await tryPatchAnthropic(session, handle);
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
  try {
    out.push(nodeRequire(path));
  } catch {
    /* no CJS variant */
  }
  try {
    const esm = await import(/* @vite-ignore */ path);
    // Module namespace objects are distinct from the CJS export; include both.
    out.push(esm);
    // `default` export is sometimes where the classes live under ESM.
    const maybeDefault = (esm as { default?: unknown }).default;
    if (maybeDefault && maybeDefault !== esm) out.push(maybeDefault);
  } catch {
    /* no ESM variant */
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
  const existingCreate = (proto as Record<string, unknown>).create;
  if (typeof existingCreate !== 'function') return;
  // Don't double-patch.
  if (handle.patches.some((p) => p.target === proto && p.attr === 'create')) return;

  const original = existingCreate as AnyFn;
  handle.patches.push({ target: proto, attr: 'create', original });

  const wrapped = async function (this: unknown, ...args: unknown[]): Promise<unknown> {
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
