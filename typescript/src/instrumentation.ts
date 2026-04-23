/**
 * Auto-instrumentation for `openai` and `@anthropic-ai/sdk`.
 *
 * In Node's ESM+CJS dual-package world, `require('openai/...')` and
 * `await import('openai/...')` return DIFFERENT class prototypes. User
 * code typically imports via ESM, so we must patch the ESM variant via
 * dynamic import. We ALSO patch the CJS variant so projects mixing
 * module systems (rare but real) are covered.
 *
 * Streaming (`stream: true`) is passed through un-recorded in v0.1.
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

interface Translators {
  req: (kwargs: Record<string, unknown>) => Record<string, unknown>;
  resp: (result: unknown, latencyMs: number) => Record<string, unknown>;
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
    if (kwargs.stream === true) {
      return original.apply(this, args);
    }
    const start = Date.now();
    const result = await original.apply(this, args);
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

const openaiTranslators: Translators = {
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
};

const anthropicTranslators: Translators = {
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
};
