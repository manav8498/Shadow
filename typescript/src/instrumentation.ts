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
  await tryPatchLangchainOpenai(session, handle);
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
 * Patch the Vercel AI SDK at the LanguageModelV3 prototype layer.
 *
 * v3.1 — earlier (broken) approach:
 *   The pre-v3.1 implementation tried to rebind `generateText` /
 *   `streamText` / `generateObject` / `streamObject` on the `ai`
 *   module namespace. That fails on real `ai` v6: the package ships
 *   ESM with read-only export bindings, so the assignment silently
 *   no-ops. User code that imported `generateText` before our patch
 *   ran (which is every consumer, since auto-instrument runs at Node
 *   startup before user imports) keeps a reference to the original
 *   function, and the wrap never takes effect. Customer evidence:
 *   `wrapped: false` on the module's `generateText` after auto-
 *   instrument; metadata-only traces from BrowserOS + direct
 *   `ai.generateText({ model: openai(...), prompt })` calls.
 *
 * v3.1 — current (working) approach:
 *   `generateText` and `streamText` ALWAYS internally call
 *   `model.doGenerate(...)` / `model.doStream(...)` on the
 *   LanguageModelV3 instance the user passed as `model`. So the
 *   robust interception point is the prototype's `doGenerate` /
 *   `doStream` methods, NOT the module export.
 *
 *   For each known `@ai-sdk/<provider>` package, we sentinel-
 *   instantiate ONE model from each sub-factory (`.chat()`,
 *   `.responses()`, `.completion()`, `.languageModel()`, plus the
 *   provider's bare callable form). The first instance gives us the
 *   prototype object via `Object.getPrototypeOf(model)`; we then
 *   patch `doGenerate` / `doStream` on that prototype. All
 *   subsequent user-created instances of the same class share the
 *   same prototype, so they're auto-wrapped the moment the user
 *   calls `openai('gpt-4o')`.
 *
 *   Sentinel instantiation never makes a network call (model
 *   construction in v6 is purely synthetic — it reads a model id
 *   string and stores config). doGenerate is what would call the
 *   API; we never trigger it.
 */
async function tryPatchVercelAi(
  session: Session,
  handle: InstrumentorHandle,
): Promise<void> {
  // Track patched prototypes so we never double-wrap when multiple
  // provider packages share a common base class.
  const patchedProtos = new WeakSet<object>();

  // Known provider packages that ship a LanguageModelV3 implementation.
  // Adding a new provider is a one-line append here — the prototype
  // patching is per-provider but provider-agnostic in shape.
  const providerPkgs = [
    '@ai-sdk/openai',
    '@ai-sdk/anthropic',
    '@ai-sdk/google',
    '@ai-sdk/google-vertex',
    '@ai-sdk/bedrock',
    '@ai-sdk/groq',
    '@ai-sdk/mistral',
    '@ai-sdk/cohere',
    '@ai-sdk/xai',
    '@ai-sdk/togetherai',
    '@ai-sdk/perplexity',
    '@ai-sdk/fireworks',
    '@ai-sdk/deepinfra',
    '@ai-sdk/cerebras',
    '@ai-sdk/replicate',
    '@ai-sdk/openai-compatible',
  ];

  for (const pkg of providerPkgs) {
    const candidates = await loadModuleBothWays(pkg);
    for (const mod of candidates) {
      if (mod === null || mod === undefined) continue;
      patchProviderPrototypes(mod, session, handle, patchedProtos);
    }
  }
}

/**
 * Walk the provider module's exports, sentinel-instantiate model
 * objects from every recognised factory, and patch the prototypes'
 * `doGenerate` / `doStream` methods.
 *
 * Each provider exports a callable factory (e.g. `openai`) that's
 * also an object with sub-properties `.chat()`, `.responses()`, etc.
 * The bare call and the sub-factories may return different concrete
 * model classes (chat vs completion vs responses), so each gets its
 * own sentinel + prototype patch.
 */
function patchProviderPrototypes(
  mod: unknown,
  session: Session,
  handle: InstrumentorHandle,
  patchedProtos: WeakSet<object>,
): void {
  // The provider object is found at one of: a default export, a named
  // export matching the provider name, or the module itself (when the
  // module's namespace IS the provider).
  const seen = new Set<unknown>();
  const candidates: unknown[] = [];
  const namespaceObj = mod as Record<string, unknown>;
  for (const v of Object.values(namespaceObj)) {
    if (v && (typeof v === 'function' || typeof v === 'object')) candidates.push(v);
  }
  // The module itself can BE callable (CJS shim).
  if (typeof mod === 'function') candidates.push(mod);

  for (const provider of candidates) {
    if (!provider || seen.has(provider)) continue;
    seen.add(provider);

    // The bare callable form: `openai('gpt-4o')` returns a chat model.
    if (typeof provider === 'function') {
      sentinelPatch(provider as AnyFn, undefined, session, handle, patchedProtos);
    }

    // Sub-factories that v6 providers consistently expose.
    const subFactories = [
      'chat',
      'responses',
      'completion',
      'languageModel',
      // Image / embedding / speech / transcription factories also exist
      // but they don't implement doGenerate/doStream, so skipping.
    ];
    if (typeof provider === 'object' || typeof provider === 'function') {
      for (const sub of subFactories) {
        const fn = (provider as Record<string, unknown>)[sub];
        if (typeof fn === 'function') {
          sentinelPatch(fn as AnyFn, provider, session, handle, patchedProtos);
        }
      }
    }
  }
}

/**
 * Call ``factory(sentinelId)`` to materialise one model instance,
 * extract its prototype, and patch ``doGenerate`` / ``doStream`` on
 * that prototype. Subsequent user-created instances of the same
 * class share the prototype, so they're caught automatically.
 *
 * Sentinel construction never makes a network call — model
 * constructors only stash config; the API call happens inside
 * ``doGenerate``. Failures are silently skipped so an exotic
 * factory shape doesn't break the whole instrumentor.
 */
function sentinelPatch(
  factory: AnyFn,
  thisArg: unknown,
  session: Session,
  handle: InstrumentorHandle,
  patchedProtos: WeakSet<object>,
): void {
  const sentinelArgs = [['__shadow_sentinel__']];
  // Some factories accept (modelId, settings); some accept (modelId).
  // Try the simplest form first; if it throws we silently skip.
  for (const args of sentinelArgs) {
    let model: unknown;
    try {
      model = factory.apply(thisArg, args);
    } catch {
      continue;
    }
    if (!model || typeof model !== 'object') continue;
    const proto = Object.getPrototypeOf(model);
    if (!proto || patchedProtos.has(proto)) return;
    patchedProtos.add(proto);
    patchProtoMethod(proto, 'doGenerate', session, handle);
    patchProtoMethod(proto, 'doStream', session, handle);
    return;
  }
}

/**
 * Wrap ``proto[methodName]`` so that every call records a chat pair
 * via the active session. The original method is preserved on the
 * patches list for clean uninstall.
 *
 * The wrapper translates the LanguageModelV3 input (``options``)
 * + result to Shadow ``chat_request`` / ``chat_response`` records.
 * Streaming results return an object with ``stream`` + various
 * promises (``finishReason``, ``usage``, ``text``); we await those
 * AFTER the user has consumed them by introducing a tap over the
 * stream — see ``wrapVercelStreamResult``.
 */
function patchProtoMethod(
  proto: object,
  methodName: 'doGenerate' | 'doStream',
  session: Session,
  handle: InstrumentorHandle,
): void {
  const target = proto as Record<string, unknown>;
  const original = target[methodName] as ShadowWrapped | undefined;
  if (typeof original !== 'function') return;
  if (original[_SHADOW_WRAPPED]) return;

  handle.patches.push({ target: proto, attr: methodName, original: original as AnyFn });

  const wrapped: ShadowWrapped = async function (
    this: any,
    options: any,
  ): Promise<unknown> {
    const start = Date.now();
    const result = await (original as AnyFn).call(this, options);
    const latencyMs = Date.now() - start;
    try {
      if (methodName === 'doGenerate') {
        const req = vercelDoGenerateOptionsToReq(options, this);
        const resp = vercelDoGenerateResultToResp(result, latencyMs, this);
        session.recordChat(req, resp);
      } else {
        // doStream: result has { stream, ... }. Wrap the stream so we
        // can record after the consumer has drained it. The
        // recording happens once on stream end via a tap.
        return wrapVercelStreamResult(result, options, this, session, latencyMs, start);
      }
    } catch {
      /* recording must never break the caller */
    }
    return result;
  };
  Object.defineProperty(wrapped, _SHADOW_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });

  target[methodName] = wrapped;
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

/**
 * Patch `@langchain/openai`'s `ChatOpenAI` family at the prototype layer.
 *
 * Why a dedicated patcher (instead of relying on `tryPatchOpenAI`):
 *   `@langchain/openai` bundles its own copy of `openai` in its nested
 *   `node_modules/openai`. That copy is a different file path and a
 *   different module instance from the user's top-level `openai`, so
 *   patching `Completions.prototype.create` on the user-anchored `openai`
 *   resolution leaves LangChain's bundled copy untouched. Customer
 *   evidence: `@langchain/openai` calls produced metadata-only traces
 *   while direct `openai` SDK calls in the same project recorded
 *   chat_request + chat_response correctly.
 *
 * The fix mirrors the Vercel AI approach (`doGenerate` / `doStream` on
 * the LanguageModelV3 prototype): patch `_generate` and
 * `_streamResponseChunks` on the documented public extension points of
 * `ChatOpenAI`, `ChatOpenAICompletions`, and `ChatOpenAIResponses`. The
 * Azure variants inherit from these without overriding the methods, so
 * they pick up the patches automatically.
 */
async function tryPatchLangchainOpenai(
  session: Session,
  handle: InstrumentorHandle,
): Promise<void> {
  const candidates = await loadModuleBothWays('@langchain/openai');
  // The three classes that own concrete `_generate` / `_streamResponseChunks`
  // implementations. Azure variants inherit from these without overriding,
  // so the parent-class patch covers them transparently.
  const classNames = ['ChatOpenAI', 'ChatOpenAICompletions', 'ChatOpenAIResponses'];
  const seenProtos = new WeakSet<object>();
  for (const mod of candidates) {
    if (!mod) continue;
    for (const name of classNames) {
      const cls = (mod as Record<string, unknown>)[name] as
        | { prototype?: object }
        | undefined;
      const proto = cls?.prototype;
      if (!proto || seenProtos.has(proto)) continue;
      seenProtos.add(proto);
      patchLangchainGenerate(session, proto, handle);
      patchLangchainStream(session, proto, handle);
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


// ---------------------------------------------------------------------------
// Vercel AI LanguageModelV3 protocol translators (the prototype-level path).
//
// `doGenerate(options)` → result shape — different from the user-facing
// `generateText` shape that `vercelAiTranslators` above handles.
//
// `options` (LanguageModelV3CallOptions):
//   { prompt: [{ role, content }],          // role-tagged messages
//     maxOutputTokens, temperature, topP, topK,
//     frequencyPenalty, presencePenalty, stopSequences, responseFormat,
//     seed, tools: [{ type: 'function', name, parameters, ... }],
//     toolChoice, providerOptions, abortSignal, headers }
//
// `result` (LanguageModelV3CallResult):
//   { content: [{ type: 'text', text }
//             | { type: 'tool-call', toolCallId, toolName, input }
//             | { type: 'reasoning', text }],
//     finishReason: { unified, raw },        // unified ∈ stop|tool-calls|...
//     usage: { inputTokens: { total, ... }, outputTokens: { total, text, reasoning } },
//     request: { body },
//     response: { id, modelId, timestamp, headers, body },
//     warnings, providerMetadata }
//
// The `this` instance carries `modelId` and `provider` (a string label).
// ---------------------------------------------------------------------------

export function vercelDoGenerateOptionsToReq(
  options: any,
  model: any,
): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  const prompt = options?.prompt;
  if (Array.isArray(prompt)) {
    // Each entry is { role: 'system'|'user'|'assistant'|'tool', content: string|Array }
    for (const m of prompt) {
      if (!m || typeof m !== 'object') continue;
      const role = (m as Record<string, unknown>).role;
      const content = (m as Record<string, unknown>).content;
      if (typeof role === 'string') {
        messages.push({ role, content: content ?? '' });
      }
    }
  }
  const params: Record<string, unknown> = {};
  const map = [
    ['maxOutputTokens', 'max_tokens'],
    ['temperature', 'temperature'],
    ['topP', 'top_p'],
    ['topK', 'top_k'],
    ['frequencyPenalty', 'frequency_penalty'],
    ['presencePenalty', 'presence_penalty'],
    ['stopSequences', 'stop'],
    ['seed', 'seed'],
  ] as const;
  for (const [src, dst] of map) {
    const v = options?.[src];
    if (v !== undefined && v !== null) params[dst] = v;
  }

  const out: Record<string, unknown> = {
    model: typeof model?.modelId === 'string' ? model.modelId : '',
    messages,
    params,
  };

  // LanguageModelV3 tools are an array of {type: 'function', name, parameters, ...}.
  const tools = options?.tools;
  if (Array.isArray(tools) && tools.length > 0) {
    const shadowTools: Array<Record<string, unknown>> = [];
    for (const t of tools) {
      if (!t || typeof t !== 'object') continue;
      const tt = t as Record<string, unknown>;
      shadowTools.push({
        name: typeof tt.name === 'string' ? tt.name : '',
        description: typeof tt.description === 'string' ? tt.description : '',
        input_schema: tt.parameters ?? tt.inputSchema ?? {},
      });
    }
    if (shadowTools.length > 0) out.tools = shadowTools;
  }
  return out;
}

export function vercelDoGenerateResultToResp(
  result: any,
  latencyMs: number,
  model: any,
): Record<string, unknown> {
  const content: Array<Record<string, unknown>> = [];
  const items = result?.content;
  if (Array.isArray(items)) {
    for (const item of items) {
      if (!item || typeof item !== 'object') continue;
      const it = item as Record<string, unknown>;
      const ttype = it.type;
      if (ttype === 'text') {
        content.push({ type: 'text', text: typeof it.text === 'string' ? it.text : '' });
      } else if (ttype === 'tool-call') {
        // Vercel emits `input` as a JSON string in v6; downstream
        // policy rules expect a parsed dict.
        let parsedInput: unknown = it.input ?? it.args ?? {};
        if (typeof parsedInput === 'string') {
          try {
            parsedInput = JSON.parse(parsedInput);
          } catch {
            parsedInput = { _raw: parsedInput };
          }
        }
        content.push({
          type: 'tool_use',
          id: typeof it.toolCallId === 'string' ? it.toolCallId : '',
          name: typeof it.toolName === 'string' ? it.toolName : '',
          input: parsedInput,
        });
      } else if (ttype === 'reasoning') {
        content.push({
          type: 'thinking',
          text: typeof it.text === 'string' ? it.text : '',
        });
      }
      // Other types (source, file, etc.) drop silently — they aren't
      // first-class chat-response parts in the Shadow envelope.
    }
  }

  // Finish reason: v6 wraps it as { unified, raw }.
  const finish = result?.finishReason;
  let unified: string;
  if (typeof finish === 'string') {
    unified = finish;
  } else if (finish && typeof finish === 'object') {
    unified = (finish as Record<string, unknown>).unified as string ?? 'stop';
  } else {
    unified = 'stop';
  }
  const stopReasonMap: Record<string, string> = {
    stop: 'end_turn',
    length: 'max_tokens',
    'tool-calls': 'tool_use',
    'content-filter': 'content_filter',
    error: 'error',
  };
  const stop_reason = stopReasonMap[unified] ?? unified;

  // Usage: v6 nests as { inputTokens: { total }, outputTokens: { total, reasoning } }.
  const usage = result?.usage ?? {};
  const inputTokens = Number(
    (usage.inputTokens?.total ?? usage.inputTokens ?? usage.promptTokens ?? 0) || 0,
  );
  const outputTokens = Number(
    (usage.outputTokens?.total ?? usage.outputTokens ?? usage.completionTokens ?? 0) || 0,
  );
  const reasoningTokens = Number(
    (usage.outputTokens?.reasoning ?? usage.reasoningTokens ?? 0) || 0,
  );

  // Response model id may be on the result.response or on the model.
  const responseModelId =
    (result?.response?.modelId as string | undefined) ??
    (typeof model?.modelId === 'string' ? model.modelId : '');

  return {
    model: responseModelId,
    content,
    stop_reason,
    latency_ms: latencyMs,
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      thinking_tokens: reasoningTokens,
    },
  };
}

/**
 * Wrap the result of `doStream(options)` so we record after the
 * caller has finished consuming the stream.
 *
 * v6 doStream returns { stream: ReadableStream<...>, request, response, ... }.
 * The stream emits chunks of various types and a final 'finish' chunk
 * carrying finishReason + usage. We tap the stream by replacing it
 * with a teeed copy that accumulates chunks and emits a single
 * chat_response record once the stream ends.
 */
function wrapVercelStreamResult(
  result: any,
  options: any,
  model: any,
  session: Session,
  latencyMs: number,
  start: number,
): unknown {
  const original = result?.stream;
  // If the result doesn't look like a doStream result, return unchanged.
  if (!original || typeof original.tee !== 'function') return result;
  const [forwarded, captured] = (original as ReadableStream).tee();

  // Accumulate captured chunks asynchronously without blocking the
  // caller. Emit the chat_response once the stream ends.
  void (async () => {
    try {
      const reader = (captured as ReadableStream).getReader();
      const buffered: any[] = [];
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffered.push(value);
      }
      const finishChunk = buffered.find(
        (c) => c && typeof c === 'object' && c.type === 'finish',
      );
      const textParts: string[] = [];
      const toolUses: Array<Record<string, unknown>> = [];
      let thinking = '';
      for (const c of buffered) {
        if (!c || typeof c !== 'object') continue;
        if (c.type === 'text-delta' && typeof c.delta === 'string') {
          textParts.push(c.delta);
        } else if (c.type === 'reasoning-delta' && typeof c.delta === 'string') {
          thinking += c.delta;
        } else if (c.type === 'tool-call') {
          let parsedInput: unknown = c.input ?? {};
          if (typeof parsedInput === 'string') {
            try {
              parsedInput = JSON.parse(parsedInput);
            } catch {
              parsedInput = { _raw: parsedInput };
            }
          }
          toolUses.push({
            type: 'tool_use',
            id: c.toolCallId ?? '',
            name: c.toolName ?? '',
            input: parsedInput,
          });
        }
      }
      const synthetic: any = {
        content: [
          ...(textParts.length > 0 ? [{ type: 'text', text: textParts.join('') }] : []),
          ...(thinking ? [{ type: 'reasoning', text: thinking }] : []),
          ...toolUses.map((tu) => ({
            type: 'tool-call',
            toolCallId: tu.id,
            toolName: tu.name,
            input: tu.input,
          })),
        ],
        finishReason: finishChunk?.finishReason ?? 'stop',
        usage: finishChunk?.usage ?? {},
        response: { modelId: model?.modelId ?? '' },
      };
      const totalLatency = Date.now() - start;
      const req = vercelDoGenerateOptionsToReq(options, model);
      const resp = vercelDoGenerateResultToResp(synthetic, totalLatency, model);
      session.recordChat(req, resp);
    } catch {
      /* recording failures must never propagate */
    }
  })();

  return { ...result, stream: forwarded };
}

// ===========================================================================
// LangChain (`@langchain/openai`) — translators + per-method patchers.
// ===========================================================================

interface LangchainBaseMessage {
  _getType?: () => string;
  content?: unknown;
  tool_calls?: Array<{ id?: string; name?: string; args?: unknown }>;
  tool_call_id?: string;
}

/** Convert a LangChain `BaseMessage[]` into Shadow's chat_request shape. */
function langchainMessagesToShadowReq(
  messages: unknown,
  modelId: string,
): Record<string, unknown> {
  const outMessages: Array<Record<string, unknown>> = [];
  if (Array.isArray(messages)) {
    for (const m of messages) {
      if (!m || typeof m !== 'object') continue;
      const msg = m as LangchainBaseMessage;
      const lcType = typeof msg._getType === 'function' ? msg._getType() : '';
      let role: string;
      switch (lcType) {
        case 'human':
          role = 'user';
          break;
        case 'system':
          role = 'system';
          break;
        case 'ai':
          role = 'assistant';
          break;
        case 'tool':
        case 'function':
          role = 'tool';
          break;
        default:
          role = lcType || 'user';
      }
      const entry: Record<string, unknown> = { role, content: msg.content ?? '' };
      if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
        entry.tool_calls = msg.tool_calls;
      }
      if (typeof msg.tool_call_id === 'string') {
        entry.tool_call_id = msg.tool_call_id;
      }
      outMessages.push(entry);
    }
  }
  return {
    model: modelId,
    messages: outMessages,
    params: {},
  };
}

interface LangchainChatResult {
  generations?: Array<{
    text?: string;
    message?: {
      content?: unknown;
      tool_calls?: Array<{ id?: string; name?: string; args?: unknown }>;
    };
    generationInfo?: { finish_reason?: string };
  }>;
  llmOutput?: {
    tokenUsage?: {
      promptTokens?: number;
      completionTokens?: number;
      totalTokens?: number;
    };
  };
}

/** Convert a LangChain `ChatResult` into Shadow's chat_response shape. */
function langchainResultToShadowResp(
  result: unknown,
  latencyMs: number,
  modelId: string,
): Record<string, unknown> {
  const r = (result as LangchainChatResult) ?? {};
  const gen = r.generations?.[0];
  const message = gen?.message;
  const content: Array<Record<string, unknown>> = [];
  const rawContent = message?.content;
  if (typeof rawContent === 'string' && rawContent.length > 0) {
    content.push({ type: 'text', text: rawContent });
  } else if (Array.isArray(rawContent)) {
    for (const part of rawContent) {
      if (part && typeof part === 'object') content.push({ ...(part as object) });
    }
  }
  for (const tc of message?.tool_calls ?? []) {
    content.push({
      type: 'tool_use',
      id: tc.id ?? '',
      name: tc.name ?? '',
      input: tc.args ?? {},
    });
  }
  const finish = gen?.generationInfo?.finish_reason ?? 'stop';
  const stopReason =
    ({ stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use' } as Record<
      string,
      string
    >)[finish] ?? finish;
  const usage = r.llmOutput?.tokenUsage;
  return {
    model: modelId,
    content,
    stop_reason: stopReason,
    latency_ms: latencyMs,
    usage: {
      input_tokens: usage?.promptTokens ?? 0,
      output_tokens: usage?.completionTokens ?? 0,
      thinking_tokens: 0,
    },
  };
}

function patchLangchainGenerate(
  session: Session,
  proto: object,
  handle: InstrumentorHandle,
): void {
  const existing = (proto as Record<string, unknown>)._generate as ShadowWrapped | undefined;
  if (typeof existing !== 'function') return;
  if (handle.patches.some((p) => p.target === proto && p.attr === '_generate')) return;
  if (existing[_SHADOW_WRAPPED]) return;

  const original = existing as AnyFn;
  handle.patches.push({ target: proto, attr: '_generate', original });

  const wrapped: ShadowWrapped = async function (
    this: unknown,
    ...args: unknown[]
  ): Promise<unknown> {
    const start = Date.now();
    const result = await original.apply(this, args);
    const latencyMs = Date.now() - start;
    try {
      const messages = args[0];
      const modelId = ((this as { model?: string } | null)?.model ?? '') as string;
      const req = langchainMessagesToShadowReq(messages, modelId);
      const resp = langchainResultToShadowResp(result, latencyMs, modelId);
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
  (proto as Record<string, unknown>)._generate = wrapped;
}

interface LangchainStreamChunk {
  text?: string;
  message?: {
    content?: unknown;
    tool_call_chunks?: Array<{
      index?: number;
      id?: string;
      name?: string;
      args?: string;
    }>;
  };
  generationInfo?: { finish_reason?: string };
  usage_metadata?: {
    input_tokens?: number;
    output_tokens?: number;
  };
}

function patchLangchainStream(
  session: Session,
  proto: object,
  handle: InstrumentorHandle,
): void {
  const existing = (proto as Record<string, unknown>)._streamResponseChunks as
    | ShadowWrapped
    | undefined;
  if (typeof existing !== 'function') return;
  if (handle.patches.some((p) => p.target === proto && p.attr === '_streamResponseChunks'))
    return;
  if (existing[_SHADOW_WRAPPED]) return;

  const original = existing as AnyFn;
  handle.patches.push({ target: proto, attr: '_streamResponseChunks', original });

  const wrapped: ShadowWrapped = async function* (
    this: unknown,
    ...args: unknown[]
  ): AsyncGenerator<unknown> {
    const start = Date.now();
    const messages = args[0];
    const modelId = ((this as { model?: string } | null)?.model ?? '') as string;
    let aggregatedText = '';
    const toolCalls = new Map<number, { id?: string; name?: string; argText: string }>();
    let finishReason: string | undefined;
    let usage: { input_tokens?: number; output_tokens?: number } | undefined;
    let recorded = false;
    const flush = () => {
      if (recorded) return;
      recorded = true;
      const latencyMs = Date.now() - start;
      try {
        const req = langchainMessagesToShadowReq(messages, modelId);
        const content: Array<Record<string, unknown>> = [];
        if (aggregatedText.length > 0) content.push({ type: 'text', text: aggregatedText });
        for (const [, tc] of toolCalls) {
          let parsed: unknown = {};
          if (tc.argText.length > 0) {
            try {
              parsed = JSON.parse(tc.argText);
            } catch {
              parsed = { _raw: tc.argText };
            }
          }
          content.push({
            type: 'tool_use',
            id: tc.id ?? '',
            name: tc.name ?? '',
            input: parsed,
          });
        }
        const stopReason =
          ({ stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use' } as Record<
            string,
            string
          >)[finishReason ?? 'stop'] ?? (finishReason ?? 'end_turn');
        session.recordChat(req, {
          model: modelId,
          content,
          stop_reason: stopReason,
          latency_ms: latencyMs,
          usage: {
            input_tokens: usage?.input_tokens ?? 0,
            output_tokens: usage?.output_tokens ?? 0,
            thinking_tokens: 0,
          },
        });
      } catch {
        /* swallow — never break the caller */
      }
    };
    try {
      const iter = original.apply(this, args) as AsyncGenerator<unknown>;
      for await (const chunk of iter) {
        const c = (chunk ?? {}) as LangchainStreamChunk;
        if (typeof c.text === 'string') aggregatedText += c.text;
        const fr = c.generationInfo?.finish_reason;
        if (fr) finishReason = fr;
        for (const tc of c.message?.tool_call_chunks ?? []) {
          const idx = tc.index ?? 0;
          const entry = toolCalls.get(idx) ?? { argText: '' };
          if (tc.id) entry.id = tc.id;
          if (tc.name) entry.name = tc.name;
          if (tc.args) entry.argText += tc.args;
          toolCalls.set(idx, entry);
        }
        if (c.usage_metadata) {
          usage = {
            input_tokens: c.usage_metadata.input_tokens,
            output_tokens: c.usage_metadata.output_tokens,
          };
        }
        yield chunk;
      }
    } finally {
      flush();
    }
  };
  Object.defineProperty(wrapped, _SHADOW_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  (proto as Record<string, unknown>)._streamResponseChunks = wrapped;
}
