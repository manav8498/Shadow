/**
 * Zero-config session bootstrap for `shadow record -- node ...`.
 *
 * Node analogue of `python/src/shadow/sdk/_autostart.py`. When this
 * module is imported (typically via `node --import shadow-diff/auto`
 * or `NODE_OPTIONS='--import shadow-diff/auto'`), it:
 *
 *   1. Reads `SHADOW_SESSION_OUTPUT` from the environment. When the
 *      env var is unset, this module is a no-op so it's safe to
 *      activate unconditionally.
 *   2. Constructs a `Session({ outputPath, tags })`, enters it, and
 *      runs `autoInstrument(session)` — same auto-instrumentation
 *      `Session` exposes for in-process callers.
 *   3. Registers `beforeExit`, `SIGINT`, `SIGTERM`, and
 *      `uncaughtException` handlers that flush the session to disk
 *      cleanly on process shutdown.
 *
 * The Python `shadow record` CLI sets the env vars + `NODE_OPTIONS`
 * before launching the wrapped command, so JS agents pick up
 * recording without any code change. Mirrors the
 * `sitecustomize.py` flow that does the same job for Python agents.
 *
 * ## Guard rails
 *
 *   * **No-op when `SHADOW_SESSION_OUTPUT` is unset** — safe to
 *     activate via `NODE_OPTIONS='--import shadow-diff/auto'` in a
 *     shell init script.
 *   * **Idempotent** — a module-level flag prevents double-starting
 *     even when the user's package.json or build pipeline ends up
 *     loading this twice.
 *   * **Never crashes the agent** — every error path is caught and
 *     logged to stderr with a `hint:` line; the wrapped command
 *     keeps running, just without recording.
 *   * **Best-effort flush** — exit handlers await the session, but
 *     because `process.exit(N)` is synchronous in Node, an
 *     uncooperative agent that calls `process.exit()` directly
 *     before our handlers run will lose the trailing records. The
 *     content-addressed envelope means partial traces still parse;
 *     they just stop early.
 */

import { Session, type SessionOptions } from './session.js';

let _bootstrapped = false;

function _parseTagsEnv(raw: string | undefined): Record<string, string> {
  // SHADOW_SESSION_TAGS=key1=val1,key2=val2 — same shape the Python
  // CLI emits. Tolerant of empty pairs, leading commas, missing `=`
  // (malformed pairs are dropped silently rather than failing the
  // whole startup).
  const out: Record<string, string> = {};
  if (!raw) return out;
  for (const pair of raw.split(',')) {
    const trimmed = pair.trim();
    if (!trimmed) continue;
    const eq = trimmed.indexOf('=');
    if (eq <= 0) continue;
    const k = trimmed.slice(0, eq).trim();
    const v = trimmed.slice(eq + 1).trim();
    if (k) out[k] = v;
  }
  return out;
}

function _emitEmptyCaptureWarning(outPath: string): void {
  // Stderr-only loud warning. Names the four canonical causes so users
  // can self-diagnose without reading source. Mirrors the Python
  // _emit_empty_capture_warning in shadow.sdk._autostart.
  const lines = [
    'shadow: WARNING — zero LLM calls were captured.',
    `  Session output: ${outPath}`,
    '  The agent ran to completion but no `chat_request` records',
    '  were intercepted. Most common causes:',
    '    * Agent uses an SDK Shadow does not yet auto-instrument.',
    '      Currently patched: openai, @anthropic-ai/sdk, ai (Vercel AI SDK',
    '      v6 via @ai-sdk/<provider> — openai/anthropic/google/bedrock',
    '      and a dozen others).',
    '      File a request: https://github.com/manav8498/Shadow/issues',
    '    * shadow-diff isn\'t installed in the agent\'s project.',
    '      Fix:    npm install shadow-diff   (or `bun add shadow-diff`)',
    '    * Agent stored a bound-method reference to the SDK call before',
    '      Session entered. Open an explicit `Session` around the call',
    '      site instead of relying on shadow record\'s zero-config.',
    '    * Agent didn\'t actually make any LLM calls (rare).',
  ];
  // eslint-disable-next-line no-console
  console.error(lines.join('\n'));
}

function _warnNoCrash(message: string, error?: unknown): void {
  // Stderr only — never throw. The agent must keep running.
  // eslint-disable-next-line no-console
  console.warn(`[shadow-diff/auto] ${message}`);
  if (error !== undefined) {
    // eslint-disable-next-line no-console
    console.warn('  hint: shadow recording is disabled for this run; the agent continues.');
    // eslint-disable-next-line no-console
    console.warn(error);
  }
}

// Match-only-the-CLI-wrapper patterns. When `shadow record -- npm
// start` runs, NODE_OPTIONS is inherited by BOTH the npm process and
// the user-agent grandchild. If both processes ran the auto bootstrap
// they'd both write to SHADOW_SESSION_OUTPUT and (because Session
// uses writeFile, not appendFile) the last one to exit would
// overwrite the other — most often producing a metadata-only trace
// in the npm wrapper that obliterates the user's real records.
//
// Skip the bootstrap when we recognise we're running the package-
// manager CLI itself, not the user's actual agent. The grandchild
// (user agent) doesn't match these patterns, so it records normally.
const _PACKAGE_MANAGER_CLI_PATTERNS = [
  /[/\\](?:npm|npx)-cli\.(?:js|cjs|mjs)$/,
  /[/\\]npm[/\\]bin[/\\](?:npm|npx)-cli\.(?:js|cjs|mjs)$/,
  /[/\\](?:pnpm|pnpx)\.(?:cjs|js|mjs)$/,
  /[/\\]pnpm[/\\]bin[/\\]pnpm\.(?:cjs|js|mjs)$/,
  /[/\\]yarn(?:\.js)?$/,
  /[/\\]yarn[/\\]bin[/\\]yarn\.js$/,
];

function _is_package_manager_wrapper(): boolean {
  // process.argv[1] is the script Node was launched with — the npm
  // CLI for a `node /path/to/npm-cli.js` invocation, or the user
  // script for a `node user-agent.js` invocation.
  const script = process.argv[1];
  if (!script) return false;
  for (const re of _PACKAGE_MANAGER_CLI_PATTERNS) {
    if (re.test(script)) return true;
  }
  return false;
}

async function _start(): Promise<Session | null> {
  if (_bootstrapped) return null;
  _bootstrapped = true;

  const output = process.env.SHADOW_SESSION_OUTPUT;
  if (!output) return null; // No-op — explicit env-var contract.

  if (_is_package_manager_wrapper()) {
    // We're running inside `npm` / `pnpm` / `yarn` itself, not the
    // user's agent. Skip — the agent will load us in its own process.
    return null;
  }

  let session: Session;
  try {
    // Session(opts).enter() already runs `autoInstrument(this)` when
    // `opts.autoInstrument !== false` (the default). Calling
    // `autoInstrument(session)` here in addition would patch the
    // openai/anthropic prototypes a SECOND time — the second wrap
    // saves the first wrapper as its `original` and we end up
    // recording every chat call twice. Don't do that.
    const opts: SessionOptions = {
      outputPath: output,
      tags: _parseTagsEnv(process.env.SHADOW_SESSION_TAGS),
    };
    session = new Session(opts);
    await session.enter();
  } catch (e) {
    _warnNoCrash('failed to enter Session', e);
    return null;
  }

  // Coordinate a single exit pass across the multiple shutdown paths
  // Node can take. process.once handlers prevent re-entry on the
  // common case (SIGINT followed by beforeExit).
  let exiting = false;
  const flush = async (): Promise<void> => {
    if (exiting) return;
    exiting = true;
    try {
      await session.exit();
    } catch (e) {
      _warnNoCrash('failed to flush Session on exit', e);
    }
    // LOUD-FAILURE check: under zero-config shadow record, if the
    // session captured zero chat_request records, the user is in the
    // silent-uninstrument trap that bit BrowserOS / Vercel-AI agents
    // before the LanguageModelV3 prototype patcher landed. Surface
    // the warning to stderr so a CI run can't pass with a misleading
    // metadata-only result. Mirrors the Python autostart fix.
    try {
      const stats = session.recordStats();
      if (stats.chatRequests === 0) {
        _emitEmptyCaptureWarning(output);
      }
    } catch {
      /* diagnostics never block — file is still on disk */
    }
  };

  // Normal exit — beforeExit fires before the event loop drains.
  process.once('beforeExit', () => {
    void flush();
  });

  // SIGINT (^C) and SIGTERM (orchestrator kill) need explicit exit
  // codes — Node's default doesn't run beforeExit on signal-driven
  // shutdown, so we re-issue process.exit after flushing.
  process.once('SIGINT', () => {
    void flush().then(() => process.exit(130));
  });
  process.once('SIGTERM', () => {
    void flush().then(() => process.exit(143));
  });

  // Uncaught exception path — flush, log, propagate the failure.
  process.once('uncaughtException', (err) => {
    void flush().then(() => {
      // eslint-disable-next-line no-console
      console.error(err);
      process.exit(1);
    });
  });

  return session;
}

// Top-level await — ESM only. Node 20.6+ supports `--import` for this.
// Fire-and-forget: we don't propagate errors; _start swallows them.
await _start();

export {}; // Mark as a module so tsc treats top-level statements correctly.
