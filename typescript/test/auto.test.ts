/**
 * Tests for `shadow-diff/auto` — the zero-config entrypoint that
 * `shadow record -- node ...` injects via NODE_OPTIONS.
 *
 * The entrypoint can't be imported in the same process twice with
 * different env vars (its module body runs at import time and is
 * cached), so these tests spawn child processes with `--import` to
 * exercise the real shape end-to-end.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const distAuto = fileURLToPath(new URL('../dist/auto.js', import.meta.url));

function runChild(env: NodeJS.ProcessEnv, code: string): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync(
    process.execPath,
    ['--import', distAuto, '-e', code],
    { env: { ...process.env, ...env }, encoding: 'utf-8' },
  );
  return { stdout: result.stdout, stderr: result.stderr, status: result.status };
}

describe('shadow-diff/auto', () => {
  it('is a no-op when SHADOW_SESSION_OUTPUT is unset', () => {
    const r = runChild({ SHADOW_SESSION_OUTPUT: undefined }, 'console.log("ok")');
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('ok');
    // No warning prefix should appear when the env var is absent —
    // the auto module is a deliberate no-op.
    expect(r.stderr).not.toContain('[shadow-diff/auto]');
  });

  it('writes a .agentlog when SHADOW_SESSION_OUTPUT is set and the agent exits cleanly', () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');
    const r = runChild(
      { SHADOW_SESSION_OUTPUT: out },
      'console.log("agent ran")',
    );
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('agent ran');
    expect(existsSync(out)).toBe(true);
    const content = readFileSync(out, 'utf-8');
    // First line must be the metadata envelope record.
    const firstLine = content.split('\n')[0];
    const meta = JSON.parse(firstLine);
    expect(meta.kind).toBe('metadata');
    expect(meta.id).toMatch(/^sha256:/);
  });

  it('parses SHADOW_SESSION_TAGS into the metadata payload', () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');
    const r = runChild(
      {
        SHADOW_SESSION_OUTPUT: out,
        SHADOW_SESSION_TAGS: 'env=ci,framework=langgraph',
      },
      'console.log("done")',
    );
    expect(r.status).toBe(0);
    const meta = JSON.parse(readFileSync(out, 'utf-8').split('\n')[0]);
    expect(meta.payload.tags).toEqual({ env: 'ci', framework: 'langgraph' });
  });

  it('tolerates malformed SHADOW_SESSION_TAGS without crashing the agent', () => {
    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');
    const r = runChild(
      {
        SHADOW_SESSION_OUTPUT: out,
        // Missing `=`, leading commas, empty pairs — all malformed,
        // none should fail the bootstrap.
        SHADOW_SESSION_TAGS: 'no_equals,,,=novalue,=,,key=value,',
      },
      'console.log("done")',
    );
    expect(r.status).toBe(0);
    // Only the well-formed pair should survive.
    const meta = JSON.parse(readFileSync(out, 'utf-8').split('\n')[0]);
    expect(meta.payload.tags).toEqual({ key: 'value' });
  });

  it('does NOT crash the user agent when given an unwritable path', () => {
    // /dev/null/foo is not a writable directory — the auto entry must
    // log a warning to stderr and let the user code keep running.
    const r = runChild(
      { SHADOW_SESSION_OUTPUT: '/dev/null/cant-write/trace.agentlog' },
      'console.log("user code ran anyway")',
    );
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('user code ran anyway');
    expect(r.stderr).toContain('[shadow-diff/auto]');
  });

  it('records ONE chat_request + ONE chat_response per real chat call (no duplicates)', async () => {
    // Regression: when auto.ts also called autoInstrument(session)
    // after Session.enter() (which already auto-instruments), every
    // chat call recorded TWICE — the second wrapper saw the first
    // wrapper as its `original` and chained the records. The fix
    // removes the redundant call AND tags wrappers with a Symbol so
    // future patchCreate calls skip already-wrapped prototypes.
    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');

    // Inline child program that:
    //  1. Stubs global fetch so no network is needed.
    //  2. Imports openai dynamically (matches real-world ESM usage).
    //  3. Issues exactly ONE chat.completions.create.
    // The auto entrypoint is loaded via --import on the runChild
    // command line below, so by the time this code runs Session is
    // already entered + openai is patched.
    const code = `
      globalThis.fetch = async () => new Response(JSON.stringify({
        id: 'cmpl-1',
        choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
      }), { status: 200, headers: { 'content-type': 'application/json' } });
      const { default: OpenAI } = await import('openai');
      const c = new OpenAI({ apiKey: 'x' });
      await c.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'hi' }],
      });
    `;

    // Skip when openai isn't installed in the parent's node_modules
    // (CI installs it via the test deps; local dev may not).
    let openaiAvailable = true;
    try {
      await import('openai');
    } catch {
      openaiAvailable = false;
    }
    if (!openaiAvailable) return;

    const r = runChild({ SHADOW_SESSION_OUTPUT: out }, code);
    expect(r.status).toBe(0);
    expect(existsSync(out)).toBe(true);

    const lines = readFileSync(out, 'utf-8').split('\n').filter(Boolean);
    const kinds = lines.map((l) => JSON.parse(l).kind);
    // Exactly one of each (plus the metadata at index 0).
    expect(kinds.filter((k) => k === 'chat_request')).toHaveLength(1);
    expect(kinds.filter((k) => k === 'chat_response')).toHaveLength(1);
  });

  it('preserves recorded chat pairs even when the agent crashes after recording', async () => {
    // Regression: Session.exit() previously buffered all records in
    // memory and flushed via writeFile() at exit. An agent that
    // process.exit(1)'d or threw uncaughtException after a recorded
    // chat call would leave a 0-byte trace on disk. Per-record
    // appendFileSync in recordChat() means the records hit disk
    // immediately and survive any exit path.
    let openaiAvailable = true;
    try {
      await import('openai');
    } catch {
      openaiAvailable = false;
    }
    if (!openaiAvailable) return;

    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');

    // Record one chat call, then hard-exit with a non-zero code.
    // beforeExit will NOT fire on process.exit() — the test verifies
    // the file already has the records BEFORE any exit handler.
    const code = `
      globalThis.fetch = async () => new Response(JSON.stringify({
        id: 'cmpl-1',
        choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
      }), { status: 200, headers: { 'content-type': 'application/json' } });
      const { default: OpenAI } = await import('openai');
      const c = new OpenAI({ apiKey: 'x' });
      await c.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'hi' }],
      });
      // Hard exit — skips beforeExit, never runs Session.exit() flush.
      process.exit(42);
    `;

    const r = runChild({ SHADOW_SESSION_OUTPUT: out }, code);
    expect(r.status).toBe(42);
    // Trace file MUST exist and contain the metadata + the chat pair.
    expect(existsSync(out)).toBe(true);
    const lines = readFileSync(out, 'utf-8').split('\n').filter(Boolean);
    const kinds = lines.map((l) => JSON.parse(l).kind);
    expect(kinds[0]).toBe('metadata');
    expect(kinds.filter((k) => k === 'chat_request')).toHaveLength(1);
    expect(kinds.filter((k) => k === 'chat_response')).toHaveLength(1);
  });

  it('skips the bootstrap when running inside an npm CLI wrapper', () => {
    // `shadow record -- npm start` injects NODE_OPTIONS on the npm
    // process AND its grandchild. Without the wrapper-skip check
    // both would write to SHADOW_SESSION_OUTPUT and (since Session
    // uses writeFile, not append) one would clobber the other.
    // We simulate the wrapper case by setting argv[1] to a path
    // matching the package-manager CLI pattern.
    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');

    // Spawn a fake "npm-cli.js" so process.argv[1] looks like npm.
    const fakeNpmDir = mkdtempSync(join(tmpdir(), 'fake-npm-'));
    const fakeNpm = join(fakeNpmDir, 'npm-cli.js');
    require('node:fs').writeFileSync(fakeNpm, 'console.log("would-be-npm");');

    const result = spawnSync(
      process.execPath,
      ['--import', distAuto, fakeNpm],
      {
        env: { ...process.env, SHADOW_SESSION_OUTPUT: out },
        encoding: 'utf-8',
      },
    );
    expect(result.status).toBe(0);
    // Wrapper-skip path = no trace file created (auto returned early).
    // The would-be-npm script ran fine.
    expect(result.stdout).toContain('would-be-npm');
    expect(existsSync(out)).toBe(false);
  });

  it('emits a loud zero-capture warning when no chat calls were recorded', () => {
    // Mirrors the Python autostart fix in c815e34: when the agent exits
    // and the session captured zero chat_request records, stderr must
    // carry a warning naming the canonical causes. Without this, CI runs
    // pass silently with metadata-only traces (the BrowserOS / Vercel AI
    // pre-v3.1.1 failure mode the customer reported).
    const dir = mkdtempSync(join(tmpdir(), 'shadow-auto-'));
    const out = join(dir, 'trace.agentlog');
    const r = runChild(
      { SHADOW_SESSION_OUTPUT: out },
      'console.log("agent ran but made no LLM calls")',
    );
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('agent ran but made no LLM calls');
    // The loud stderr warning is the key assertion.
    expect(r.stderr).toContain('WARNING');
    expect(r.stderr).toContain('zero LLM calls were captured');
    // Names the four canonical causes so users can self-diagnose.
    expect(r.stderr.toLowerCase()).toContain('shadow does not yet auto-instrument');
    expect(r.stderr).toContain('npm install shadow-diff');
    expect(r.stderr).toContain('bound-method reference');
    // Names the supported SDK list so users know what to expect.
    expect(r.stderr).toContain('openai');
    expect(r.stderr).toContain('@anthropic-ai/sdk');
    expect(r.stderr).toContain('@ai-sdk');
  });
});
