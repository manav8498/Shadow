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
});
