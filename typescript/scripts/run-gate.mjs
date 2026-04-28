#!/usr/bin/env node
/**
 * Cross-validation harness for the TypeScript gate against the Python
 * implementation.
 *
 * Reads a JSON object on stdin:
 *   { records: AgentlogRecord[], rules: PolicyRule[], ltlFormulas: Formula[] }
 *
 * Runs `gate()` and writes a JSON object to stdout:
 *   {
 *     passed: boolean,
 *     violations: [{ ruleId, kind, pairIndex }, ...],
 *     ltlResults: [{ formula, passed }, ...]
 *   }
 *
 * The output is deliberately minimal: callers compare on the
 * `(ruleId, pairIndex, kind)` tuples and the per-formula pass/fail.
 * Detail strings are language-specific and intentionally omitted.
 */

import { readFileSync } from 'node:fs';
import { gate } from '../dist/gate/index.js';

function main() {
  const stdin = readFileSync(0, 'utf-8');
  const input = JSON.parse(stdin);
  const result = gate(input.records ?? [], {
    rules: input.rules ?? [],
    ltlFormulas: input.ltlFormulas ?? [],
  });
  const out = {
    passed: result.passed,
    violations: result.violations.map((v) => ({
      ruleId: v.ruleId,
      kind: v.kind,
      pairIndex: v.pairIndex,
    })),
    ltlResults: result.ltlResults.map((r) => ({
      formula: r.formula,
      passed: r.passed,
    })),
  };
  process.stdout.write(JSON.stringify(out));
}

main();
