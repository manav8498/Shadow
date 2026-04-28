/**
 * Gate decision: combine LTLf safety formulas and policy rules into a
 * single pass/fail verdict against a recorded trace.
 *
 * Designed for CI gating: a TS-only team running the SDK can call
 * `gate(records, { rules, ltlFormulas })` and exit 1 on `!result.passed`
 * without invoking Python.
 *
 * The result is structured so a CI step can render either a short
 * human summary (`renderGateSummary`) or pipe the JSON to a downstream
 * tool. Decisions are byte-identical to the Python equivalent on the
 * same inputs (verified by `python/tests/test_typescript_parity.py`).
 */

import type { AgentlogRecord } from '../agentlog.js';
import {
  check as ltlCheck,
  traceFromRecords,
  type EvalFn,
} from '../ltl/checker.js';
import type { Formula } from '../ltl/formula.js';
import { formulaToString } from '../ltl/formula.js';
import { checkPolicy } from '../policy/check.js';
import type { PolicyRule, PolicyViolation } from '../policy/rules.js';

export interface LtlResult {
  /** Human-readable rendering of the formula (`G(¬tool_call:foo)` etc.). */
  formula: string;
  /** True iff the formula holds at position 0. */
  passed: boolean;
}

export interface GateOptions {
  /** Policy rules to evaluate (no_call / must_call_before / ...). */
  rules?: PolicyRule[];
  /** LTLf safety formulas to evaluate at trace position 0. */
  ltlFormulas?: Formula[];
  /** Optional custom predicate evaluator passed to the LTLf checker. */
  ltlEvalFn?: EvalFn;
}

export interface GateResult {
  /** Overall pass/fail. False iff any policy violation OR any LTL formula fails. */
  passed: boolean;
  /** All policy violations from the rule list. */
  violations: PolicyViolation[];
  /** Per-formula LTL evaluation results. */
  ltlResults: LtlResult[];
}

/**
 * Evaluate `records` against the supplied gate configuration.
 *
 * The verdict is the conjunction of:
 *   - all `rules` produce zero violations
 *   - all `ltlFormulas` evaluate to true at position 0 of the trace
 *
 * Empty `rules`/`ltlFormulas` are vacuously satisfied. An empty trace
 * passes any safety formula whose semantics permit the empty case
 * (e.g. `G(¬X)`); strong-existence formulas like `F(X)` fail.
 */
export function gate(
  records: AgentlogRecord[],
  options: GateOptions = {},
): GateResult {
  const rules = options.rules ?? [];
  const formulas = options.ltlFormulas ?? [];

  const violations = rules.length > 0 ? checkPolicy(records, rules) : [];

  let ltlResults: LtlResult[] = [];
  if (formulas.length > 0) {
    const trace = traceFromRecords(records);
    ltlResults = formulas.map((phi) => ({
      formula: formulaToString(phi),
      passed: ltlCheck(phi, trace, 0, options.ltlEvalFn),
    }));
  }

  const passed =
    violations.length === 0 && ltlResults.every((r) => r.passed);
  return { passed, violations, ltlResults };
}

/**
 * Render a one-paragraph summary of a `GateResult`. Stable, machine-
 * parseable: callers can pipe this into `gh pr comment --body` or a
 * `console.error` line in their test runner.
 */
export function renderGateSummary(result: GateResult): string {
  if (result.passed) {
    const checks =
      result.violations.length +
      result.ltlResults.length;
    return `[shadow.gate] PASSED — ${checks} check(s) cleared.`;
  }
  const lines: string[] = [];
  lines.push(`[shadow.gate] FAILED`);
  if (result.violations.length > 0) {
    lines.push(`  policy violations: ${result.violations.length}`);
    for (const v of result.violations) {
      const at = v.pairIndex === null ? '<trace>' : `pair=${v.pairIndex}`;
      lines.push(`    - ${v.ruleId} [${v.kind}/${v.severity}] ${at}: ${v.detail}`);
    }
  }
  const failedLtl = result.ltlResults.filter((r) => !r.passed);
  if (failedLtl.length > 0) {
    lines.push(`  LTL failures: ${failedLtl.length}`);
    for (const r of failedLtl) {
      lines.push(`    - ${r.formula}`);
    }
  }
  return lines.join('\n');
}
