/**
 * Policy rule types for runtime gating.
 *
 * The TypeScript SDK ships a focused subset of the Python
 * `shadow.hierarchical` rule kinds that are most useful for CI gating
 * decisions on a recorded trace:
 *
 * - `no_call`            — tool name must never appear
 * - `must_call_before`   — tool A must precede tool B if both are called
 * - `must_call_once`     — tool must appear exactly once if it appears
 * - `forbidden_text`     — substring must not appear in any response text
 * - `must_include_text`  — substring must appear in at least one response
 *
 * The full Python rule set (`must_be_grounded`, `must_remain_consistent`,
 * `must_followup`, JSON schema, token budgets, etc.) is intentionally
 * *not* ported — those rules either depend on RAG context, judge
 * scoring, or stateful obligations that don't translate cleanly to a
 * pure-TS environment. TS-only teams that need them invoke the Python
 * CLI on the same trace.
 */

export type Severity = 'low' | 'medium' | 'high' | 'critical';

export interface NoCallRule {
  kind: 'no_call';
  ruleId: string;
  tool: string;
  severity?: Severity;
}

export interface MustCallBeforeRule {
  kind: 'must_call_before';
  ruleId: string;
  first: string;
  then: string;
  severity?: Severity;
}

export interface MustCallOnceRule {
  kind: 'must_call_once';
  ruleId: string;
  tool: string;
  severity?: Severity;
}

export interface ForbiddenTextRule {
  kind: 'forbidden_text';
  ruleId: string;
  substring: string;
  severity?: Severity;
}

export interface MustIncludeTextRule {
  kind: 'must_include_text';
  ruleId: string;
  substring: string;
  severity?: Severity;
}

export type PolicyRule =
  | NoCallRule
  | MustCallBeforeRule
  | MustCallOnceRule
  | ForbiddenTextRule
  | MustIncludeTextRule;

export interface PolicyViolation {
  ruleId: string;
  kind: PolicyRule['kind'];
  severity: Severity;
  detail: string;
  /**
   * Index of the chat-pair where the violation manifests, or null when
   * the violation is a property of the whole trace (e.g. a missing
   * required call).
   */
  pairIndex: number | null;
}
