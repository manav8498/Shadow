/**
 * Stateless policy evaluation against a recorded trace.
 *
 * Mirrors the subset of `shadow.hierarchical.check_policy` covered by
 * the rule kinds in `rules.ts`. Same semantics, byte-identical
 * decisions on the same fixtures (verified by the Python
 * cross-validation test).
 */

import type { AgentlogRecord } from '../agentlog.js';
import type {
  ForbiddenTextRule,
  MustCallBeforeRule,
  MustCallOnceRule,
  MustIncludeTextRule,
  NoCallRule,
  PolicyRule,
  PolicyViolation,
  Severity,
} from './rules.js';

interface ToolCallEvent {
  pairIndex: number;
  tool: string;
}

interface ResponseEvent {
  pairIndex: number;
  text: string;
}

const DEFAULT_SEVERITY: Severity = 'medium';

function severityOf(rule: PolicyRule): Severity {
  return rule.severity ?? DEFAULT_SEVERITY;
}

function extractEvents(records: AgentlogRecord[]): {
  toolCalls: ToolCallEvent[];
  responses: ResponseEvent[];
} {
  const toolCalls: ToolCallEvent[] = [];
  const responses: ResponseEvent[] = [];
  let pairIdx = 0;
  for (const rec of records) {
    if (rec.kind !== 'chat_response') continue;
    const payload = (rec.payload ?? {}) as Record<string, unknown>;
    const content = (payload.content ?? []) as Array<Record<string, unknown>>;
    const textParts: string[] = [];
    for (const block of content) {
      if (typeof block !== 'object' || block === null) continue;
      const type = block.type;
      if (type === 'tool_use') {
        toolCalls.push({ pairIndex: pairIdx, tool: String(block.name ?? '') });
      } else if (type === 'text') {
        const t = block.text;
        if (typeof t === 'string') textParts.push(t);
      }
    }
    responses.push({ pairIndex: pairIdx, text: textParts.join('\n') });
    pairIdx += 1;
  }
  return { toolCalls, responses };
}

function checkNoCall(
  rule: NoCallRule,
  toolCalls: ToolCallEvent[],
): PolicyViolation[] {
  const violations: PolicyViolation[] = [];
  for (const ev of toolCalls) {
    if (ev.tool === rule.tool) {
      violations.push({
        ruleId: rule.ruleId,
        kind: 'no_call',
        severity: severityOf(rule),
        detail: `tool ${rule.tool} was called at pair ${ev.pairIndex} but the policy forbids it`,
        pairIndex: ev.pairIndex,
      });
    }
  }
  return violations;
}

function checkMustCallBefore(
  rule: MustCallBeforeRule,
  toolCalls: ToolCallEvent[],
): PolicyViolation[] {
  const firstIdx = toolCalls.findIndex((e) => e.tool === rule.first);
  const thenIdx = toolCalls.findIndex((e) => e.tool === rule.then);
  // Vacuous when `then` never fires — the rule has no obligation.
  if (thenIdx === -1) return [];
  // Violation when `then` fires without a prior `first`.
  if (firstIdx === -1 || firstIdx > thenIdx) {
    const offending = toolCalls[thenIdx]!;
    return [
      {
        ruleId: rule.ruleId,
        kind: 'must_call_before',
        severity: severityOf(rule),
        detail: `${rule.then} was called at pair ${offending.pairIndex} without a preceding ${rule.first}`,
        pairIndex: offending.pairIndex,
      },
    ];
  }
  return [];
}

function checkMustCallOnce(
  rule: MustCallOnceRule,
  toolCalls: ToolCallEvent[],
): PolicyViolation[] {
  const matches = toolCalls.filter((e) => e.tool === rule.tool);
  // Vacuous when the tool never fires.
  if (matches.length <= 1) return [];
  // Report the second and subsequent calls. The first call is allowed.
  return matches.slice(1).map((m) => ({
    ruleId: rule.ruleId,
    kind: 'must_call_once' as const,
    severity: severityOf(rule),
    detail: `${rule.tool} was called more than once; extra call at pair ${m.pairIndex}`,
    pairIndex: m.pairIndex,
  }));
}

function checkForbiddenText(
  rule: ForbiddenTextRule,
  responses: ResponseEvent[],
): PolicyViolation[] {
  const violations: PolicyViolation[] = [];
  for (const ev of responses) {
    if (ev.text.includes(rule.substring)) {
      violations.push({
        ruleId: rule.ruleId,
        kind: 'forbidden_text',
        severity: severityOf(rule),
        detail: `response at pair ${ev.pairIndex} contains forbidden substring "${rule.substring}"`,
        pairIndex: ev.pairIndex,
      });
    }
  }
  return violations;
}

function checkMustIncludeText(
  rule: MustIncludeTextRule,
  responses: ResponseEvent[],
): PolicyViolation[] {
  const found = responses.some((e) => e.text.includes(rule.substring));
  if (found) return [];
  return [
    {
      ruleId: rule.ruleId,
      kind: 'must_include_text',
      severity: severityOf(rule),
      detail: `no response in the trace contained required substring "${rule.substring}"`,
      pairIndex: null,
    },
  ];
}

/**
 * Evaluate `rules` against `records` and return all violations.
 *
 * The result is sorted by `(pairIndex ?? Infinity, ruleId)` so the
 * order is stable across runs and identical to the Python output.
 */
export function checkPolicy(
  records: AgentlogRecord[],
  rules: PolicyRule[],
): PolicyViolation[] {
  const { toolCalls, responses } = extractEvents(records);
  const violations: PolicyViolation[] = [];

  for (const rule of rules) {
    switch (rule.kind) {
      case 'no_call':
        violations.push(...checkNoCall(rule, toolCalls));
        break;
      case 'must_call_before':
        violations.push(...checkMustCallBefore(rule, toolCalls));
        break;
      case 'must_call_once':
        violations.push(...checkMustCallOnce(rule, toolCalls));
        break;
      case 'forbidden_text':
        violations.push(...checkForbiddenText(rule, responses));
        break;
      case 'must_include_text':
        violations.push(...checkMustIncludeText(rule, responses));
        break;
    }
  }

  violations.sort((a, b) => {
    const ai = a.pairIndex ?? Number.MAX_SAFE_INTEGER;
    const bi = b.pairIndex ?? Number.MAX_SAFE_INTEGER;
    if (ai !== bi) return ai - bi;
    return a.ruleId < b.ruleId ? -1 : a.ruleId > b.ruleId ? 1 : 0;
  });
  return violations;
}
