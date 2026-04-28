/**
 * Linear Temporal Logic (LTLf) formula AST.
 *
 * Discriminated union mirroring the Python `shadow.ltl.formula` types
 * one-to-one. Same operator semantics (finite-trace LTL), same
 * boundary behaviour at the end of the trace, same string forms for
 * atomic predicates ("tool_call:<name>", "stop_reason:<value>",
 * "text_contains:<substr>", "true", "false", "extra:<key>=<value>").
 *
 * The Python AST and this TypeScript AST are kept byte-identical at
 * the JSON level so a formula serialised in either language parses
 * cleanly in the other.
 */

export type Formula =
  | { kind: 'atom'; pred: string }
  | { kind: 'not'; child: Formula }
  | { kind: 'and'; left: Formula; right: Formula }
  | { kind: 'or'; left: Formula; right: Formula }
  | { kind: 'implies'; left: Formula; right: Formula }
  | { kind: 'next'; child: Formula }
  | { kind: 'until'; left: Formula; right: Formula }
  | { kind: 'weakUntil'; left: Formula; right: Formula }
  | { kind: 'globally'; child: Formula }
  | { kind: 'finally'; child: Formula };

export const TRUE: Formula = { kind: 'atom', pred: 'true' };
export const FALSE: Formula = { kind: 'atom', pred: 'false' };

export function atom(pred: string): Formula {
  return { kind: 'atom', pred };
}

export function not(child: Formula): Formula {
  return { kind: 'not', child };
}

export function and(left: Formula, right: Formula): Formula {
  return { kind: 'and', left, right };
}

export function or(left: Formula, right: Formula): Formula {
  return { kind: 'or', left, right };
}

export function implies(left: Formula, right: Formula): Formula {
  return { kind: 'implies', left, right };
}

export function next(child: Formula): Formula {
  return { kind: 'next', child };
}

export function globallyOp(child: Formula): Formula {
  return { kind: 'globally', child };
}

export function finallyOp(child: Formula): Formula {
  return { kind: 'finally', child };
}

export function until(left: Formula, right: Formula): Formula {
  return { kind: 'until', left, right };
}

export function weakUntil(left: Formula, right: Formula): Formula {
  return { kind: 'weakUntil', left, right };
}

export function conj(...args: Formula[]): Formula {
  if (args.length === 0) return TRUE;
  let result = args[0]!;
  for (let i = 1; i < args.length; i++) {
    result = and(result, args[i]!);
  }
  return result;
}

export function disj(...args: Formula[]): Formula {
  if (args.length === 0) return FALSE;
  let result = args[0]!;
  for (let i = 1; i < args.length; i++) {
    result = or(result, args[i]!);
  }
  return result;
}

export function formulaToString(f: Formula): string {
  switch (f.kind) {
    case 'atom':
      return f.pred;
    case 'not':
      return `¬(${formulaToString(f.child)})`;
    case 'and':
      return `(${formulaToString(f.left)} ∧ ${formulaToString(f.right)})`;
    case 'or':
      return `(${formulaToString(f.left)} ∨ ${formulaToString(f.right)})`;
    case 'implies':
      return `(${formulaToString(f.left)} → ${formulaToString(f.right)})`;
    case 'next':
      return `X(${formulaToString(f.child)})`;
    case 'until':
      return `(${formulaToString(f.left)} U ${formulaToString(f.right)})`;
    case 'weakUntil':
      return `(${formulaToString(f.left)} W ${formulaToString(f.right)})`;
    case 'globally':
      return `G(${formulaToString(f.child)})`;
    case 'finally':
      return `F(${formulaToString(f.child)})`;
  }
}
