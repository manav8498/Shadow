"""Linear Temporal Logic (LTL) policy verification for Shadow.

Three layers:

- :mod:`formula` — immutable AST nodes: ``Atom``, ``Not``, ``And``,
  ``Or``, ``Implies``, ``Next``, ``Until``, ``Globally``, ``Finally``.

- :mod:`checker` — finite-trace model checker: evaluates an LTL formula
  against a list of ``TraceState`` objects (one per agent turn).

- :mod:`compiler` — compiles Shadow YAML policy rules to LTL formulas
  and parses raw LTL strings.

Quickstart
----------
```python
from shadow.ltl import check_trace, compile_rule

# Check "globally no call to delete_all" against a recorded trace.
formula = compile_rule("no_call", {"tool": "delete_all"})
violations = check_trace(formula, records)  # returns list of pair indices
```
"""

from shadow.ltl.checker import (
    EvalFn,
    TraceState,
    check,
    default_eval,
    eval_all_positions,
    trace_from_records,
)
from shadow.ltl.compiler import parse_ltl, rule_to_ltl
from shadow.ltl.formula import (
    FALSE,
    TRUE,
    And,
    Atom,
    Finally,
    Formula,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    Until,
    WeakUntil,
    atom,
    conj,
    disj,
    f,
    g,
    implies,
    neg,
    u,
    w,
    x,
)


def compile_rule(kind: str, params: dict[str, object]) -> "Formula | None":
    """Compile a YAML policy rule kind+params to an LTL formula.

    Returns None for rule kinds that have no LTL encoding (procedural
    rules such as ``max_turns``, ``must_match_json_schema``, etc.).
    """
    return rule_to_ltl(kind, params)


def check_trace(
    formula: "Formula",
    records: list[dict[str, object]],
    eval_fn: "EvalFn | None" = None,
) -> list[int]:
    """Model-check ``formula`` against a trace; return violating pair indices.

    Returns an empty list when the formula is satisfied at position 0.
    Otherwise returns the pair indices of every state where the formula
    does not hold.  For globally-scoped safety properties (G φ) this is
    the natural list of "first violating state, then all subsequent
    states where the violation persists."

    The full truth-vector is computed once via bottom-up DP, so the
    cost is O(|trace| × |formula|) regardless of how many positions
    violate.
    """
    states = trace_from_records(records)
    if not states:
        return []

    _eval = eval_fn if eval_fn is not None else default_eval
    arr = eval_all_positions(formula, states, _eval)

    # Position-0 short-circuit: if the formula holds at the start, the
    # trace is safe under standard safety-property semantics.
    if arr[0]:
        return []

    return [states[i].pair_index for i in range(len(states)) if not arr[i]]


__all__ = [
    "FALSE",
    "TRUE",
    "And",
    "Atom",
    "EvalFn",
    "Finally",
    "Formula",
    "Globally",
    "Implies",
    "Next",
    "Not",
    "Or",
    "TraceState",
    "Until",
    "WeakUntil",
    "atom",
    "check",
    "check_trace",
    "compile_rule",
    "conj",
    "default_eval",
    "disj",
    "eval_all_positions",
    "f",
    "g",
    "implies",
    "neg",
    "parse_ltl",
    "rule_to_ltl",
    "trace_from_records",
    "u",
    "w",
    "x",
]
