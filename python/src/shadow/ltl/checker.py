"""Finite-trace LTL model checker.

Evaluates LTL formulas over finite sequences of trace states using the
standard finite-trace semantics (also called LTLf):

    - G φ  at position i  ⟺  ∀ j ≥ i: φ holds at j
    - F φ  at position i  ⟺  ∃ j ≥ i: φ holds at j
    - X φ  at position i  ⟺  i+1 < len(π) and φ holds at i+1
    - φ U ψ at position i ⟺  ∃ j ≥ i: ψ holds at j ∧ ∀ k ∈ [i,j): φ holds at k
    - φ W ψ at position i ⟺  (φ U ψ)@i ∨ (∀ j ≥ i: φ holds at j)

At the end of the trace (i == len(π)) the semantics are:

    - Atom, Not, And, Or, Implies: evaluated normally
    - Next(φ): False (no next state exists)
    - G(φ): True (vacuously)
    - F(φ): False
    - U(φ, ψ): False (ψ never becomes true)
    - W(φ, ψ): True (φ vacuously holds forever)

This "weak" finite-trace convention means G(¬call_X) passes when
call_X never appears in the trace — the safety reading of "globally,
don't call X."

Algorithm
---------
The checker compiles the formula once and evaluates it over the whole
trace using **bottom-up dynamic programming**.  For each subformula φ',
we compute a length-|π| boolean vector ``holds[i]`` in a single pass:

    - Atom: O(|π|) one eval per position
    - Not / And / Or / Implies: O(|π|) elementwise
    - Next: O(|π|) shift
    - Globally / Finally / Until / WeakUntil: O(|π|) right-to-left sweep
      using the recurrences:
        G(φ)[i]    = φ[i] ∧ G(φ)[i+1]   ; G(φ)[n] = True
        F(φ)[i]    = φ[i] ∨ F(φ)[i+1]   ; F(φ)[n] = False
        (φ U ψ)[i] = ψ[i] ∨ (φ[i] ∧ (φ U ψ)[i+1])    ; [n] = False
        (φ W ψ)[i] = ψ[i] ∨ (φ[i] ∧ (φ W ψ)[i+1])    ; [n] = True

Total complexity: **O(|π| × |φ|)** where |φ| is the number of
subformula nodes.  This is asymptotically optimal for finite-trace
LTL without automaton compilation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shadow.ltl.formula import (
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
)


@dataclass
class TraceState:
    """One step in a trace as seen by the LTL checker.

    Constructed from a chat_response record (and its paired request).
    Multiple tool calls in one turn are all "present" at this state.
    """

    pair_index: int
    tool_calls: list[str] = field(default_factory=list)
    """Names of tool_use content blocks emitted in this response."""
    stop_reason: str = ""
    text_content: str = ""

    # Extra key-value store for custom predicates.
    extra: dict[str, Any] = field(default_factory=dict)


EvalFn = Callable[[str, TraceState], bool]
"""Custom atomic-predicate evaluator. ``(pred, state) -> bool``."""


def default_eval(pred: str, state: TraceState) -> bool:
    """Built-in predicate evaluator.

    Supported ``pred`` forms:

    - ``"true"`` / ``"false"``
    - ``"tool_call:<name>"`` — state.tool_calls contains <name>
    - ``"stop_reason:<value>"`` — state.stop_reason == <value>
    - ``"text_contains:<substr>"`` — <substr> in state.text_content
    - ``"extra:<key>=<value>"`` — state.extra[key] == value (string eq)
    """
    if pred == "true":
        return True
    if pred == "false":
        return False
    if pred.startswith("tool_call:"):
        name = pred[len("tool_call:") :]
        return name in state.tool_calls
    if pred.startswith("stop_reason:"):
        value = pred[len("stop_reason:") :]
        return state.stop_reason == value
    if pred.startswith("text_contains:"):
        substr = pred[len("text_contains:") :]
        return substr in state.text_content
    if pred.startswith("extra:"):
        rest = pred[len("extra:") :]
        if "=" in rest:
            k, v = rest.split("=", 1)
            return str(state.extra.get(k, "")) == v
    return False


def eval_all_positions(
    formula: Formula,
    trace: list[TraceState],
    eval_fn: EvalFn = default_eval,
) -> list[bool]:
    """Compute the truth value of ``formula`` at every position 0..n.

    Returns a list of length ``len(trace) + 1`` where ``out[i]`` is
    True iff ``formula`` holds at position i (and ``out[n]`` is the
    end-of-trace value).

    This is the canonical bottom-up DP entry point.  Total work is
    O(|π| × |φ|) where |φ| is the number of distinct subformula nodes.
    """
    n = len(trace)
    cache: dict[Formula, list[bool]] = {}
    return _holds(formula, trace, n, eval_fn, cache)


def check(
    formula: Formula,
    trace: list[TraceState],
    i: int = 0,
    eval_fn: EvalFn = default_eval,
) -> bool:
    """Evaluate ``formula`` on ``trace`` starting at position ``i``."""
    arr = eval_all_positions(formula, trace, eval_fn)
    if not (0 <= i <= len(trace)):
        raise IndexError(f"position {i} out of range for trace of length {len(trace)}")
    return arr[i]


def _holds(
    phi: Formula,
    trace: list[TraceState],
    n: int,
    eval_fn: EvalFn,
    cache: dict[Formula, list[bool]],
) -> list[bool]:
    """Bottom-up DP. Returns truth-vector of length n+1 for ``phi``."""
    cached = cache.get(phi)
    if cached is not None:
        return cached

    out: list[bool]

    if isinstance(phi, Atom):
        # End-of-trace: only the literal "true" holds.
        out = [eval_fn(phi.pred, trace[i]) for i in range(n)]
        out.append(phi.pred == "true")

    elif isinstance(phi, Not):
        child = _holds(phi.child, trace, n, eval_fn, cache)
        out = [not v for v in child]

    elif isinstance(phi, And):
        a = _holds(phi.left, trace, n, eval_fn, cache)
        b = _holds(phi.right, trace, n, eval_fn, cache)
        out = [a[i] and b[i] for i in range(n + 1)]

    elif isinstance(phi, Or):
        a = _holds(phi.left, trace, n, eval_fn, cache)
        b = _holds(phi.right, trace, n, eval_fn, cache)
        out = [a[i] or b[i] for i in range(n + 1)]

    elif isinstance(phi, Implies):
        a = _holds(phi.left, trace, n, eval_fn, cache)
        b = _holds(phi.right, trace, n, eval_fn, cache)
        out = [(not a[i]) or b[i] for i in range(n + 1)]

    elif isinstance(phi, Next):
        child = _holds(phi.child, trace, n, eval_fn, cache)
        # X(φ)[i] = φ[i+1] for i < n; X(φ)[n] = False
        out = [False] * (n + 1)
        for i in range(n):
            out[i] = child[i + 1] if (i + 1) < n else False
        out[n] = False

    elif isinstance(phi, Globally):
        # G(φ)[i] = φ[i] ∧ G(φ)[i+1]; G(φ)[n] = True
        child = _holds(phi.child, trace, n, eval_fn, cache)
        out = [True] * (n + 1)
        for i in range(n - 1, -1, -1):
            out[i] = child[i] and out[i + 1]

    elif isinstance(phi, Finally):
        # F(φ)[i] = φ[i] ∨ F(φ)[i+1]; F(φ)[n] = False
        child = _holds(phi.child, trace, n, eval_fn, cache)
        out = [False] * (n + 1)
        for i in range(n - 1, -1, -1):
            out[i] = child[i] or out[i + 1]

    elif isinstance(phi, Until):
        # (φ U ψ)[i] = ψ[i] ∨ (φ[i] ∧ (φ U ψ)[i+1]); [n] = False
        a = _holds(phi.left, trace, n, eval_fn, cache)
        b = _holds(phi.right, trace, n, eval_fn, cache)
        out = [False] * (n + 1)
        for i in range(n - 1, -1, -1):
            out[i] = b[i] or (a[i] and out[i + 1])

    elif isinstance(phi, WeakUntil):
        # (φ W ψ)[i] = ψ[i] ∨ (φ[i] ∧ (φ W ψ)[i+1]); [n] = True
        a = _holds(phi.left, trace, n, eval_fn, cache)
        b = _holds(phi.right, trace, n, eval_fn, cache)
        out = [True] * (n + 1)
        for i in range(n - 1, -1, -1):
            out[i] = b[i] or (a[i] and out[i + 1])

    else:
        raise TypeError(f"unknown LTL formula node type: {type(phi).__name__}")

    cache[phi] = out
    return out


# Legacy single-position recursive entry point. Kept for callers that
# want a one-shot answer at a specific index without computing the
# full truth-vector. Internally still uses the DP routine.
def _check(
    phi: Formula,
    trace: list[TraceState],
    i: int,
    eval_fn: EvalFn,
    memo: dict[tuple[int, Formula], bool],
) -> bool:
    arr = eval_all_positions(phi, trace, eval_fn)
    if not (0 <= i <= len(trace)):
        return False
    return arr[i]


def trace_from_records(records: list[dict[str, Any]]) -> list[TraceState]:
    """Build a list of ``TraceState`` from agentlog records.

    One ``TraceState`` per ``chat_response`` record, populated from
    the response payload (tool_use content blocks, stop_reason, text).
    """
    states: list[TraceState] = []
    pair_idx = 0
    for rec in records:
        if rec.get("kind") != "chat_response":
            continue
        payload = rec.get("payload") or {}
        content = payload.get("content") or []
        tool_calls: list[str] = []
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(str(block.get("name") or ""))
            elif block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        states.append(
            TraceState(
                pair_index=pair_idx,
                tool_calls=tool_calls,
                stop_reason=str(payload.get("stop_reason") or ""),
                text_content="\n".join(text_parts),
            )
        )
        pair_idx += 1
    return states


__all__ = [
    "EvalFn",
    "TraceState",
    "check",
    "default_eval",
    "eval_all_positions",
    "trace_from_records",
]
