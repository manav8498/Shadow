"""Compile YAML policy rules to LTL formulas.

Each ``PolicyRule.kind`` maps to an LTL formula over tool-call and
stop-reason atoms.  This gives policy rules a formally specified
semantics that can be model-checked, composed, and documented.

Supported kinds and their LTL translations
------------------------------------------
``must_call_before(first=A, then=B)``
    B may only appear after A:  ``(¬B) W A`` (B is blocked until A fires,
    or forever if A never fires).  Uses *weak* until: a trace that calls
    neither A nor B vacuously satisfies the rule, which matches the
    intuitive reading.  (Strong-until ``(¬B) U A`` would incorrectly
    fail when A never fires.)

``no_call(tool=A)``
    ``G(¬tool_call:A)``
    A is globally forbidden.

``must_call_once(tool=A)``
    ``F(tool_call:A) ∧ G(tool_call:A → X(G(¬tool_call:A)))``
    A must be called exactly once. The second conjunct ensures that
    once A fires, it never fires again.

``required_stop_reason(allowed=[r1, r2, ...])``
    The final stop reason must be in the allowed set:
    ``F(stop_reason:r1 ∨ stop_reason:r2 ∨ ...)``
    (Finite-trace: "eventually, at least one allowed stop reason is
    reached." For single-turn agents this is the only stop reason.)

``forbidden_text(text=T)``
    ``G(¬text_contains:T)``

``must_include_text(text=T)``
    ``F(text_contains:T)``

``ltl_formula(formula=<string>)``
    Raw LTL parsed from a string using :func:`parse_ltl`.
    Syntax: standard LTL with the operators G, F, X, U, W, !, &, |, ->
    and atoms using the ``pred:value`` convention.

Kinds that do not have a clean LTL encoding (``max_turns``,
``max_total_tokens``, ``must_match_json_schema``,
``must_remain_consistent``, ``must_followup``, ``must_be_grounded``)
are left as ``None`` — the caller should fall back to the existing
procedural checker for those.
"""

from __future__ import annotations

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
    disj,
)


def rule_to_ltl(kind: str, params: dict[str, Any]) -> Formula | None:
    """Compile a policy rule kind+params to an LTL formula.

    Returns ``None`` for kinds that have no LTL encoding in this
    version (procedural rules stay procedural).
    """
    if kind == "no_call":
        tool = params.get("tool")
        if not isinstance(tool, str):
            return None
        return Globally(Not(Atom(f"tool_call:{tool}")))

    if kind == "must_call_before":
        first = params.get("first")
        then = params.get("then")
        if not isinstance(first, str) or not isinstance(then, str):
            return None
        # (¬B) W A — B is blocked until A fires (or forever).
        # Weak-until matches the intuitive rule: a trace that never
        # calls B is safe even if A also never fires.
        return WeakUntil(Not(Atom(f"tool_call:{then}")), Atom(f"tool_call:{first}"))

    if kind == "must_call_once":
        tool = params.get("tool")
        if not isinstance(tool, str):
            return None
        call = Atom(f"tool_call:{tool}")
        no_call = Not(call)
        # F(call) ∧ G(call → X(G(¬call)))
        must_eventually = Finally(call)
        no_repeat = Globally(Implies(call, Next(Globally(no_call))))
        return And(must_eventually, no_repeat)

    if kind == "required_stop_reason":
        allowed = params.get("allowed")
        if isinstance(allowed, str):
            allowed = [allowed]
        if not isinstance(allowed, list) or not allowed:
            return None
        stop_atoms = [Atom(f"stop_reason:{r}") for r in allowed]
        return Finally(disj(*stop_atoms))

    if kind == "forbidden_text":
        text = params.get("text")
        if not isinstance(text, str):
            return None
        return Globally(Not(Atom(f"text_contains:{text}")))

    if kind == "must_include_text":
        text = params.get("text")
        if not isinstance(text, str):
            return None
        return Finally(Atom(f"text_contains:{text}"))

    if kind == "ltl_formula":
        raw = params.get("formula")
        if not isinstance(raw, str):
            return None
        return parse_ltl(raw)

    return None


def parse_ltl(expr: str) -> Formula:
    """Parse an LTL formula string into an AST.

    Supported syntax:

        atoms         ::= pred:value  |  true  |  false
        unary         ::= ! φ  |  G φ  |  F φ  |  X φ
        binary        ::= φ U ψ  |  φ W ψ  |  φ & ψ  |  φ | ψ  |  φ -> ψ
        grouping      ::= ( φ )

    Operator precedence (highest to lowest):
        1.  unary: ! G F X
        2.  U  W
        3.  &
        4.  |
        5.  ->

    Raises ``ValueError`` on parse errors.
    """
    tokens = _tokenize(expr)
    parser = _Parser(tokens)
    formula = parser.parse_implies()
    if not parser.at_end():
        raise ValueError(f"unexpected token {parser.peek()!r} in LTL expression {expr!r}")
    return formula


# ---- tokenizer ---------------------------------------------------------------

_KEYWORDS = {"G", "F", "X", "U", "W", "true", "false"}
_SYMBOLS = {"!", "&", "|", "->", "(", ")"}


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
            continue
        if expr[i:i+2] == "->":
            tokens.append("->")
            i += 2
            continue
        if expr[i] in "!&|()":
            tokens.append(expr[i])
            i += 1
            continue
        # Keyword or atom: read until whitespace or symbol.
        j = i
        while j < len(expr) and not expr[j].isspace() and expr[j] not in "!&|()":
            if expr[j:j+2] == "->":
                break
            j += 1
        tok = expr[i:j]
        if not tok:
            raise ValueError(f"unexpected character {expr[i]!r} in LTL expression {expr!r}")
        tokens.append(tok)
        i = j
    return tokens


# ---- recursive descent parser ------------------------------------------------

class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def at_end(self) -> bool:
        return self._pos >= len(self._tokens)

    def peek(self) -> str | None:
        if self.at_end():
            return None
        return self._tokens[self._pos]

    def consume(self) -> str:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.consume()
        if got != tok:
            raise ValueError(f"expected {tok!r}, got {got!r}")

    # Precedence levels (lowest first in parse methods):
    #   implies > or > and > until > unary

    def parse_implies(self) -> Formula:
        left = self.parse_or()
        if self.peek() == "->":
            self.consume()
            right = self.parse_implies()  # right-associative
            return Implies(left, right)
        return left

    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.peek() == "|":
            self.consume()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Formula:
        left = self.parse_until()
        while self.peek() == "&":
            self.consume()
            right = self.parse_until()
            left = And(left, right)
        return left

    def parse_until(self) -> Formula:
        left = self.parse_unary()
        tok = self.peek()
        if tok == "U":
            self.consume()
            right = self.parse_unary()
            return Until(left, right)
        if tok == "W":
            self.consume()
            right = self.parse_unary()
            return WeakUntil(left, right)
        return left

    def parse_unary(self) -> Formula:
        tok = self.peek()
        if tok == "!":
            self.consume()
            return Not(self.parse_unary())
        if tok == "G":
            self.consume()
            return Globally(self.parse_unary())
        if tok == "F":
            self.consume()
            return Finally(self.parse_unary())
        if tok == "X":
            self.consume()
            return Next(self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        tok = self.peek()
        if tok == "(":
            self.consume()
            inner = self.parse_implies()
            self.expect(")")
            return inner
        if tok is None:
            raise ValueError("unexpected end of LTL expression")
        self.consume()
        if tok == "true":
            return Atom("true")
        if tok == "false":
            return Atom("false")
        # Validate atom form: must be non-empty string.
        if not tok:
            raise ValueError("empty atom in LTL expression")
        return Atom(tok)


__all__ = ["parse_ltl", "rule_to_ltl"]
