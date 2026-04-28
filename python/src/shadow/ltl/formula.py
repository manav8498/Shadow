"""Linear Temporal Logic (LTL) formula AST.

Operators
---------
- ``Atom(pred)``      — atomic proposition, evaluated by a truth assignment
- ``Not(φ)``          — logical negation
- ``And(φ, ψ)``       — logical conjunction
- ``Or(φ, ψ)``        — logical disjunction
- ``Implies(φ, ψ)``   — logical implication (sugar: ¬φ ∨ ψ)
- ``Next(φ)``         — X φ: φ holds at the next state
- ``Until(φ, ψ)``     — φ U ψ: φ holds until ψ becomes true (strong: ψ must occur)
- ``WeakUntil(φ, ψ)`` — φ W ψ: φ holds until ψ, *or* φ holds forever (ψ optional)
- ``Globally(φ)``     — G φ = ¬(⊤ U ¬φ) = φ W ⊥: φ holds at every future state
- ``Finally(φ)``      — F φ = ⊤ U φ: φ holds at some future state

``Atom.pred`` is a string in one of the canonical forms the LTL
checker understands:

    "tool_call:<name>"          — a tool named <name> was called this turn
    "stop_reason:<value>"       — response stop_reason == <value>
    "text_contains:<substring>" — response text contains <substring>
    "true"                      — always true (⊤)
    "false"                     — always false (⊥)

Custom predicates can be injected by passing an ``EvalFn`` to
:func:`~shadow.ltl.checker.check`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    pred: str

    def __str__(self) -> str:
        return self.pred


@dataclass(frozen=True)
class Not:
    child: Formula

    def __str__(self) -> str:
        return f"¬({self.child})"


@dataclass(frozen=True)
class And:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class Next:
    child: Formula

    def __str__(self) -> str:
        return f"X({self.child})"


@dataclass(frozen=True)
class Until:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} U {self.right})"


@dataclass(frozen=True)
class WeakUntil:
    """Weak until: φ W ψ ≡ (φ U ψ) ∨ G(φ).

    Unlike strong Until, WeakUntil does not require ψ to ever hold —
    if φ holds forever, the formula is satisfied. Used to encode
    "B may only fire after A" without forcing A to actually fire.
    """

    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} W {self.right})"


@dataclass(frozen=True)
class Globally:
    child: Formula

    def __str__(self) -> str:
        return f"G({self.child})"


@dataclass(frozen=True)
class Finally:
    child: Formula

    def __str__(self) -> str:
        return f"F({self.child})"


Formula = Atom | Not | And | Or | Implies | Next | Until | WeakUntil | Globally | Finally


# Convenience constructors for common sub-formulas.


def atom(pred: str) -> Atom:
    return Atom(pred)


def g(phi: Formula) -> Globally:
    return Globally(phi)


def f(phi: Formula) -> Finally:
    return Finally(phi)


def x(phi: Formula) -> Next:
    return Next(phi)


def u(phi: Formula, psi: Formula) -> Until:
    return Until(phi, psi)


def w(phi: Formula, psi: Formula) -> WeakUntil:
    return WeakUntil(phi, psi)


def neg(phi: Formula) -> Not:
    return Not(phi)


def conj(*args: Formula) -> Formula:
    """Conjoin a sequence of formulas (left-associative)."""
    if not args:
        return Atom("true")
    result: Formula = args[0]
    for phi in args[1:]:
        result = And(result, phi)
    return result


def disj(*args: Formula) -> Formula:
    """Disjoin a sequence of formulas (left-associative)."""
    if not args:
        return Atom("false")
    result: Formula = args[0]
    for phi in args[1:]:
        result = Or(result, phi)
    return result


def implies(phi: Formula, psi: Formula) -> Implies:
    return Implies(phi, psi)


# Common shorthands.
TRUE: Atom = Atom("true")
FALSE: Atom = Atom("false")


__all__ = [
    "FALSE",
    "TRUE",
    "And",
    "Atom",
    "Finally",
    "Formula",
    "Globally",
    "Implies",
    "Next",
    "Not",
    "Or",
    "Until",
    "WeakUntil",
    "atom",
    "conj",
    "disj",
    "f",
    "g",
    "implies",
    "neg",
    "u",
    "w",
    "x",
]
