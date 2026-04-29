"""Property-based parity test: TS LTLf evaluator must agree with the
Python evaluator on every random LTLf formula x trace combination.

The basic parity test in ``test_typescript_parity.py`` covers a fixed
set of fixtures. This suite generates random LTLf formulas (depth
1-6, all 10 operators) and random traces, and asserts the two
implementations return identical truth-vectors at every position
within float tolerance (truth values are booleans so tolerance is
exact).

Run conditions: skips automatically when node is not on PATH or the
TS dist isn't built. Run via:

    npm --prefix typescript run build
    pytest python/tests/test_typescript_parity_property.py -v

Number of random trials: ``N_TRIALS = 500``. At depth-6 formulas
x length-12 traces this exercises ~6000 (formula, position)
evaluation points per run.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from shadow.ltl.checker import (
    TraceState,
    eval_all_positions,
)
from shadow.ltl.formula import (
    And,
    Atom,
    Finally,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    Until,
    WeakUntil,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_DIR = REPO_ROOT / "typescript"
EVAL_CLI = TS_DIR / "scripts" / "eval-ltlf.mjs"
GATE_DIST = TS_DIR / "dist" / "gate" / "index.js"


def _node_available() -> bool:
    return shutil.which("node") is not None and GATE_DIST.exists()


pytestmark = pytest.mark.skipif(
    not _node_available(),
    reason="node not available or TS not built",
)


# ---------------------------------------------------------------------------
# Random LTLf formula generator
# ---------------------------------------------------------------------------

# Atomic predicates that the default_eval recogniser supports. Mix of
# tool_call, stop_reason, and the literal true/false so the random
# truth-vectors aren't trivially all-true or all-false.
_TOOLS = ["a", "b", "c", "delete_user", "verify"]
_STOPS = ["end_turn", "tool_use", "content_filter"]


def _atoms() -> list[str]:
    return (
        ["true", "false"]
        + [f"tool_call:{t}" for t in _TOOLS]
        + [f"stop_reason:{s}" for s in _STOPS]
    )


def _random_formula(rng: random.Random, max_depth: int) -> Any:
    """Return a random LTLf formula AST (Python objects). Depth-bounded
    so terms terminate; depth=0 forces an atom."""
    if max_depth <= 0:
        return Atom(rng.choice(_atoms()))
    op = rng.randrange(11)
    if op == 0:
        return Atom(rng.choice(_atoms()))
    if op == 1:
        return Not(_random_formula(rng, max_depth - 1))
    if op == 2:
        return And(_random_formula(rng, max_depth - 1), _random_formula(rng, max_depth - 1))
    if op == 3:
        return Or(_random_formula(rng, max_depth - 1), _random_formula(rng, max_depth - 1))
    if op == 4:
        return Implies(_random_formula(rng, max_depth - 1), _random_formula(rng, max_depth - 1))
    if op == 5:
        return Next(_random_formula(rng, max_depth - 1))
    if op == 6:
        return Globally(_random_formula(rng, max_depth - 1))
    if op == 7:
        return Finally(_random_formula(rng, max_depth - 1))
    if op == 8:
        return Until(_random_formula(rng, max_depth - 1), _random_formula(rng, max_depth - 1))
    if op == 9:
        return WeakUntil(_random_formula(rng, max_depth - 1), _random_formula(rng, max_depth - 1))
    return Atom(rng.choice(_atoms()))


# ---------------------------------------------------------------------------
# Random trace generator
# ---------------------------------------------------------------------------


def _random_state(rng: random.Random, idx: int) -> TraceState:
    n_tools = rng.randrange(0, 3)
    tool_calls = [rng.choice(_TOOLS) for _ in range(n_tools)]
    return TraceState(
        pair_index=idx,
        tool_calls=tool_calls,
        stop_reason=rng.choice([*_STOPS, ""]),
        text_content="",
        extra={},
    )


def _random_trace(rng: random.Random, max_len: int) -> list[TraceState]:
    n = rng.randrange(0, max_len + 1)
    return [_random_state(rng, i) for i in range(n)]


# ---------------------------------------------------------------------------
# AST → JSON shape (matches the TS Formula union)
# ---------------------------------------------------------------------------


def _formula_to_json(f: Any) -> dict[str, Any]:
    if isinstance(f, Atom):
        return {"kind": "atom", "pred": f.pred}
    if isinstance(f, Not):
        return {"kind": "not", "child": _formula_to_json(f.child)}
    if isinstance(f, And):
        return {
            "kind": "and",
            "left": _formula_to_json(f.left),
            "right": _formula_to_json(f.right),
        }
    if isinstance(f, Or):
        return {
            "kind": "or",
            "left": _formula_to_json(f.left),
            "right": _formula_to_json(f.right),
        }
    if isinstance(f, Implies):
        return {
            "kind": "implies",
            "left": _formula_to_json(f.left),
            "right": _formula_to_json(f.right),
        }
    if isinstance(f, Next):
        return {"kind": "next", "child": _formula_to_json(f.child)}
    if isinstance(f, Globally):
        return {"kind": "globally", "child": _formula_to_json(f.child)}
    if isinstance(f, Finally):
        return {"kind": "finally", "child": _formula_to_json(f.child)}
    if isinstance(f, Until):
        return {
            "kind": "until",
            "left": _formula_to_json(f.left),
            "right": _formula_to_json(f.right),
        }
    if isinstance(f, WeakUntil):
        return {
            "kind": "weakUntil",
            "left": _formula_to_json(f.left),
            "right": _formula_to_json(f.right),
        }
    raise TypeError(f"unhandled formula node: {type(f).__name__}")


def _trace_to_json(trace: list[TraceState]) -> list[dict[str, Any]]:
    return [
        {
            "pairIndex": s.pair_index,
            "toolCalls": s.tool_calls,
            "stopReason": s.stop_reason,
            "textContent": s.text_content,
            "extra": s.extra,
        }
        for s in trace
    ]


# ---------------------------------------------------------------------------
# Node CLI helper — calls into the TS evaluator on JSON input
# ---------------------------------------------------------------------------


def _run_ts_eval(formula_json: dict[str, Any], trace_json: list[dict[str, Any]]) -> list[bool]:
    payload = json.dumps({"formula": formula_json, "trace": trace_json})
    proc = subprocess.run(
        ["node", str(EVAL_CLI)],
        input=payload.encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=30,
    )
    return list(json.loads(proc.stdout.decode("utf-8")))


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


N_TRIALS = 500
MAX_FORMULA_DEPTH = 6
MAX_TRACE_LEN = 12


class TestPropertyBasedTsPythonParity:
    def test_random_formula_truth_vectors_match(self) -> None:
        rng = random.Random(20260428)
        mismatches: list[str] = []
        for trial in range(N_TRIALS):
            formula = _random_formula(rng, MAX_FORMULA_DEPTH)
            trace = _random_trace(rng, MAX_TRACE_LEN)
            py_arr = eval_all_positions(formula, trace)
            ts_arr = _run_ts_eval(_formula_to_json(formula), _trace_to_json(trace))
            if list(py_arr) != ts_arr:
                mismatches.append(
                    f"trial={trial} formula={formula} trace_len={len(trace)} "
                    f"py={list(py_arr)} ts={ts_arr}"
                )
                if len(mismatches) >= 3:
                    break
        assert not mismatches, (
            f"TS and Python LTLf evaluators disagree on {len(mismatches)} "
            f"random fixtures:\n" + "\n".join(mismatches)
        )

    def test_long_trace_smoke(self) -> None:
        """Performance smoke: the TS evaluator must handle a 1000-state
        trace without timeout. We don't assert a tight latency bound
        (TS startup cost dominates), just that the call returns and
        agrees with Python."""
        rng = random.Random(42)
        formula = Globally(Implies(Atom("tool_call:a"), Finally(Atom("stop_reason:end_turn"))))
        trace = [_random_state(rng, i) for i in range(1000)]
        py_arr = eval_all_positions(formula, trace)
        ts_arr = _run_ts_eval(_formula_to_json(formula), _trace_to_json(trace))
        assert list(py_arr) == ts_arr
