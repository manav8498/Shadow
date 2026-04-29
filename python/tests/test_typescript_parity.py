"""Cross-language conformance: TS gate must match Python decisions.

For each fixture we generate the SAME records, rules, and LTLf
formulas in both languages and assert that the gate decisions agree
on the `(rule_id, pair_index, kind)` violation tuples and the
per-formula pass/fail.

Detail strings (the human-readable explanation of WHY a rule fired)
are intentionally NOT compared — they are language-specific by
design. The gating decision is what matters; the detail is for
human-rendered diagnostics.

Skips automatically when:
  - `node` is not on PATH
  - `typescript/dist/gate/index.js` does not exist (TS not built)

Run `npm run build` inside `typescript/` before running this suite
locally; CI runs `npm run build` as a prerequisite step.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from shadow.ltl.checker import TraceState, eval_all_positions
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
GATE_CLI = TS_DIR / "scripts" / "run-gate.mjs"
EVAL_CLI = TS_DIR / "scripts" / "eval-ltlf.mjs"
GATE_DIST = TS_DIR / "dist" / "gate" / "index.js"


def _node_available() -> bool:
    return shutil.which("node") is not None and GATE_DIST.exists()


pytestmark = pytest.mark.skipif(
    not _node_available(),
    reason="node not available or TS not built (run `npm --prefix typescript run build`)",
)


def _run_ts_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Invoke the TypeScript gate via the Node helper script."""
    proc = subprocess.run(
        ["node", str(GATE_CLI)],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(proc.stdout.decode("utf-8"))


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _record(idx: int, *, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{idx:064x}",
        "kind": kind,
        "ts": "2026-04-28T00:00:00.000Z",
        "parent": "sha256:" + "0" * 64,
        "meta": {},
        "payload": payload,
    }


def _response(
    *,
    idx: int,
    tools: list[str] | None = None,
    text: str = "",
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for j, name in enumerate(tools or []):
        content.append({"type": "tool_use", "id": f"t{idx}_{j}", "name": name, "input": {}})
    return _record(
        idx,
        kind="chat_response",
        payload={
            "model": "x",
            "content": content,
            "stop_reason": stop_reason,
            "latency_ms": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
        },
    )


# ---------------------------------------------------------------------------
# Python-side reference implementation matching the TS rule subset
# ---------------------------------------------------------------------------


def _python_check_policy(
    records: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Direct Python port of the TS `checkPolicy` semantics.

    Implemented locally rather than calling `shadow.hierarchical` so
    the parity test exercises the precise decision surface we ported
    to TS — not the full Python rule engine which has a wider surface.
    """
    tool_calls: list[tuple[int, str]] = []
    responses: list[tuple[int, str]] = []
    pair_idx = 0
    for rec in records:
        if rec.get("kind") != "chat_response":
            continue
        payload = rec.get("payload") or {}
        content = payload.get("content") or []
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append((pair_idx, str(block.get("name") or "")))
            elif block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        responses.append((pair_idx, "\n".join(text_parts)))
        pair_idx += 1

    violations: list[dict[str, Any]] = []
    for rule in rules:
        kind = rule["kind"]
        rid = rule["ruleId"]
        if kind == "no_call":
            tool = rule["tool"]
            for p, t in tool_calls:
                if t == tool:
                    violations.append({"ruleId": rid, "kind": kind, "pairIndex": p})
        elif kind == "must_call_before":
            first, then = rule["first"], rule["then"]
            first_pos = next((i for i, (_, t) in enumerate(tool_calls) if t == first), -1)
            then_pos = next((i for i, (_, t) in enumerate(tool_calls) if t == then), -1)
            if then_pos == -1:
                continue
            if first_pos == -1 or first_pos > then_pos:
                violations.append(
                    {
                        "ruleId": rid,
                        "kind": kind,
                        "pairIndex": tool_calls[then_pos][0],
                    }
                )
        elif kind == "must_call_once":
            tool = rule["tool"]
            matches = [(p, t) for p, t in tool_calls if t == tool]
            if len(matches) > 1:
                for p, _ in matches[1:]:
                    violations.append({"ruleId": rid, "kind": kind, "pairIndex": p})
        elif kind == "forbidden_text":
            substr = rule["substring"]
            for p, txt in responses:
                if substr in txt:
                    violations.append({"ruleId": rid, "kind": kind, "pairIndex": p})
        elif kind == "must_include_text":
            substr = rule["substring"]
            if not any(substr in txt for _, txt in responses):
                violations.append({"ruleId": rid, "kind": kind, "pairIndex": None})

    violations.sort(
        key=lambda v: (
            v["pairIndex"] if v["pairIndex"] is not None else 9999999,
            v["ruleId"],
        )
    )
    return violations


# ---------------------------------------------------------------------------
# Parity test cases
# ---------------------------------------------------------------------------


class TestPolicyParity:
    def test_no_call_rule_matches(self) -> None:
        records = [
            _response(idx=1, tools=["safe"]),
            _response(idx=2, tools=["delete_user"]),
        ]
        rules = [{"kind": "no_call", "ruleId": "r1", "tool": "delete_user"}]
        ts = _run_ts_gate({"records": records, "rules": rules})
        py = _python_check_policy(records, rules)
        assert ts["violations"] == py
        assert ts["passed"] is False

    def test_must_call_before_passes(self) -> None:
        records = [
            _response(idx=1, tools=["verify_user"]),
            _response(idx=2, tools=["issue_refund"]),
        ]
        rules = [
            {
                "kind": "must_call_before",
                "ruleId": "r1",
                "first": "verify_user",
                "then": "issue_refund",
            }
        ]
        ts = _run_ts_gate({"records": records, "rules": rules})
        py = _python_check_policy(records, rules)
        assert ts["violations"] == py
        assert ts["passed"] is True

    def test_must_call_before_fails_when_reversed(self) -> None:
        records = [
            _response(idx=1, tools=["issue_refund"]),
            _response(idx=2, tools=["verify_user"]),
        ]
        rules = [
            {
                "kind": "must_call_before",
                "ruleId": "r1",
                "first": "verify_user",
                "then": "issue_refund",
            }
        ]
        ts = _run_ts_gate({"records": records, "rules": rules})
        py = _python_check_policy(records, rules)
        assert ts["violations"] == py
        assert ts["passed"] is False

    def test_must_call_once(self) -> None:
        records = [
            _response(idx=1, tools=["lookup"]),
            _response(idx=2, tools=["lookup"]),
            _response(idx=3, tools=["lookup"]),
        ]
        rules = [{"kind": "must_call_once", "ruleId": "r", "tool": "lookup"}]
        ts = _run_ts_gate({"records": records, "rules": rules})
        py = _python_check_policy(records, rules)
        assert ts["violations"] == py

    def test_forbidden_and_must_include(self) -> None:
        records = [
            _response(idx=1, text="please consult a doctor"),
            _response(idx=2, text="here is your refund"),
        ]
        rules = [
            {"kind": "forbidden_text", "ruleId": "no-refund", "substring": "refund"},
            {"kind": "must_include_text", "ruleId": "must-consult", "substring": "consult"},
        ]
        ts = _run_ts_gate({"records": records, "rules": rules})
        py = _python_check_policy(records, rules)
        assert ts["violations"] == py


class TestLtlParity:
    """Cross-language LTLf parity. We render the formula on the TS side
    via the canonical formula encoding and assert pass/fail equality."""

    @staticmethod
    def _atom(pred: str) -> dict[str, Any]:
        return {"kind": "atom", "pred": pred}

    @staticmethod
    def _not(child: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "not", "child": child}

    @staticmethod
    def _globally(child: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "globally", "child": child}

    @staticmethod
    def _finally(child: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "finally", "child": child}

    def test_globally_no_delete_passes_clean_trace(self) -> None:
        records = [_response(idx=1, tools=["safe"])]
        formula = self._globally(self._not(self._atom("tool_call:delete_user")))
        ts = _run_ts_gate({"records": records, "ltlFormulas": [formula]})
        assert ts["ltlResults"][0]["passed"] is True
        assert ts["passed"] is True

    def test_globally_no_delete_fails_dirty_trace(self) -> None:
        records = [_response(idx=1, tools=["delete_user"])]
        formula = self._globally(self._not(self._atom("tool_call:delete_user")))
        ts = _run_ts_gate({"records": records, "ltlFormulas": [formula]})
        assert ts["ltlResults"][0]["passed"] is False
        assert ts["passed"] is False

    def test_finally_target(self) -> None:
        records = [
            _response(idx=1, tools=["x"]),
            _response(idx=2, tools=["target"]),
        ]
        formula = self._finally(self._atom("tool_call:target"))
        ts = _run_ts_gate({"records": records, "ltlFormulas": [formula]})
        assert ts["ltlResults"][0]["passed"] is True

    def test_python_ltl_evaluator_matches(self) -> None:
        """Sanity check: the canonical Python LTL checker produces the
        same decisions on these fixtures as the TS evaluator does."""
        from shadow.ltl import (
            Atom,
            Globally,
            Not,
        )
        from shadow.ltl.checker import check, trace_from_records

        records = [_response(idx=1, tools=["delete_user"])]
        py_formula = Globally(Not(Atom("tool_call:delete_user")))
        py_trace = trace_from_records(records)
        py_result = check(py_formula, py_trace, 0)

        ts_formula = self._globally(self._not(self._atom("tool_call:delete_user")))
        ts = _run_ts_gate({"records": records, "ltlFormulas": [ts_formula]})

        assert ts["ltlResults"][0]["passed"] == py_result


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
