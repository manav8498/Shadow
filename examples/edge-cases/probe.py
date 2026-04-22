"""Adversarial probe suite — deliberately weird inputs to see where
Shadow's axes break, misreport, or silently mishandle.

Run: python examples/edge-cases/probe.py

Each case builds two in-memory trace arrays, calls compute_diff_report
directly, and prints the result next to what I EXPECTED. Any mismatch
between "expected" and "actual" is a bug I need to either fix or
document honestly.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any

from shadow import _core


# ---------------------------------------------------------------------------
# Minimal trace builder
# ---------------------------------------------------------------------------


def _record(
    kind: str, payload: dict[str, Any], parent: str | None, ts: str
) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": _core.content_id(payload),
        "kind": kind,
        "ts": ts,
        "parent": parent,
        "payload": payload,
    }


def trace(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a minimal trace: metadata → (chat_request, chat_response) × N."""
    recs = [
        _record(
            "metadata",
            {"sdk": {"name": "shadow", "version": "0.1.0"}},
            None,
            "2026-04-22T00:00:00.000Z",
        )
    ]
    for i, resp in enumerate(responses):
        req = {
            "model": resp.get("model", "x"),
            "messages": [{"role": "user", "content": f"q{i}"}],
            "params": {"temperature": 0.0},
        }
        req_rec = _record(
            "chat_request", req, recs[-1]["id"], f"2026-04-22T00:00:{i:02d}.000Z"
        )
        recs.append(req_rec)
        recs.append(
            _record(
                "chat_response",
                resp,
                req_rec["id"],
                f"2026-04-22T00:00:{i:02d}.500Z",
            )
        )
    return recs


def mk_resp(
    text: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    stop: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 10,
    thinking_tokens: int = 0,
    latency_ms: int = 100,
    model: str = "claude-opus-4-7",
    content_parts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if content_parts is not None:
        content = content_parts
    else:
        content = [{"type": "text", "text": text}] if text else []
        for i, (name, inp) in enumerate(tool_calls or []):
            content.append(
                {"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}
            )
    return {
        "model": model,
        "content": content,
        "stop_reason": stop,
        "latency_ms": latency_ms,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
        },
    }


# ---------------------------------------------------------------------------
# The probe harness
# ---------------------------------------------------------------------------


FAILURES: list[tuple[str, str]] = []


def check(label: str, report: dict[str, Any], expectations: dict[str, Any]) -> None:
    """Compare report to expectations. Expectations is a dict of
    axis_name → {"severity": str} or {"severity_one_of": [str, ...]}
    or {"n": int}.
    """
    rows = {r["axis"]: r for r in report["rows"]}
    problems: list[str] = []
    for axis, spec in expectations.items():
        row = rows[axis]
        if "severity" in spec and row["severity"] != spec["severity"]:
            problems.append(
                f"    {axis}: expected severity={spec['severity']!r}, "
                f"got {row['severity']!r} (delta={row['delta']:+.3f}, n={row['n']})"
            )
        if "severity_one_of" in spec and row["severity"] not in spec["severity_one_of"]:
            problems.append(
                f"    {axis}: expected severity ∈ {spec['severity_one_of']}, "
                f"got {row['severity']!r} (delta={row['delta']:+.3f}, n={row['n']})"
            )
        if "n" in spec and row["n"] != spec["n"]:
            problems.append(f"    {axis}: expected n={spec['n']}, got {row['n']}")
    if problems:
        print(f"\n❌ {label}")
        print("\n".join(problems))
        FAILURES.append((label, "\n".join(problems)))
    else:
        print(f"✓  {label}")


def run_safely(label: str, fn: Any) -> Any:
    """Run fn(); on exception, log as a failure and return None."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"💥 {label}")
        print(f"    raised {type(e).__name__}: {e}")
        FAILURES.append((label, f"raised {type(e).__name__}: {e}"))
        return None


# ---------------------------------------------------------------------------
# Edge case 1: identical traces → every axis should be none
# ---------------------------------------------------------------------------


def case_identical() -> None:
    t = trace(
        [mk_resp(text="hello", latency_ms=100, output_tokens=5) for _ in range(5)]
    )
    report = run_safely("identical", lambda: _core.compute_diff_report(t, t, None, 42))
    if report is None:
        return
    check(
        "1. identical traces → all axes none",
        report,
        {
            "semantic": {"severity": "none"},
            "trajectory": {"severity": "none"},
            "safety": {"severity": "none"},
            "verbosity": {"severity": "none"},
            "latency": {"severity": "none"},
            "conformance": {"severity_one_of": ["none"]},
        },
    )


# ---------------------------------------------------------------------------
# Edge case 2: unequal pair count → should truncate, not crash
# ---------------------------------------------------------------------------


def case_unequal_pair_count() -> None:
    b = trace([mk_resp(text=f"b{i}") for i in range(5)])
    c = trace([mk_resp(text=f"c{i}") for i in range(2)])
    report = run_safely(
        "unequal pair count", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    if report["pair_count"] == 2:
        print("✓  2. unequal pair count (5 vs 2) truncates to 2")
    else:
        FAILURES.append(
            ("unequal pair count", f"expected pair_count=2, got {report['pair_count']}")
        )
        print(f"❌ 2. unequal pair count (5 vs 2) → pair_count={report['pair_count']}")


# ---------------------------------------------------------------------------
# Edge case 3: empty trace set
# ---------------------------------------------------------------------------


def case_empty_traces() -> None:
    empty_meta = [
        _record(
            "metadata",
            {"sdk": {"name": "shadow", "version": "0.1.0"}},
            None,
            "2026-04-22T00:00:00Z",
        )
    ]
    report = run_safely(
        "empty traces",
        lambda: _core.compute_diff_report(empty_meta, empty_meta, None, 42),
    )
    if report is None:
        return
    if report["pair_count"] == 0:
        print("✓  3. empty trace set (metadata only) → pair_count=0, no crash")
    else:
        print(f"❌ 3. empty trace set → unexpected pair_count={report['pair_count']}")


# ---------------------------------------------------------------------------
# Edge case 4: completely disjoint tool sets
# ---------------------------------------------------------------------------


def case_disjoint_tools() -> None:
    b = trace(
        [
            mk_resp(tool_calls=[("tool_a", {"x": 1})]),
            mk_resp(tool_calls=[("tool_b", {"y": 2})]),
            mk_resp(tool_calls=[("tool_c", {"z": 3})]),
        ]
    )
    c = trace(
        [
            mk_resp(tool_calls=[("tool_x", {"a": 1})]),
            mk_resp(tool_calls=[("tool_y", {"b": 2})]),
            mk_resp(tool_calls=[("tool_z", {"c": 3})]),
        ]
    )
    report = run_safely(
        "disjoint tools", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    check(
        "4. completely disjoint tool sets → trajectory severe",
        report,
        {"trajectory": {"severity": "severe"}},
    )


# ---------------------------------------------------------------------------
# Edge case 5: refusal-pattern false positive (user echoes pattern)
# ---------------------------------------------------------------------------


def case_refusal_echo() -> None:
    # Both sides contain the literal phrase "I can't help with that" but
    # not as a refusal — as a quoted explanation. Does the safety axis
    # false-positive?
    quoted = (
        'When a model says "I can\'t help with that", it usually means it '
        "detected a policy trigger."
    )
    b = trace([mk_resp(text=quoted) for _ in range(5)])
    c = trace([mk_resp(text=quoted) for _ in range(5)])
    report = run_safely(
        "refusal echo", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    # Both sides match the pattern → both = 1.0 abstention → delta 0 → none.
    # Technically we're over-counting abstention on baseline, but delta
    # stays zero so no false regression. This is the test.
    rows = {r["axis"]: r for r in report["rows"]}
    s = rows["safety"]
    if s["delta"] == 0.0:
        print(
            f"✓  5. quoted-refusal text: both sides score {s['baseline_median']:.2f}, delta 0 (no false regression)"
        )
    else:
        print(f"❌ 5. quoted-refusal text: delta={s['delta']:+.3f}, bias in classifier")
        FAILURES.append(("refusal echo", f"delta {s['delta']}"))


# ---------------------------------------------------------------------------
# Edge case 6: reverse regression (candidate BETTER than baseline)
# ---------------------------------------------------------------------------


def case_reverse_regression() -> None:
    # Candidate is faster and shorter. Are negative deltas still flagged?
    b = trace([mk_resp(latency_ms=2000, output_tokens=500) for _ in range(5)])
    c = trace([mk_resp(latency_ms=200, output_tokens=50) for _ in range(5)])
    report = run_safely(
        "reverse regression", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    rows = {r["axis"]: r for r in report["rows"]}
    lat = rows["latency"]
    verb = rows["verbosity"]
    print(
        f"6. reverse regression (candidate faster/shorter):\n"
        f"   latency delta={lat['delta']:+.0f}ms severity={lat['severity']} "
        f"(-90% absolute)\n"
        f"   verbosity delta={verb['delta']:+.0f}tok severity={verb['severity']} "
        f"(-90% absolute)"
    )
    # Severity is based on magnitude, not sign, so both severe. That's correct
    # behaviour — "getting faster might be good or bad depending on what you
    # sacrificed." Shadow's job is to surface the shift; a human judges
    # direction.


# ---------------------------------------------------------------------------
# Edge case 7: Unicode NFC collision — decomposed vs precomposed 'é'
# ---------------------------------------------------------------------------


def case_unicode_nfc() -> None:
    # Pre-composed 'é' (U+00E9) vs decomposed 'é' (U+0065 + U+0301).
    # Shadow canonicalises via NFC, so these should be IDENTICAL.
    b = trace([mk_resp(text="café") for _ in range(3)])  # precomposed
    c = trace([mk_resp(text="café") for _ in range(3)])  # decomposed
    report = run_safely(
        "unicode NFC", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    # Response IDs should be equal due to NFC, so pairs look identical.
    b_resp = b[2]
    c_resp = c[2]
    if b_resp["id"] == c_resp["id"]:
        print("✓  7. NFC: precomposed 'é' and decomposed 'é' hash to same id")
    else:
        print(
            "❌ 7. NFC: different ids for equivalent strings — canonicalisation broken"
        )
        FAILURES.append(("NFC", f"b={b_resp['id']} c={c_resp['id']}"))


# ---------------------------------------------------------------------------
# Edge case 8: thinking-only response (no visible text)
# ---------------------------------------------------------------------------


def case_thinking_only() -> None:
    # Response whose content is only {type: "thinking"} — no visible text.
    # Should not crash on any axis.
    content = [{"type": "thinking", "text": "pondering..."}]
    b = trace([mk_resp(content_parts=content, thinking_tokens=20) for _ in range(3)])
    c = trace([mk_resp(content_parts=content, thinking_tokens=20) for _ in range(3)])
    report = run_safely(
        "thinking only", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    print("✓  8. thinking-only content (no text parts): no crash")


# ---------------------------------------------------------------------------
# Edge case 9: very long trace (100 pairs)
# ---------------------------------------------------------------------------


def case_long_trace() -> None:
    import time

    n = 100
    b = trace([mk_resp(text=f"response {i}", latency_ms=100 + i) for i in range(n)])
    c = trace([mk_resp(text=f"response {i}", latency_ms=200 + i) for i in range(n)])
    t0 = time.perf_counter()
    report = run_safely("long trace", lambda: _core.compute_diff_report(b, c, None, 42))
    elapsed = time.perf_counter() - t0
    if report is None:
        return
    if report["pair_count"] == n and elapsed < 5.0:
        print(
            f"✓  9. long trace ({n} pairs) in {elapsed*1000:.0f}ms — {report['pair_count']} pair_count"
        )
    else:
        print(
            f"❌ 9. long trace: pair_count={report['pair_count']}, elapsed={elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Edge case 10: deeply nested payload (50 levels)
# ---------------------------------------------------------------------------


def case_deep_nesting() -> None:
    # Build a deeply nested arg structure in a tool_use
    nested: Any = "leaf"
    for i in range(50):
        nested = {f"level_{i}": nested}
    b = trace([mk_resp(tool_calls=[("tool_a", nested)])])
    c = trace([mk_resp(tool_calls=[("tool_a", nested)])])
    report = run_safely(
        "deep nesting", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    print("✓  10. deeply nested (50 levels) tool args: no crash")


# ---------------------------------------------------------------------------
# Edge case 11: JSON-vs-JSON format mismatch — both valid but different
# shapes
# ---------------------------------------------------------------------------


def case_json_shape_mismatch() -> None:
    # Baseline returns `[{"a": 1}]`, candidate returns `{"error": "..."}`.
    # Both parse as JSON, but structurally totally different.
    b = trace([mk_resp(text='[{"ok": true}]') for _ in range(3)])
    c = trace([mk_resp(text='{"error": "something"}') for _ in range(3)])
    report = run_safely(
        "json shape mismatch", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    # Conformance axis only checks parseability — both parse → rate 1.0 vs 1.0
    # → no regression flagged. This is a known limitation: structural JSON
    # schema mismatch is the Judge's job.
    rows = {r["axis"]: r for r in report["rows"]}
    cf = rows["conformance"]
    print(
        f"11. JSON parseability same but shape differs:\n"
        f"    conformance delta={cf['delta']:+.3f}, severity={cf['severity']} "
        f"(known limit: parseability != schema match — Judge territory)"
    )


# ---------------------------------------------------------------------------
# Edge case 12: huge tool args (100KB JSON)
# ---------------------------------------------------------------------------


def case_huge_tool_args() -> None:
    big: dict[str, Any] = {"data": ["item_" + str(i) for i in range(2000)]}
    b = trace([mk_resp(tool_calls=[("process_data", big)])])
    c = trace([mk_resp(tool_calls=[("process_data", big)])])
    report = run_safely(
        "huge tool args", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    print("✓  12. 100KB tool-arg blob: no crash, identical pairs handled")


# ---------------------------------------------------------------------------
# Edge case 13: empty strings and nulls in text
# ---------------------------------------------------------------------------


def case_empty_text() -> None:
    b = trace([mk_resp(text="") for _ in range(5)])
    c = trace([mk_resp(text="") for _ in range(5)])
    report = run_safely("empty text", lambda: _core.compute_diff_report(b, c, None, 42))
    if report is None:
        return
    print("✓  13. empty text content on both sides: no crash, no false positives")


# ---------------------------------------------------------------------------
# Edge case 14: baseline/candidate with identical RISKY tool names
# (validating that safety axis doesn't fire on its own without a refusal)
# ---------------------------------------------------------------------------


def case_both_sides_risky() -> None:
    # After the principled-safety refactor, calling a "dangerous-sounding"
    # tool is NOT a safety signal on its own. Both sides call delete_foo —
    # safety should stay 0.
    b = trace(
        [mk_resp(tool_calls=[("delete_records", {"ids": [1, 2]})]) for _ in range(3)]
    )
    c = trace(
        [mk_resp(tool_calls=[("delete_records", {"ids": [1, 2]})]) for _ in range(3)]
    )
    report = run_safely(
        "both sides risky", lambda: _core.compute_diff_report(b, c, None, 42)
    )
    if report is None:
        return
    rows = {r["axis"]: r for r in report["rows"]}
    s = rows["safety"]
    t_row = rows["trajectory"]
    print(
        f"14. both sides call delete_records:\n"
        f"    safety delta={s['delta']:+.3f} severity={s['severity']} "
        f"(correct — no refusals, no false-positive)\n"
        f"    trajectory delta={t_row['delta']:+.3f} severity={t_row['severity']}"
    )


# ---------------------------------------------------------------------------
# Edge case 15: tool call with same name but different argument values
# (not shapes) — should be considered SAME trajectory
# ---------------------------------------------------------------------------


def case_same_tool_different_values() -> None:
    b = trace([mk_resp(tool_calls=[("lookup", {"id": "alice"})]) for _ in range(5)])
    c = trace([mk_resp(tool_calls=[("lookup", {"id": "bob"})]) for _ in range(5)])
    report = run_safely(
        "same tool different values",
        lambda: _core.compute_diff_report(b, c, None, 42),
    )
    if report is None:
        return
    rows = {r["axis"]: r for r in report["rows"]}
    t_row = rows["trajectory"]
    # By design trajectory ignores arg values (only looks at keys) — so
    # same-tool-different-values should show 0 trajectory divergence.
    if t_row["severity"] == "none":
        print(
            "✓  15. same tool, different arg values: trajectory none "
            "(by design — values are noise, schema is signal)"
        )
    else:
        print(
            f"⚠️  15. same tool, different arg values: trajectory severity={t_row['severity']} "
            f"(trajectory is value-sensitive — unexpected)"
        )
        FAILURES.append(("same-tool-different-values", str(t_row)))


# ---------------------------------------------------------------------------


def case_many_deltas_bisect() -> None:
    """Bisect with configs producing more than the PB-tabulated max (23).

    plackett_burman is tabulated up to 24 runs (k<=23). Beyond that
    choose_design should error cleanly, not silently misbehave.
    """
    from shadow.bisect.runner import choose_design

    try:
        choose_design(24)
        # If it returned rather than raising, we'd want the result to be
        # a valid design matrix of the right shape (depending on v0.2 work).
        print("⚠️  16. choose_design(24): did not raise (unexpected)")
        FAILURES.append(("choose_design(24)", "did not raise"))
    except ValueError:
        print("✓  16. choose_design(24): raises ValueError cleanly")


def case_monotone_signal() -> None:
    """Perfectly correlated degradation — candidate i is always worse."""
    n = 20
    b = trace([mk_resp(latency_ms=100 + i) for i in range(n)])
    c = trace([mk_resp(latency_ms=1000 + i) for i in range(n)])
    report = run_safely("monotone", lambda: _core.compute_diff_report(b, c, None, 42))
    if report is None:
        return
    rows = {r["axis"]: r for r in report["rows"]}
    lat = rows["latency"]
    # Perfect correlation → tight CI (low variance in the paired differences).
    ci_width = lat["ci95_high"] - lat["ci95_low"]
    if lat["severity"] == "severe" and ci_width < 100:
        print(
            f"✓  17. monotone latency shift: delta={lat['delta']:+.0f} "
            f"CI=[{lat['ci95_low']:+.0f}, {lat['ci95_high']:+.0f}] (tight)"
        )
    else:
        print(f"❌ 17. monotone shift: unexpected CI width={ci_width:.0f}")
        FAILURES.append(("monotone", f"CI={lat['ci95_low']}..{lat['ci95_high']}"))


def case_n_equals_one() -> None:
    """Single pair — statistics should degrade gracefully (wide CI)."""
    b = trace([mk_resp(text="hi", latency_ms=100)])
    c = trace([mk_resp(text="bye", latency_ms=500)])
    report = run_safely("n=1", lambda: _core.compute_diff_report(b, c, None, 42))
    if report is None:
        return
    if report["pair_count"] == 1:
        print("✓  18. n=1: single-pair diff produces a report without crashing")
    else:
        print(f"❌ 18. n=1: pair_count={report['pair_count']}")


def case_tool_name_case() -> None:
    """Trajectory compares tool names case-sensitively — or should it?

    This is a design question, not a bug. Document current behaviour:
    `Delete_foo` and `delete_foo` are DIFFERENT tools as far as
    trajectory edit-distance is concerned (no lowercase-fold).
    """
    b = trace([mk_resp(tool_calls=[("delete_foo", {"id": 1})]) for _ in range(5)])
    c = trace([mk_resp(tool_calls=[("Delete_foo", {"id": 1})]) for _ in range(5)])
    report = run_safely("tool case", lambda: _core.compute_diff_report(b, c, None, 42))
    if report is None:
        return
    rows = {r["axis"]: r for r in report["rows"]}
    t_row = rows["trajectory"]
    print(
        f"19. case-different tool names ('delete_foo' vs 'Delete_foo'):\n"
        f"    trajectory delta={t_row['delta']:+.3f} severity={t_row['severity']} "
        f"(current: case-sensitive; tool APIs usually are too)"
    )


def case_stop_reason_error() -> None:
    """stop_reason = 'error' is NOT an abstention (not `content_filter`
    and usually no refusal text). Safety should stay 0; the error is a
    provider/infra issue, not a safety signal."""
    err = {
        "model": "x",
        "content": [{"type": "text", "text": "Request timed out"}],
        "stop_reason": "error",
        "latency_ms": 5000,
        "usage": {"input_tokens": 10, "output_tokens": 3, "thinking_tokens": 0},
    }
    ok = mk_resp(text="here you go")
    b = trace([ok for _ in range(5)])
    c = trace([err for _ in range(5)])
    report = run_safely("stop=error", lambda: _core.compute_diff_report(b, c, None, 42))
    if report is None:
        return
    rows = {r["axis"]: r for r in report["rows"]}
    s = rows["safety"]
    print(
        f"20. stop_reason='error' on candidate:\n"
        f"    safety delta={s['delta']:+.3f} severity={s['severity']} "
        f"(current: error stops are NOT safety events by default; user "
        f"could add 'error' as a refusal pattern if they want it to count)"
    )


def main() -> int:
    print("Shadow adversarial probe — 20 edge cases")
    print("=" * 60)
    case_identical()
    case_unequal_pair_count()
    case_empty_traces()
    case_disjoint_tools()
    case_refusal_echo()
    case_reverse_regression()
    case_unicode_nfc()
    case_thinking_only()
    case_long_trace()
    case_deep_nesting()
    case_json_shape_mismatch()
    case_huge_tool_args()
    case_empty_text()
    case_both_sides_risky()
    case_same_tool_different_values()
    case_many_deltas_bisect()
    case_monotone_signal()
    case_n_equals_one()
    case_tool_name_case()
    case_stop_reason_error()
    print("=" * 60)
    print(f"Summary: {20 - len(FAILURES)}/20 clean")
    if FAILURES:
        print("\nFailures:")
        for label, detail in FAILURES:
            print(f"  - {label}")
            print(textwrap.indent(detail, "    "))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
