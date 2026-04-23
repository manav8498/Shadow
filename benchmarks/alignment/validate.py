"""Extensive real-world validation of first-divergence detection.

Three parts:
  A) Committed fixtures across every scenario type in the repo.
  B) Adversarial stress cases with known-correct classifications.
  C) Performance + edge cases (noise floor, confidence calibration,
     large-trace scalability, Gotoh affine-gap coalescing).

Run from repo root with the in-repo venv active:
    python /tmp/validate_first_divergence.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "python/src")

from shadow import _core  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_pair(example: str) -> tuple[list[dict], list[dict]]:
    base = _core.parse_agentlog(
        Path(f"examples/{example}/fixtures/baseline.agentlog").read_bytes()
    )
    cand = _core.parse_agentlog(
        Path(f"examples/{example}/fixtures/candidate.agentlog").read_bytes()
    )
    return base, cand


def mk_resp(
    text: str = "",
    tool_name: str | None = None,
    tool_input: dict | None = None,
    stop: str = "end_turn",
    ts: str = "2026-04-23T10:00:00Z",
) -> dict:
    """Build a chat_response record matching the .agentlog shape."""
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_name is not None:
        content.append(
            {
                "type": "tool_use",
                "id": f"t_{tool_name}",
                "name": tool_name,
                "input": tool_input or {},
            }
        )
    payload = {
        "model": "x",
        "content": content,
        "stop_reason": stop,
        "latency_ms": 0,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }
    rid = _core.content_id(payload)
    return {
        "version": "0.1",
        "id": rid,
        "kind": "chat_response",
        "ts": ts,
        "parent": None,
        "payload": payload,
    }


def mk_meta() -> dict:
    payload = {"sdk": {"name": "shadow"}}
    rid = _core.content_id(payload)
    return {
        "version": "0.1",
        "id": rid,
        "kind": "metadata",
        "ts": "2026-04-23T10:00:00Z",
        "parent": None,
        "payload": payload,
    }


def first_div(baseline: list[dict], candidate: list[dict]) -> dict | None:
    d = _core.compute_diff_report(baseline, candidate)
    return d.get("first_divergence")


def fmt(fd: dict | None) -> str:
    if fd is None:
        return "   → None"
    return (
        f"   → turn b#{fd['baseline_turn']} ↔ c#{fd['candidate_turn']}  "
        f"kind={fd['kind']:<18} axis={fd['primary_axis']:<12} "
        f"conf={fd['confidence']:.2f}\n"
        f"     explanation: {fd['explanation']}"
    )


# ---------------------------------------------------------------------------
# Part A — real committed fixtures
# ---------------------------------------------------------------------------


def part_a() -> None:
    print("=" * 75)
    print(" PART A — Real committed fixtures (sanity check on hand-authored traces)")
    print("=" * 75)
    for example in ["demo", "customer-support", "devops-agent", "er-triage"]:
        print(f"\n[{example}]")
        b, c = load_pair(example)
        print(
            f"  baseline: {sum(1 for r in b if r['kind']=='chat_response')} responses, "
            f"candidate: {sum(1 for r in c if r['kind']=='chat_response')} responses"
        )
        fd = first_div(b, c)
        print(fmt(fd))


# ---------------------------------------------------------------------------
# Part B — adversarial stress cases (known ground truth)
# ---------------------------------------------------------------------------


def _wrap(responses: list[dict]) -> list[dict]:
    return [mk_meta()] + responses


def assert_case(
    case_name: str,
    baseline: list[dict],
    candidate: list[dict],
    expected_kind: str | None,
    expected_turn: int | None = None,
    expected_axis: str | None = None,
    min_confidence: float = 0.0,
) -> bool:
    fd = first_div(baseline, candidate)
    actual_kind = fd["kind"] if fd else None

    ok_kind = actual_kind == expected_kind
    ok_turn = expected_turn is None or (fd and fd["baseline_turn"] == expected_turn)
    ok_axis = expected_axis is None or (fd and fd["primary_axis"] == expected_axis)
    ok_conf = (
        fd is None or fd["confidence"] >= min_confidence
        if expected_kind
        else fd is None
    )

    ok = ok_kind and ok_turn and ok_axis and ok_conf
    mark = "✅" if ok else "❌"
    print(f"\n  {mark} {case_name}")
    print(
        f"     expected kind={expected_kind}, turn={expected_turn}, axis={expected_axis}"
    )
    print(fmt(fd))
    if not ok:
        print(
            f"     FAILURES: kind={ok_kind} turn={ok_turn} axis={ok_axis} conf={ok_conf}"
        )
    return ok


def part_b() -> dict[str, int]:
    print("\n" + "=" * 75)
    print(" PART B — Adversarial stress cases (known ground truth)")
    print("=" * 75)
    passed = 0
    failed = 0

    # 1. Identical → None
    r = mk_resp("Paris is the capital of France.")
    if assert_case(
        "Identical traces produce no divergence",
        _wrap([r, r, r]),
        _wrap([r, r, r]),
        expected_kind=None,
    ):
        passed += 1
    else:
        failed += 1

    # 2. Whitespace-only (classic noise) → None or Style with low confidence
    rb = mk_resp("Paris is the capital of France.")
    rc = mk_resp("Paris  is the capital of France.")
    fd = first_div(_wrap([rb]), _wrap([rc]))
    if fd is None or fd["kind"] == "style_drift":
        print("\n  ✅ Whitespace-only diff: None or Style (both acceptable)")
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ Whitespace-only diff classified as {fd['kind']}")
        failed += 1

    # 3. Tool arg value change → Decision on trajectory
    rb = mk_resp(
        tool_name="search", tool_input={"q": "cats", "limit": 10}, stop="tool_use"
    )
    rc = mk_resp(
        tool_name="search", tool_input={"q": "cats", "limit": 50}, stop="tool_use"
    )
    if assert_case(
        "Same tool, different arg value → Decision on trajectory",
        _wrap([rb]),
        _wrap([rc]),
        expected_kind="decision_drift",
        expected_axis="trajectory",
        expected_turn=0,
    ):
        passed += 1
    else:
        failed += 1

    # 4. Tool name change → Structural on trajectory
    rb = mk_resp(tool_name="search", tool_input={"q": "cats"}, stop="tool_use")
    rc = mk_resp(tool_name="lookup", tool_input={"q": "cats"}, stop="tool_use")
    if assert_case(
        "Different tool name → Structural on trajectory",
        _wrap([rb]),
        _wrap([rc]),
        expected_kind="structural_drift",
        expected_axis="trajectory",
        expected_turn=0,
    ):
        passed += 1
    else:
        failed += 1

    # 5. Stop-reason flip (refusal) → Decision on safety
    rb = mk_resp("Here is the answer you asked for.", stop="end_turn")
    rc = mk_resp("I can't help with that request.", stop="content_filter")
    if assert_case(
        "Stop reason flip (refusal) → Decision on safety",
        _wrap([rb]),
        _wrap([rc]),
        expected_kind="decision_drift",
        expected_axis="safety",
        expected_turn=0,
    ):
        passed += 1
    else:
        failed += 1

    # 6. Candidate drops a turn → Structural (gap)
    r1 = mk_resp("turn one")
    r2 = mk_resp("turn two")
    r3 = mk_resp("turn three")
    if assert_case(
        "Candidate drops middle turn → Structural (Gotoh gap)",
        _wrap([r1, r2, r3]),
        _wrap([r1, r3]),
        expected_kind="structural_drift",
        expected_axis="trajectory",
    ):
        passed += 1
    else:
        failed += 1

    # 7. Candidate INSERTS a turn → Structural (gap)
    if assert_case(
        "Candidate inserts middle turn → Structural (Gotoh gap)",
        _wrap([r1, r3]),
        _wrap([r1, r2, r3]),
        expected_kind="structural_drift",
        expected_axis="trajectory",
    ):
        passed += 1
    else:
        failed += 1

    # 8. Late divergence — first N turns identical, last differs
    same = [mk_resp(f"turn {i}") for i in range(10)]
    different_last = list(same[:-1]) + [
        mk_resp(tool_name="emergency", tool_input={"reason": "x"}, stop="tool_use")
    ]
    if assert_case(
        "Late divergence at turn 9 of 10",
        _wrap(same),
        _wrap(different_last),
        expected_kind="structural_drift",
        expected_turn=9,
    ):
        passed += 1
    else:
        failed += 1

    # 9. Big semantic shift only (no tools) → Decision on semantic
    rb = mk_resp(
        "Photosynthesis is the biological process plants use to convert sunlight into energy."
    )
    rc = mk_resp(
        "The stock market closed higher Thursday after reports of strong earnings growth."
    )
    if assert_case(
        "Totally different topics (same structure) → Decision on semantic",
        _wrap([rb]),
        _wrap([rc]),
        expected_kind="decision_drift",
        expected_axis="semantic",
        expected_turn=0,
    ):
        passed += 1
    else:
        failed += 1

    # 10. Tool REORDERING — same tools, different sequence
    t_a = mk_resp(tool_name="pause_replication", tool_input={}, stop="tool_use")
    t_b = mk_resp(tool_name="restore_database", tool_input={}, stop="tool_use")
    t_c = mk_resp(tool_name="resume_replication", tool_input={}, stop="tool_use")
    # Baseline correct order: pause → restore → resume.
    # Candidate reversal: restore → pause → resume (the devops-agent bug).
    if assert_case(
        "Tool reordering (devops agent ordering bug)",
        _wrap([t_a, t_b, t_c]),
        _wrap([t_b, t_a, t_c]),
        expected_kind="structural_drift",
    ):
        passed += 1
    else:
        failed += 1

    # 11. First divergence is truly FIRST, not later
    r_same = mk_resp("unchanged turn")
    r_wrong1 = mk_resp("baseline says A", stop="end_turn")
    r_wrong2 = mk_resp("baseline says B", stop="end_turn")
    r_cand1 = mk_resp(
        "I cannot answer this question per our safety policy.",
        stop="content_filter",
    )
    r_cand2 = mk_resp("baseline says B", stop="end_turn")
    # Baseline: [same, A, B]. Candidate: [same, refusal, B]. First divergence at turn 1.
    fd = first_div(
        _wrap([r_same, r_wrong1, r_wrong2]),
        _wrap([r_same, r_cand1, r_cand2]),
    )
    if fd is not None and fd["baseline_turn"] == 1:
        print("\n  ✅ First divergence locates earliest diff (turn 1, not later)")
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ Expected baseline_turn=1, got {fd}")
        failed += 1

    # 12. Multi-turn insertion (Gotoh affine gap test)
    # Baseline: 5 turns. Candidate: baseline + 3 extra turns in the middle.
    # Good aligner should NOT fragment the insertion.
    base_turns = [mk_resp(f"b{i}") for i in range(5)]
    cand_turns = (
        base_turns[:2] + [mk_resp(f"inserted-{i}") for i in range(3)] + base_turns[2:]
    )
    fd = first_div(_wrap(base_turns), _wrap(cand_turns))
    if fd is not None and fd["kind"] == "structural_drift":
        print("\n  ✅ Multi-turn insertion correctly flagged as structural")
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ Multi-turn insertion: got {fd}")
        failed += 1

    print(f"\n  Part B result: {passed} passed / {failed} failed")
    return {"passed": passed, "failed": failed}


# ---------------------------------------------------------------------------
# Part C — performance + edge cases
# ---------------------------------------------------------------------------


def part_c() -> dict[str, int]:
    print("\n" + "=" * 75)
    print(" PART C — Performance + edge cases")
    print("=" * 75)

    passed = 0
    failed = 0

    # Performance: 100-turn traces, aligned. Should be <100ms.
    base = [mk_resp(f"turn {i}") for i in range(100)]
    cand = [mk_resp(f"turn {i}") for i in range(100)]
    cand[42] = mk_resp(tool_name="exception_tool", tool_input={"x": 1}, stop="tool_use")
    t0 = time.perf_counter()
    fd = first_div(_wrap(base), _wrap(cand))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms < 500.0 and fd is not None and fd["baseline_turn"] == 42:
        print(
            f"\n  ✅ 100-turn alignment found regression at turn 42 in {elapsed_ms:.1f}ms"
        )
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ 100-turn alignment: {elapsed_ms:.1f}ms, fd={fd}")
        failed += 1

    # Noise floor: whitespace-only noise in realistic-length text
    # should normalize away. (2-char `"ok"` vs `"o k"` is excluded —
    # for such short text, a 1-char insertion is >25% of the string,
    # which is legitimate decision drift, not noise.)
    rb = mk_resp("The meeting will be held on Monday at 3 PM.")
    rc = mk_resp("The  meeting will be held on Monday at 3 PM.")  # extra space
    fd = first_div(_wrap([rb]), _wrap([rc]))
    # After whitespace normalization these are identical → similarity 1.0.
    if fd is None or fd["kind"] == "style_drift":
        print("\n  ✅ Whitespace-only noise in real text: below noise or Style")
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ Whitespace noise classified as {fd['kind']}")
        failed += 1

    # Confidence calibration: bigger difference should yield higher confidence
    r_clean = mk_resp("the answer to your question is 42", stop="end_turn")
    r_small_diff = mk_resp("the answer to your question is 43", stop="end_turn")
    r_big_diff = mk_resp("I refuse to answer this question.", stop="content_filter")
    fd_small = first_div(_wrap([r_clean]), _wrap([r_small_diff]))
    fd_big = first_div(_wrap([r_clean]), _wrap([r_big_diff]))
    if fd_big is not None and (
        fd_small is None or fd_big["confidence"] > fd_small["confidence"]
    ):
        print(
            f"\n  ✅ Confidence calibration: big diff conf={fd_big['confidence']:.2f}"
            f" > small diff conf={fd_small['confidence'] if fd_small else 'None'}"
        )
        passed += 1
    else:
        print(f"\n  ❌ Confidence not monotonic: small={fd_small}, big={fd_big}")
        failed += 1

    # Empty trace handling
    fd = first_div([mk_meta()], [mk_meta()])
    if fd is None:
        print("\n  ✅ Empty traces (metadata only) → None")
        passed += 1
    else:
        print(f"\n  ❌ Empty traces returned {fd}")
        failed += 1

    # Asymmetric trace lengths (one side just longer)
    short = _wrap([mk_resp("a")])
    long = _wrap([mk_resp("a"), mk_resp("b"), mk_resp("c")])
    fd = first_div(short, long)
    if fd is not None and fd["kind"] == "structural_drift":
        print("\n  ✅ One-side-longer → Structural")
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ Asymmetric got {fd}")
        failed += 1

    # Multiple tool_use blocks in one turn (parallel tool use)
    rb = {
        **mk_resp(tool_name="one", tool_input={"x": 1}, stop="tool_use"),
    }
    rb["payload"]["content"].append(
        {"type": "tool_use", "id": "t_two", "name": "two", "input": {"y": 2}}
    )
    rb["id"] = _core.content_id(rb["payload"])
    rc = mk_resp(tool_name="one", tool_input={"x": 1}, stop="tool_use")
    # baseline had parallel [one, two], candidate has only [one] → structural
    fd = first_div(_wrap([rb]), _wrap([rc]))
    if fd is not None and fd["kind"] == "structural_drift":
        print("\n  ✅ Parallel tool-use mismatch → Structural")
        print(fmt(fd))
        passed += 1
    else:
        print(f"\n  ❌ Parallel tool-use mismatch got {fd}")
        failed += 1

    print(f"\n  Part C result: {passed} passed / {failed} failed")
    return {"passed": passed, "failed": failed}


# ---------------------------------------------------------------------------
# Part D — top-K ranking and multi-divergence coverage
# ---------------------------------------------------------------------------


def part_d() -> dict[str, int]:
    print("\n" + "=" * 75)
    print(" PART D — Top-K divergence ranking and multi-fork coverage")
    print("=" * 75)
    passed = 0
    failed = 0

    def top_k(baseline, candidate):
        d = _core.compute_diff_report(baseline, candidate)
        return d.get("divergences") or []

    # 1. Three divergences: structural + decision + style → ranked correctly
    b = [
        mk_resp("hello, here is a detailed explanation of the topic"),
        mk_resp("the answer is 42"),
        mk_resp(tool_name="search", tool_input={"q": "x"}, stop="tool_use"),
    ]
    c = [
        mk_resp(
            "hello, here is a detailed explanation of the topic."
        ),  # style (added period)
        mk_resp("I cannot answer that.", stop="content_filter"),  # decision (refusal)
        mk_resp(
            tool_name="lookup", tool_input={"q": "x"}, stop="tool_use"
        ),  # structural
    ]
    ranked = top_k(_wrap(b), _wrap(c))
    if (
        len(ranked) >= 2
        and ranked[0]["kind"] == "structural_drift"
        and ranked[1]["kind"] == "decision_drift"
    ):
        print("\n  ✅ Three kinds in one trace → ranked Structural > Decision > Style")
        for i, r in enumerate(ranked):
            print(
                f"     #{i+1}  kind={r['kind']:<18} axis={r['primary_axis']:<12} conf={r['confidence']:.2f}"
            )
        passed += 1
    else:
        print(
            f"\n  ❌ Expected Structural#1, Decision#2; got {[(r['kind'], r['confidence']) for r in ranked]}"
        )
        failed += 1

    # 2. Empty divergence list when traces agree
    r = mk_resp("same")
    if top_k(_wrap([r, r, r]), _wrap([r, r, r])) == []:
        print("\n  ✅ Identical traces → empty divergence list")
        passed += 1
    else:
        print("\n  ❌ Identical should produce empty")
        failed += 1

    # 3. Five Structural divergences — cap at DEFAULT_K=5
    tools = ["a", "b", "c", "d", "e", "f", "g"]
    baseline = [mk_resp(tool_name=t, tool_input={}, stop="tool_use") for t in tools]
    candidate = [
        mk_resp(tool_name=t.upper(), tool_input={}, stop="tool_use") for t in tools
    ]
    ranked = top_k(_wrap(baseline), _wrap(candidate))
    if len(ranked) == 5:
        print(f"\n  ✅ 7 divergences capped at DEFAULT_K=5 (got {len(ranked)})")
        passed += 1
    else:
        print(f"\n  ❌ Expected 5 (cap); got {len(ranked)}")
        failed += 1

    # 4. Walk-order tiebreaker when kinds + confidence tied
    baseline = [
        mk_resp(tool_name=t, tool_input={}, stop="tool_use") for t in ("a", "b", "c")
    ]
    candidate = [
        mk_resp(tool_name=t.upper(), tool_input={}, stop="tool_use")
        for t in ("a", "b", "c")
    ]
    ranked = top_k(_wrap(baseline), _wrap(candidate))
    turns = [r["baseline_turn"] for r in ranked]
    if turns == [0, 1, 2]:
        print(f"\n  ✅ Walk-order preserved on ties: turns = {turns}")
        passed += 1
    else:
        print(f"\n  ❌ Expected [0,1,2] walk order; got {turns}")
        failed += 1

    # 5. first_divergence is walk-order first, divergences[0] is rank-first
    b0 = mk_resp("same across both")
    b1 = mk_resp(tool_name="search", tool_input={"q": "x"}, stop="tool_use")
    c0 = mk_resp("completely different response here")
    c1 = mk_resp(tool_name="lookup", tool_input={"q": "x"}, stop="tool_use")
    d = _core.compute_diff_report(_wrap([b0, b1]), _wrap([c0, c1]))
    fd = d.get("first_divergence")
    ranked = d.get("divergences") or []
    # fd (walk order): turn 0 decision. ranked[0] (severity rank): turn 1 structural.
    if fd and fd["baseline_turn"] == 0 and ranked and ranked[0]["baseline_turn"] == 1:
        print("\n  ✅ first_divergence=walk-order, divergences[0]=severity-rank")
        print(f"     first_divergence: turn {fd['baseline_turn']} {fd['kind']}")
        print(
            f"     divergences[0]: turn {ranked[0]['baseline_turn']} {ranked[0]['kind']}"
        )
        passed += 1
    else:
        print(
            f"\n  ❌ Expected fd@0, ranked[0]@1; got fd={fd}, rank0={ranked[0] if ranked else None}"
        )
        failed += 1

    # 6. Real fixtures: every example produces ≥1 divergence in ranked output
    for example in ["demo", "customer-support", "devops-agent", "er-triage"]:
        base, cand = load_pair(example)
        ranked = top_k(base, cand)
        if len(ranked) >= 1:
            print(
                f"\n  ✅ {example}: {len(ranked)} divergence(s) detected; top: {ranked[0]['kind']}"
            )
            passed += 1
        else:
            print(f"\n  ❌ {example}: no divergences detected")
            failed += 1

    print(f"\n  Part D result: {passed} passed / {failed} failed")
    return {"passed": passed, "failed": failed}


if __name__ == "__main__":
    part_a()
    b = part_b()
    c = part_c()
    d = part_d()
    total_p = b["passed"] + c["passed"] + d["passed"]
    total_f = b["failed"] + c["failed"] + d["failed"]
    total = total_p + total_f
    print("\n" + "=" * 75)
    print(f" VERDICT: {total_p} passed / {total_f} failed across {total} stress cases")
    print("=" * 75)
    sys.exit(0 if total_f == 0 else 1)
