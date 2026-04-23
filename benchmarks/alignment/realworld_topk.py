"""Real-world validation of top-K divergence ranking.

Scenario: "Customer support agent — quarterly prompt refresh PR"

An e-commerce support agent handles a 10-turn interactive session
spanning order lookup, exchange, refund, and confirmation. A candidate
PR from a careless engineer introduces FIVE known regressions:

  Turn 2 — STYLE drift        (minor rewording)
  Turn 4 — DECISION drift     (safety: refuses a benign cancellation)
  Turn 6 — DECISION drift     (trajectory: wrong refund amount)
  Turn 8 — STRUCTURAL drift   (extra tool call inserted)
  Turn 9 — STRUCTURAL drift   (required tool call dropped)

Ground truth: top-K should find all 5, rank Structural > Decision >
Style, and produce explanations accurate enough for a human reviewer
to identify each regression without looking at the traces.

Second scenario: a 50-turn trace to validate performance and ranking
quality at scale.
"""

from __future__ import annotations

import sys
import time
from typing import Any

sys.path.insert(0, "python/src")
from shadow import _core  # noqa: E402


# ---------------------------------------------------------------------------
# record builders
# ---------------------------------------------------------------------------


def mk_meta() -> dict[str, Any]:
    payload = {"sdk": {"name": "shadow"}}
    return {
        "version": "0.1",
        "id": _core.content_id(payload),
        "kind": "metadata",
        "ts": "2026-04-23T10:00:00Z",
        "parent": None,
        "payload": payload,
    }


def mk_turn(
    turn_index: int,
    text: str = "",
    tools: list[tuple[str, dict]] | None = None,
    stop: str = "end_turn",
) -> dict[str, Any]:
    """Build a chat_response record for turn `turn_index`."""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for name, args in tools or []:
        content.append(
            {
                "type": "tool_use",
                "id": f"t{turn_index}_{name}",
                "name": name,
                "input": args,
            }
        )
    payload = {
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop,
        "latency_ms": 420 + turn_index * 17,
        "usage": {"input_tokens": 200, "output_tokens": 60, "thinking_tokens": 0},
    }
    return {
        "version": "0.1",
        "id": _core.content_id(payload),
        "kind": "chat_response",
        "ts": f"2026-04-23T10:00:{turn_index:02d}.000Z",
        "parent": None,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Scenario: "customer support agent quarterly prompt refresh"
# ---------------------------------------------------------------------------


def build_scenario_10_turn() -> tuple[list[dict], list[dict]]:
    """Build baseline + candidate with 5 known regressions.

    Baseline turns (ground truth, what SHOULD happen):
      0: greet, lookup_order(order_id=A123)
      1: confirm details, offer options
      2: "We will process your refund. It should arrive in 3-5 days."
      3: ask about replacement preference
      4: "Sure, I can cancel that for you." (agreeable)
      5: cancel_order(order_id=A123)
      6: refund(order_id=A123, amount=99.99)
      7: ask if anything else
      8: lookup_order(order_id=A123)  [one lookup to verify refund applied]
      9: send_confirmation_email(to=customer@example.com, order_id=A123)

    Candidate turns (with planted regressions):
      0: identical
      1: identical
      2: "We'll process your refund. It should arrive in 3-5 days."
         [STYLE: "We will" → "We'll"]
      3: identical
      4: "I cannot help with that request."
         [DECISION: refused an OK cancellation]
      5: cancel_order(order_id=A123)
      6: refund(order_id=A123, amount=9.99)
         [DECISION: wrong refund amount (100x smaller!)]
      7: identical
      8: lookup_order + lookup_order  [STRUCTURAL: extra tool call]
      9: (nothing — email dropped)  [STRUCTURAL: missing required tool]
    """
    # Baseline: correct 10-turn flow
    baseline = [
        mk_meta(),
        # Turn 0: greet + lookup
        mk_turn(
            0,
            text="Hello! I'll help you with your order. Let me look that up.",
            tools=[("lookup_order", {"order_id": "A123"})],
            stop="tool_use",
        ),
        # Turn 1: confirm + options
        mk_turn(
            1,
            text="I see your order for the wireless headphones. "
            "Would you like a refund, exchange, or repair?",
        ),
        # Turn 2: acknowledge refund choice
        mk_turn(
            2,
            text="We will process your refund. It should arrive in 3 to 5 business days.",
        ),
        # Turn 3: ask about replacement
        mk_turn(
            3,
            text="Before we finalize, would you like us to ship a replacement pair?",
        ),
        # Turn 4: OK, cancel request
        mk_turn(
            4,
            text="Sure, I can cancel that for you. Let me process the cancellation now.",
        ),
        # Turn 5: actually cancel
        mk_turn(
            5,
            tools=[("cancel_order", {"order_id": "A123"})],
            stop="tool_use",
        ),
        # Turn 6: issue refund
        mk_turn(
            6,
            tools=[("refund", {"order_id": "A123", "amount": 99.99})],
            stop="tool_use",
        ),
        # Turn 7: follow up
        mk_turn(7, text="Is there anything else I can help you with today?"),
        # Turn 8: verify refund applied
        mk_turn(
            8,
            tools=[("lookup_order", {"order_id": "A123"})],
            stop="tool_use",
        ),
        # Turn 9: send confirmation email
        mk_turn(
            9,
            tools=[
                (
                    "send_confirmation_email",
                    {"to": "customer@example.com", "order_id": "A123"},
                )
            ],
            stop="tool_use",
        ),
    ]

    # Candidate: 5 planted regressions
    candidate = [
        mk_meta(),
        # Turn 0: identical
        mk_turn(
            0,
            text="Hello! I'll help you with your order. Let me look that up.",
            tools=[("lookup_order", {"order_id": "A123"})],
            stop="tool_use",
        ),
        # Turn 1: identical
        mk_turn(
            1,
            text="I see your order for the wireless headphones. "
            "Would you like a refund, exchange, or repair?",
        ),
        # Turn 2: STYLE — contraction
        mk_turn(
            2,
            text="We'll process your refund. It should arrive in 3 to 5 business days.",
        ),
        # Turn 3: identical
        mk_turn(
            3,
            text="Before we finalize, would you like us to ship a replacement pair?",
        ),
        # Turn 4: DECISION — refused an OK cancellation
        mk_turn(
            4,
            text="I cannot help with that request.",
            stop="content_filter",
        ),
        # Turn 5: identical
        mk_turn(
            5,
            tools=[("cancel_order", {"order_id": "A123"})],
            stop="tool_use",
        ),
        # Turn 6: DECISION — wrong refund amount (100x less!)
        mk_turn(
            6,
            tools=[("refund", {"order_id": "A123", "amount": 9.99})],
            stop="tool_use",
        ),
        # Turn 7: identical
        mk_turn(7, text="Is there anything else I can help you with today?"),
        # Turn 8: STRUCTURAL — extra tool call inserted
        mk_turn(
            8,
            tools=[
                ("lookup_order", {"order_id": "A123"}),
                ("lookup_order", {"order_id": "A123"}),  # duplicated!
            ],
            stop="tool_use",
        ),
        # Turn 9: STRUCTURAL — dropped send_confirmation_email, returned text only
        mk_turn(
            9,
            text="All done!",
        ),
    ]

    return baseline, candidate


def build_scenario_50_turn() -> tuple[list[dict], list[dict]]:
    """Build a 50-turn trace with 3 planted regressions at scale.

    Most turns are identical; divergences placed at turns 7, 23, 42.
    Tests ranking + performance at realistic multi-conversation scale.
    """
    baseline = [mk_meta()]
    candidate = [mk_meta()]
    for i in range(50):
        # Regression at turn 7 (STRUCTURAL)
        if i == 7:
            baseline.append(
                mk_turn(
                    i,
                    tools=[("fetch_policy", {"id": "refund"})],
                    stop="tool_use",
                )
            )
            candidate.append(
                mk_turn(
                    i,
                    text="I don't know the policy. Let me just process it.",
                )
            )
        # Regression at turn 23 (DECISION — arg value)
        elif i == 23:
            baseline.append(
                mk_turn(
                    i,
                    tools=[("charge_card", {"id": "C42", "amount": 500})],
                    stop="tool_use",
                )
            )
            candidate.append(
                mk_turn(
                    i,
                    tools=[
                        ("charge_card", {"id": "C42", "amount": 50})
                    ],  # wrong amount
                    stop="tool_use",
                )
            )
        # Regression at turn 42 (DECISION — stop_reason flip)
        elif i == 42:
            baseline.append(
                mk_turn(i, text="Absolutely, let me handle that for you right away.")
            )
            candidate.append(
                mk_turn(
                    i,
                    text="I'm unable to process that request.",
                    stop="content_filter",
                )
            )
        else:
            # Identical turns
            r = mk_turn(i, text=f"This is turn {i}; nothing interesting here.")
            baseline.append(r)
            candidate.append(
                {
                    **r,
                    "id": _core.content_id(r["payload"]),
                }
            )
    return baseline, candidate


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 76)
    print(" REAL-WORLD VALIDATION — top-K ranking on multi-regression scenarios")
    print("=" * 76)
    failures = 0

    # -------- Scenario 1: 10-turn with 5 planted regressions --------
    print("\n" + "━" * 76)
    print(" Scenario 1: 10-turn customer support agent, 5 planted regressions")
    print("━" * 76)
    baseline, candidate = build_scenario_10_turn()
    report = _core.compute_diff_report(baseline, candidate)
    divs = report["divergences"]
    print(f"\n Top-K returned {len(divs)} divergence(s):\n")
    for i, d in enumerate(divs):
        print(
            f"   #{i + 1:<2} turn b#{d['baseline_turn']} ↔ c#{d['candidate_turn']}  "
            f"kind={d['kind']:<18} axis={d['primary_axis']:<12} conf={d['confidence']:.2f}"
        )
        print(f"       {d['explanation']}")

    # Expected — we planted 5 regressions:
    #   turn 2: Style
    #   turn 4: Decision (safety — stop_reason flip)
    #   turn 6: Decision (trajectory — arg value)
    #   turn 8: Structural (extra tool)
    #   turn 9: Structural (missing tool)
    # DEFAULT_K = 5, so we expect all 5 returned.

    # Ground-truth checks:
    print("\n Ground-truth checks:")
    checks = []

    # 1. We expect >= 4 divergences (at DEFAULT_K=5 cap, but noise floor may filter Style)
    checks.append(
        ("at least 4 divergences returned", len(divs) >= 4, f"got {len(divs)}")
    )

    # 2. Top-ranked must be Structural (our 2 structural turns have the highest-severity class)
    checks.append(
        (
            "rank #1 is Structural",
            divs[0]["kind"] == "structural_drift" if divs else False,
            divs[0]["kind"] if divs else "no divergences",
        )
    )

    # 3. Both structural turns (8 and 9) must appear in top 3 (they're highest class)
    top3_turns = [d["baseline_turn"] for d in divs[:3]]
    has_turn8 = 8 in top3_turns
    has_turn9 = 9 in top3_turns
    checks.append(
        (
            "both structural regressions (turns 8 & 9) in top 3",
            has_turn8 and has_turn9,
            f"top-3 turns = {top3_turns}",
        )
    )

    # 4. Decision drifts (turns 4 and 6) should appear somewhere in output
    all_turns = [d["baseline_turn"] for d in divs]
    has_turn4 = 4 in all_turns
    has_turn6 = 6 in all_turns
    checks.append(
        (
            "decision drifts (turns 4 & 6) are detected",
            has_turn4 and has_turn6,
            f"detected turns = {sorted(all_turns)}",
        )
    )

    # 5. Decision drifts must rank below Structural (class hierarchy)
    decision_ranks = [i for i, d in enumerate(divs) if d["kind"] == "decision_drift"]
    structural_ranks = [
        i for i, d in enumerate(divs) if d["kind"] == "structural_drift"
    ]
    if decision_ranks and structural_ranks:
        checks.append(
            (
                "every Structural ranks above every Decision",
                max(structural_ranks) < min(decision_ranks),
                f"structural={structural_ranks} decision={decision_ranks}",
            )
        )

    # 6. If Style appears, it must be last (ranks below everything)
    style_ranks = [i for i, d in enumerate(divs) if d["kind"] == "style_drift"]
    if style_ranks:
        other_ranks = [i for i, d in enumerate(divs) if d["kind"] != "style_drift"]
        if other_ranks:
            checks.append(
                (
                    "style drift ranks last",
                    min(style_ranks) > max(other_ranks),
                    f"style={style_ranks} other={other_ranks}",
                )
            )

    # 7. Explanations should mention key terms for each regression
    explanations = " ".join(d["explanation"].lower() for d in divs)
    checks.append(
        (
            "explanations mention refund/tool/cancel somewhere",
            any(
                kw in explanations
                for kw in [
                    "tool",
                    "refund",
                    "lookup_order",
                    "send_confirmation",
                    "stop_reason",
                ]
            ),
            "explanations don't mention expected terms",
        )
    )

    for desc, ok, detail in checks:
        print(f"   {'✅' if ok else '❌'} {desc}")
        if not ok:
            print(f"      detail: {detail}")
            failures += 1

    # -------- Scenario 2: 50-turn with 3 planted regressions --------
    print("\n" + "━" * 76)
    print(" Scenario 2: 50-turn trace with 3 planted regressions")
    print("━" * 76)
    baseline, candidate = build_scenario_50_turn()
    t0 = time.perf_counter()
    report = _core.compute_diff_report(baseline, candidate)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    divs = report["divergences"]
    print(f"\n 50-turn compute_diff_report finished in {elapsed_ms:.1f}ms")
    print(f" Top-K returned {len(divs)} divergence(s):\n")
    for i, d in enumerate(divs):
        print(
            f"   #{i + 1:<2} turn b#{d['baseline_turn']} ↔ c#{d['candidate_turn']}  "
            f"kind={d['kind']:<18} axis={d['primary_axis']:<12} conf={d['confidence']:.2f}"
        )
        print(f"       {d['explanation'][:90]}")

    # Expected: 3 divergences at turns 7 (structural), 23 (decision), 42 (decision).
    checks2 = []
    checks2.append(("runs in under 500ms", elapsed_ms < 500, f"{elapsed_ms:.1f}ms"))
    checks2.append(("finds 3 divergences", len(divs) == 3, f"got {len(divs)}"))
    turns = sorted(d["baseline_turn"] for d in divs)
    checks2.append(
        (
            "located at expected turns [7, 23, 42]",
            turns == [7, 23, 42],
            f"found at {turns}",
        )
    )
    if divs:
        checks2.append(
            (
                "rank #1 is Structural (turn 7)",
                divs[0]["kind"] == "structural_drift" and divs[0]["baseline_turn"] == 7,
                f"got kind={divs[0]['kind']} turn={divs[0]['baseline_turn']}",
            )
        )
    for desc, ok, detail in checks2:
        print(f"   {'✅' if ok else '❌'} {desc}")
        if not ok:
            print(f"      detail: {detail}")
            failures += 1

    # -------- Summary --------
    total = len(checks) + len(checks2)
    passed = total - failures
    print("\n" + "=" * 76)
    print(f" VERDICT: {passed}/{total} real-world checks passed")
    print("=" * 76)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
