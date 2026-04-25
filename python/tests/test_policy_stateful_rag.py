"""Tests for the v2.0 stateful + RAG-grounding policy rule kinds:
``must_remain_consistent``, ``must_followup``, ``must_be_grounded``.
"""

from __future__ import annotations

from typing import Any

import pytest

from shadow.errors import ShadowConfigError
from shadow.hierarchical import (
    PolicyRule,
    _check_must_be_grounded,
    _check_must_followup,
    _check_must_remain_consistent,
    _unigram_precision,
    load_policy,
    policy_diff,
)

# ---- helpers ------------------------------------------------------------


def _meta() -> dict[str, Any]:
    return {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}}


def _req(model: str = "m", **payload: Any) -> dict[str, Any]:
    p: dict[str, Any] = {"model": model, "messages": [], "params": {}}
    p.update(payload)
    return {"kind": "chat_request", "id": "sha256:q", "ts": "t", "parent": "sha256:m", "payload": p}


def _resp(text: str = "", *, tool_use: dict[str, Any] | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_use:
        content.append({"type": "tool_use", **tool_use})
    payload = {
        "model": "m",
        "content": content,
        "stop_reason": "tool_use" if tool_use else "end_turn",
        "latency_ms": 0,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }
    return {
        "kind": "chat_response",
        "id": "sha256:r",
        "ts": "t",
        "parent": "sha256:q",
        "payload": payload,
    }


def _trace(*pairs: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [_meta()]
    for req, resp in pairs:
        out.extend([req, resp])
    return out


# ====================================================================
# must_remain_consistent
# ====================================================================


def test_must_remain_consistent_passes_when_value_is_stable() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_remain_consistent",
        params={"path": "request.params.amount"},
        severity="error",
    )
    records = _trace(
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={"amount": 500}), _resp("ok")),
    )
    assert _check_must_remain_consistent(rule, records) == []


def test_must_remain_consistent_flags_a_change() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_remain_consistent",
        params={"path": "request.params.amount"},
        severity="error",
    )
    records = _trace(
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={"amount": 700}), _resp("ok")),  # changed
    )
    out = _check_must_remain_consistent(rule, records)
    assert len(out) == 1
    assert out[0].pair_index == 2
    assert "500" in out[0].detail
    assert "700" in out[0].detail


def test_must_remain_consistent_skips_pairs_where_path_is_absent() -> None:
    """Absence is not change. The rule pins consistency only when
    observed."""
    rule = PolicyRule(
        id="r",
        kind="must_remain_consistent",
        params={"path": "request.params.amount"},
        severity="error",
    )
    records = _trace(
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={}), _resp("ok")),  # path absent — skipped
        (_req(params={"amount": 500}), _resp("ok")),  # back to anchor — fine
    )
    assert _check_must_remain_consistent(rule, records) == []


def test_must_remain_consistent_anchors_on_first_observed_value() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_remain_consistent",
        params={"path": "response.stop_reason"},
        severity="error",
    )
    records = _trace(
        (_req(), _resp("hi")),  # stop_reason = end_turn
        (_req(), _resp("hi")),  # end_turn — OK
        (_req(), _resp(tool_use={"id": "t", "name": "f", "input": {}})),  # tool_use — VIOLATION
    )
    out = _check_must_remain_consistent(rule, records)
    assert len(out) == 1
    assert out[0].pair_index == 2


def test_must_remain_consistent_missing_path_param_is_violation() -> None:
    rule = PolicyRule(id="r", kind="must_remain_consistent", params={}, severity="error")
    out = _check_must_remain_consistent(rule, _trace((_req(), _resp("hi"))))
    assert len(out) == 1
    assert "missing" in out[0].detail.lower()


# ====================================================================
# must_followup
# ====================================================================


def test_must_followup_tool_call_passes_when_obligation_met() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_followup",
        params={
            "trigger": [{"path": "response.stop_reason", "op": "==", "value": "end_turn"}],
            "must": {"kind": "tool_call", "tool_name": "send_followup_email"},
        },
        severity="error",
    )
    records = _trace(
        (_req(), _resp("done.")),  # trigger fires (stop_reason==end_turn)
        (_req(), _resp(tool_use={"id": "t", "name": "send_followup_email", "input": {}})),
        # Trigger fires again on the last pair — final-pair violation expected.
    )
    out = _check_must_followup(rule, records)
    # Pair 0 trigger: pair 1 satisfies.
    # Pair 1 trigger (also end_turn? no — pair 1's stop_reason is tool_use)
    # so pair 1 doesn't trigger. Result: 0 violations.
    assert out == []


def test_must_followup_flags_unmet_tool_call_obligation() -> None:
    """Trigger only on tool_use stop_reason so only pair 0 fires —
    keeps the test scoped to the unmet-obligation case alone."""
    rule = PolicyRule(
        id="r",
        kind="must_followup",
        params={
            "trigger": [{"path": "response.stop_reason", "op": "==", "value": "tool_use"}],
            "must": {"kind": "tool_call", "tool_name": "send_followup_email"},
        },
        severity="error",
    )
    records = _trace(
        (_req(), _resp(tool_use={"id": "t", "name": "f", "input": {}})),  # trigger fires
        (_req(), _resp("nothing happened")),  # NO send_followup_email — VIOLATION at pair 1
    )
    out = _check_must_followup(rule, records)
    assert len(out) == 1
    assert out[0].pair_index == 1


def test_must_followup_flags_trigger_on_final_pair() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_followup",
        params={
            "trigger": [{"path": "response.stop_reason", "op": "==", "value": "end_turn"}],
            "must": {"kind": "text_includes", "text": "thanks"},
        },
        severity="error",
    )
    records = _trace(
        (_req(), _resp("first turn")),
        (_req(), _resp("done.")),  # trigger fires on FINAL pair — violation
    )
    out = _check_must_followup(rule, records)
    # Pair 0 doesn't trigger (no end_turn match? actually default
    # stop_reason is end_turn for plain text). Both pairs trigger.
    # Pair 0: pair 1 must satisfy text_includes 'thanks' — "done." doesn't.
    # Pair 1: final-pair violation.
    assert len(out) == 2


def test_must_followup_text_includes_satisfied_by_substring() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_followup",
        params={
            "trigger": [{"path": "response.stop_reason", "op": "==", "value": "tool_use"}],
            "must": {"kind": "text_includes", "text": "complete"},
        },
        severity="error",
    )
    records = _trace(
        (_req(), _resp(tool_use={"id": "t", "name": "f", "input": {}})),  # trigger
        (_req(), _resp("the operation is complete now")),  # satisfies
    )
    assert _check_must_followup(rule, records) == []


def test_must_followup_unknown_must_kind_is_violation() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_followup",
        params={"must": {"kind": "wave_a_magic_wand"}},
        severity="error",
    )
    out = _check_must_followup(rule, _trace((_req(), _resp("hi"))))
    assert len(out) == 1
    assert "unknown" in out[0].detail.lower()


# ====================================================================
# must_be_grounded
# ====================================================================


def test_unigram_precision_full_overlap_is_one() -> None:
    assert _unigram_precision("the cat sat", ["the cat sat on the mat"]) == 1.0


def test_unigram_precision_zero_overlap_is_zero() -> None:
    assert _unigram_precision("alpha bravo charlie", ["delta echo foxtrot"]) == 0.0


def test_unigram_precision_strips_punctuation_and_short_tokens() -> None:
    """`a , .` shouldn't count as 3 tokens; punctuation drops to nothing
    and len-1 tokens are filtered. Otherwise an attacker could grind
    precision up by emitting only stopwords."""
    assert _unigram_precision("a , .", ["completely unrelated"]) == 1.0  # vacuous


def test_must_be_grounded_passes_when_response_overlaps_chunks() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_be_grounded",
        params={
            "retrieval_path": "request.metadata.retrieved_chunks",
            "min_unigram_precision": 0.5,
        },
        severity="error",
    )
    records = _trace(
        (
            _req(metadata={"retrieved_chunks": ["the refund window is thirty days from purchase"]}),
            _resp("Your refund window is thirty days from the date of purchase."),
        )
    )
    assert _check_must_be_grounded(rule, records) == []


def test_must_be_grounded_flags_hallucination() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_be_grounded",
        params={
            "retrieval_path": "request.metadata.retrieved_chunks",
            "min_unigram_precision": 0.5,
        },
        severity="error",
    )
    records = _trace(
        (
            _req(metadata={"retrieved_chunks": ["the refund window is thirty days"]}),
            _resp("I will issue a refund of one million dollars to your bank account immediately."),
        )
    )
    out = _check_must_be_grounded(rule, records)
    assert len(out) == 1
    assert "precision" in out[0].detail.lower()


def test_must_be_grounded_skips_pairs_without_retrieval() -> None:
    """No retrieval = no obligation. The rule fires only when RAG is
    in play."""
    rule = PolicyRule(
        id="r",
        kind="must_be_grounded",
        params={"retrieval_path": "request.metadata.retrieved_chunks"},
        severity="error",
    )
    records = _trace(
        (_req(), _resp("totally ungrounded statement about nothing")),  # no retrieval — skipped
    )
    assert _check_must_be_grounded(rule, records) == []


def test_must_be_grounded_accepts_string_or_list_chunks() -> None:
    rule_string = PolicyRule(
        id="r",
        kind="must_be_grounded",
        params={"retrieval_path": "request.context", "min_unigram_precision": 0.5},
        severity="error",
    )
    records_string = _trace(
        (_req(context="alpha bravo charlie delta"), _resp("alpha bravo charlie")),
    )
    assert _check_must_be_grounded(rule_string, records_string) == []


def test_must_be_grounded_invalid_threshold_is_violation() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_be_grounded",
        params={"retrieval_path": "request.context", "min_unigram_precision": 1.5},
        severity="error",
    )
    out = _check_must_be_grounded(rule, _trace((_req(context="x"), _resp("y"))))
    assert len(out) == 1


# ====================================================================
# integration through load_policy + policy_diff
# ====================================================================


def test_load_policy_accepts_all_three_new_kinds() -> None:
    rules = load_policy(
        [
            {
                "id": "consistency",
                "kind": "must_remain_consistent",
                "params": {"path": "request.params.amount"},
                "severity": "error",
            },
            {
                "id": "followup",
                "kind": "must_followup",
                "params": {"must": {"kind": "tool_call", "tool_name": "x"}},
                "severity": "warning",
            },
            {
                "id": "grounded",
                "kind": "must_be_grounded",
                "params": {"retrieval_path": "request.context"},
                "severity": "error",
            },
        ]
    )
    assert {r.kind for r in rules} == {
        "must_remain_consistent",
        "must_followup",
        "must_be_grounded",
    }


def test_policy_diff_flags_runtime_rule_regression() -> None:
    """policy_diff treats new violations in candidate as regressions —
    the new rule kinds must thread through that pipeline."""
    rules = load_policy(
        [
            {
                "id": "amount-locked",
                "kind": "must_remain_consistent",
                "params": {"path": "request.params.amount"},
                "severity": "error",
            }
        ]
    )
    baseline = _trace(
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={"amount": 500}), _resp("ok")),
    )
    candidate = _trace(
        (_req(params={"amount": 500}), _resp("ok")),
        (_req(params={"amount": 700}), _resp("ok")),  # changed!
    )
    diff = policy_diff(baseline, candidate, rules)
    assert len(diff.regressions) == 1
    assert diff.regressions[0].kind == "must_remain_consistent"


def test_invalid_severity_still_rejected_for_new_kinds() -> None:
    """The v1.7.6 severity validation must apply to the new kinds too."""
    with pytest.raises(ShadowConfigError, match="invalid severity"):
        load_policy(
            [
                {
                    "id": "x",
                    "kind": "must_remain_consistent",
                    "params": {"path": "x"},
                    "severity": "yolo",
                }
            ]
        )
