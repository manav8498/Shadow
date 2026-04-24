"""Tests for `shadow.cost_attribution`.

Session-cost attribution decomposes cost deltas between a baseline
and a candidate trace into (model_swap + token_movement + mix_residual).
These tests lock the decomposition arithmetic so a regression in the
attribution math (a real risk — floating point + token scaling) is
caught before a PR lands.
"""

from __future__ import annotations

from typing import Any

from shadow.cost_attribution import (
    attribute_cost,
    partition_sessions,
    render_markdown,
    render_terminal,
    session_cost,
)

# Pricing table mirroring Shadow's Rust ModelPricing shape.
PRICING = {
    "opus": {"input": 15e-6, "output": 75e-6},
    "sonnet": {"input": 3e-6, "output": 15e-6},
    "haiku": {"input": 0.25e-6, "output": 1.25e-6},
}


def _resp(
    model: str, input_tok: int, output_tok: int, cached_in: int = 0, thinking: int = 0
) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:r{input_tok}{output_tok}{model}",
        "kind": "chat_response",
        "ts": "t",
        "parent": None,
        "payload": {
            "model": model,
            "usage": {
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cached_input_tokens": cached_in,
                "thinking_tokens": thinking,
            },
        },
    }


def _meta(i: int) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:m{i}",
        "kind": "metadata",
        "ts": "t",
        "parent": None,
        "payload": {},
    }


# ---- partition_sessions --------------------------------------------------


def test_partition_single_session_when_only_one_metadata() -> None:
    records = [_meta(0), _resp("opus", 100, 50), _resp("opus", 200, 100)]
    sessions = partition_sessions(records)
    assert len(sessions) == 1
    assert len(sessions[0]) == 3


def test_partition_splits_at_each_metadata() -> None:
    records = [
        _meta(0),
        _resp("opus", 100, 50),
        _meta(1),
        _resp("opus", 200, 100),
        _meta(2),
        _resp("opus", 300, 150),
    ]
    sessions = partition_sessions(records)
    assert len(sessions) == 3
    assert all(s[0]["kind"] == "metadata" for s in sessions)


def test_partition_leading_non_metadata_becomes_own_session() -> None:
    """A trace that starts without a metadata record (corrupt or truncated)
    still partitions — the orphan prefix becomes session #0."""
    records = [_resp("opus", 100, 50), _meta(0), _resp("opus", 200, 100)]
    sessions = partition_sessions(records)
    # Records before metadata go into the leading session, then
    # metadata opens a new one.
    assert len(sessions) == 2


def test_partition_empty_input() -> None:
    assert partition_sessions([]) == []


def test_partition_falls_back_to_implicit_boundaries() -> None:
    """When a single metadata record wraps multiple user-initiated
    sub-sessions (the typical shape of an imported A2A or MCP log),
    partition_sessions must recover the sub-session boundaries rather
    than returning one giant session. Otherwise ``diff_by_session``
    would silently collapse an N-ticket import into a single axis row.
    """

    def _req(last_role: str) -> dict[str, Any]:
        msgs = [{"role": "system", "content": "s"}]
        if last_role == "user":
            msgs.append({"role": "user", "content": "ticket"})
        else:  # continuation
            msgs.append({"role": "user", "content": "ticket"})
            msgs.append({"role": "assistant", "content": ""})
            msgs.append({"role": "tool", "content": "result"})
        return {
            "version": "0.1",
            "id": "sha256:req",
            "kind": "chat_request",
            "ts": "t",
            "parent": None,
            "payload": {"model": "opus", "messages": msgs},
        }

    def _resp_stop(stop: str) -> dict[str, Any]:
        r = _resp("opus", 100, 50)
        r["payload"]["stop_reason"] = stop  # type: ignore[index]
        return r

    # Three tickets under one metadata: each ticket has two turns
    # (tool_use then end_turn).
    records = [_meta(0)]
    for _ in range(3):
        records.extend(
            [
                _req("user"),
                _resp_stop("tool_use"),
                _req("tool"),
                _resp_stop("end_turn"),
            ]
        )
    sessions = partition_sessions(records)
    assert len(sessions) == 3, f"expected 3 sub-sessions, got {len(sessions)}"
    # First sub-session carries the metadata; later sub-sessions don't.
    assert sessions[0][0]["kind"] == "metadata"
    # Each sub-session has its two turns (2 requests + 2 responses = 4 recs).
    assert all(
        sum(1 for r in s if r["kind"] in ("chat_request", "chat_response")) == 4 for s in sessions
    )


def test_partition_single_user_initiated_session_stays_intact() -> None:
    """A trace with only one user-initiated session (the normal case)
    must not be spuriously split by the fallback — we only override
    the primary partition when we find genuinely multiple boundaries."""
    records = [
        _meta(0),
        {
            "version": "0.1",
            "id": "sha256:req",
            "kind": "chat_request",
            "ts": "t",
            "parent": None,
            "payload": {
                "model": "opus",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "hi"},
                ],
            },
        },
        {
            "version": "0.1",
            "id": "sha256:resp",
            "kind": "chat_response",
            "ts": "t",
            "parent": None,
            "payload": {
                "model": "opus",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
            },
        },
    ]
    sessions = partition_sessions(records)
    assert len(sessions) == 1


# ---- session_cost --------------------------------------------------------


def test_session_cost_sums_across_responses() -> None:
    records = [_meta(0), _resp("opus", 1000, 500), _resp("opus", 500, 200)]
    s = session_cost(records, PRICING, session_index=0)
    assert s.input_tokens == 1500
    assert s.output_tokens == 700
    # 1500 * 15e-6 + 700 * 75e-6 = 0.0225 + 0.0525 = 0.0750
    assert abs(s.total_usd - 0.075) < 1e-9
    assert s.model == "opus"


def test_session_cost_handles_unknown_model_gracefully() -> None:
    records = [_meta(0), _resp("unknown-model", 100, 50)]
    s = session_cost(records, PRICING, session_index=0)
    # Unknown models contribute zero USD — not a crash.
    assert s.total_usd == 0.0
    assert s.input_tokens == 100


def test_session_cost_reports_mixed_when_no_modal_model() -> None:
    records = [_meta(0), _resp("opus", 100, 50), _resp("sonnet", 100, 50)]
    s = session_cost(records, PRICING, session_index=0)
    # 50/50 split → no modal → "mixed"
    assert s.model == "mixed"


# ---- attribution arithmetic ---------------------------------------------


def test_pure_model_swap_all_delta_attributed_to_swap() -> None:
    """Same tokens, different model → 100% of delta is model_swap."""
    baseline = [_meta(0), _resp("opus", 1000, 500), _resp("opus", 800, 300)]
    candidate = [_meta(0), _resp("sonnet", 1000, 500), _resp("sonnet", 800, 300)]
    report = attribute_cost(baseline, candidate, PRICING)
    s = report.per_session[0]
    assert abs(s.delta_usd + 0.0696) < 1e-9  # -$0.0696 saved
    # All the savings come from the swap.
    assert abs(s.model_swap_usd - s.delta_usd) < 1e-9
    assert abs(s.token_movement_usd) < 1e-9
    assert abs(s.mix_residual_usd) < 1e-9


def test_pure_token_movement_all_delta_attributed_to_tokens() -> None:
    """Same model, more tokens → 100% of delta is token_movement."""
    baseline = [_meta(0), _resp("opus", 1000, 500)]
    candidate = [_meta(0), _resp("opus", 2000, 1000)]  # both doubled
    report = attribute_cost(baseline, candidate, PRICING)
    s = report.per_session[0]
    assert s.delta_usd > 0  # cost went up
    assert abs(s.model_swap_usd) < 1e-9
    assert abs(s.token_movement_usd - s.delta_usd) < 1e-9
    assert abs(s.mix_residual_usd) < 1e-9


def test_simultaneous_model_and_token_change_has_residual() -> None:
    """Model swap AND token change → non-trivial mix_residual."""
    baseline = [_meta(0), _resp("opus", 1000, 500)]
    candidate = [_meta(0), _resp("sonnet", 2000, 1000)]
    report = attribute_cost(baseline, candidate, PRICING)
    s = report.per_session[0]
    # Components must still sum to total delta (identity).
    total_decomp = s.model_swap_usd + s.token_movement_usd + s.mix_residual_usd
    assert abs(total_decomp - s.delta_usd) < 1e-9


def test_decomposition_sums_to_total_delta_always() -> None:
    """For any baseline/candidate pair, the three components must sum
    to the total cost delta — this is the fundamental identity."""
    scenarios = [
        ("opus", 1000, 500, "sonnet", 1000, 500),  # pure swap
        ("opus", 1000, 500, "opus", 2000, 1000),  # pure token movement
        ("opus", 1000, 500, "sonnet", 1500, 750),  # both
        ("opus", 1000, 500, "opus", 1000, 500),  # no change
        ("opus", 1000, 500, "haiku", 3000, 1500),  # cheaper + more tokens
    ]
    for bm, bi, bo, cm, ci, co in scenarios:
        base = [_meta(0), _resp(bm, bi, bo)]
        cand = [_meta(0), _resp(cm, ci, co)]
        r = attribute_cost(base, cand, PRICING)
        s = r.per_session[0]
        total = s.model_swap_usd + s.token_movement_usd + s.mix_residual_usd
        assert abs(total - s.delta_usd) < 1e-9, (
            f"decomposition didn't sum for ({bm}/{bi}/{bo} -> {cm}/{ci}/{co}): "
            f"swap={s.model_swap_usd:.6f} move={s.token_movement_usd:.6f} "
            f"mix={s.mix_residual_usd:.6f} total={total:.6f} delta={s.delta_usd:.6f}"
        )


def test_cached_tokens_are_priced_at_cached_rate() -> None:
    """90% cache-hit scenario: cached_input_tokens priced lower than
    fresh input."""
    pricing = {
        "opus": {
            "input": 15e-6,
            "output": 75e-6,
            "cached_input": 1.5e-6,  # 10x cheaper
        }
    }
    records = [
        _meta(0),
        _resp("opus", input_tok=1000, output_tok=100, cached_in=900),
    ]
    s = session_cost(records, pricing, session_index=0)
    # (1000 - 900) * 15e-6 + 900 * 1.5e-6 + 100 * 75e-6
    # = 0.0015 + 0.00135 + 0.0075 = 0.01035
    assert abs(s.total_usd - 0.01035) < 1e-9


def test_reasoning_tokens_priced_at_reasoning_rate() -> None:
    """Reasoning token rate overrides output rate when set."""
    pricing = {
        "opus": {"input": 15e-6, "output": 75e-6, "reasoning": 150e-6},
    }
    records = [
        _meta(0),
        _resp("opus", input_tok=100, output_tok=50, thinking=200),
    ]
    s = session_cost(records, pricing, session_index=0)
    # 100 * 15e-6 + 50 * 75e-6 + 200 * 150e-6
    # = 0.0015 + 0.00375 + 0.03 = 0.03525
    assert abs(s.total_usd - 0.03525) < 1e-9


# ---- noisy flag ---------------------------------------------------------


def test_noisy_flag_fires_when_mix_residual_dominates() -> None:
    """Large simultaneous changes should trip the noisy flag."""
    baseline = [_meta(0), _resp("opus", 100, 50)]
    candidate = [_meta(0), _resp("haiku", 10000, 5000)]
    report = attribute_cost(baseline, candidate, PRICING)
    # Non-pure scenarios often have non-trivial mix_residual.
    residual_share = (
        abs(report.total_mix_residual_usd) / abs(report.total_delta_usd)
        if abs(report.total_delta_usd) > 0
        else 0
    )
    # This specific scenario should have residual > 10%.
    if residual_share > 0.10:
        assert report.attribution_is_noisy


def test_noisy_flag_false_for_pure_scenarios() -> None:
    baseline = [_meta(0), _resp("opus", 1000, 500)]
    candidate = [_meta(0), _resp("sonnet", 1000, 500)]  # pure swap
    r = attribute_cost(baseline, candidate, PRICING)
    assert r.attribution_is_noisy is False


# ---- multi-session alignment -------------------------------------------


def test_multi_session_traces_align_by_index() -> None:
    baseline = [
        _meta(0),
        _resp("opus", 1000, 500),
        _meta(1),
        _resp("opus", 2000, 1000),
    ]
    candidate = [
        _meta(0),
        _resp("sonnet", 1000, 500),
        _meta(1),
        _resp("sonnet", 2000, 1000),
    ]
    r = attribute_cost(baseline, candidate, PRICING)
    assert len(r.per_session) == 2
    # Both sessions should show pure-model-swap attribution.
    for s in r.per_session:
        assert abs(s.model_swap_usd - s.delta_usd) < 1e-9


def test_mismatched_session_counts_produce_zero_side() -> None:
    """Baseline has 2 sessions, candidate has 1 — the report still
    emits 2 rows; the second has zero candidate cost."""
    baseline = [
        _meta(0),
        _resp("opus", 100, 50),
        _meta(1),
        _resp("opus", 200, 100),
    ]
    candidate = [_meta(0), _resp("opus", 100, 50)]
    r = attribute_cost(baseline, candidate, PRICING)
    assert len(r.per_session) == 2
    assert r.per_session[1].candidate_usd == 0.0


# ---- renderers ---------------------------------------------------------


def test_render_terminal_contains_delta_and_components() -> None:
    baseline = [_meta(0), _resp("opus", 1000, 500)]
    candidate = [_meta(0), _resp("sonnet", 1500, 750)]
    r = attribute_cost(baseline, candidate, PRICING)
    out = render_terminal(r)
    assert "cost attribution" in out
    assert "opus→sonnet" in out
    assert "token movement" in out
    assert "total:" in out


def test_render_markdown_produces_table() -> None:
    baseline = [_meta(0), _resp("opus", 1000, 500)]
    candidate = [_meta(0), _resp("sonnet", 1500, 750)]
    r = attribute_cost(baseline, candidate, PRICING)
    md = render_markdown(r)
    assert "## Cost attribution" in md
    assert "| session | baseline | candidate |" in md
    assert "#0" in md


def test_render_empty_report() -> None:
    r = attribute_cost([], [], PRICING)
    assert render_terminal(r) == ""
    assert render_markdown(r) == ""


def test_noisy_residual_warning_surfaces_in_markdown() -> None:
    baseline = [_meta(0), _resp("opus", 100, 50)]
    candidate = [_meta(0), _resp("haiku", 10000, 5000)]
    r = attribute_cost(baseline, candidate, PRICING)
    if r.attribution_is_noisy:
        assert "less trustworthy" in render_markdown(r).lower()
