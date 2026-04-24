"""Tests for `shadow.hierarchical` — session-level + span-level diff.

Two layers Shadow's v0.x reports couldn't reach:

- **Session-level:** which specific conversation in a multi-session
  trace regressed?
- **Span-level:** within one turn's response, which content block
  (text, tool_use, tool_result) actually changed?

These tests pin down the invariants both layers rely on.
"""

from __future__ import annotations

from typing import Any

from shadow.hierarchical import (
    SessionDiff,
    SpanDiff,
    diff_by_session,
    render_session_summary,
    render_spans,
    span_diff,
)


def _meta(i: int) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:m{i:02d}",
        "kind": "metadata",
        "ts": "t",
        "parent": None,
        "payload": {"sdk": {"name": "shadow", "version": "test"}},
    }


def _req(i: int, parent: str) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:q{i:02d}",
        "kind": "chat_request",
        "ts": "t",
        "parent": parent,
        "payload": {
            "model": "x",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "params": {},
        },
    }


def _resp(
    i: int,
    parent: str,
    *,
    text: str = "hi",
    latency: int = 100,
    tokens: int = 10,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:r{i:02d}",
        "kind": "chat_response",
        "ts": "t",
        "parent": parent,
        "payload": {
            "model": "x",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "latency_ms": latency,
            "usage": {"input_tokens": 1, "output_tokens": tokens, "thinking_tokens": 0},
        },
    }


# ---- session-level ------------------------------------------------------


def test_single_session_trace_produces_one_session_diff() -> None:
    baseline = [_meta(0), _req(0, "sha256:m00"), _resp(0, "sha256:q00")]
    candidate = [_meta(0), _req(0, "sha256:m00"), _resp(0, "sha256:q00", text="bye")]
    result = diff_by_session(baseline, candidate)
    assert len(result) == 1
    assert result[0].session_index == 0
    assert result[0].pair_count == 1


def test_multi_session_trace_produces_one_diff_per_session() -> None:
    """3 sessions in, 3 SessionDiff out — with per-session pair counts."""
    baseline = [
        _meta(0),
        _req(0, "sha256:m00"),
        _resp(0, "sha256:q00"),
        _meta(1),
        _req(1, "sha256:m01"),
        _resp(1, "sha256:q01"),
        _meta(2),
        _req(2, "sha256:m02"),
        _resp(2, "sha256:q02"),
    ]
    candidate = [
        _meta(0),
        _req(0, "sha256:m00"),
        _resp(0, "sha256:q00"),
        _meta(1),
        _req(1, "sha256:m01"),
        _resp(1, "sha256:q01", text="changed"),
        _meta(2),
        _req(2, "sha256:m02"),
        _resp(2, "sha256:q02"),
    ]
    result = diff_by_session(baseline, candidate)
    assert len(result) == 3
    # All three sessions have exactly 1 pair.
    assert all(sd.pair_count == 1 for sd in result)


def test_mismatched_session_count_pads_with_empty_session() -> None:
    """Baseline has 2 sessions, candidate has 1 — we still get 2
    SessionDiff entries; the second pairs an empty candidate."""
    baseline = [
        _meta(0),
        _req(0, "sha256:m00"),
        _resp(0, "sha256:q00"),
        _meta(1),
        _req(1, "sha256:m01"),
        _resp(1, "sha256:q01"),
    ]
    candidate = [_meta(0), _req(0, "sha256:m00"), _resp(0, "sha256:q00")]
    result = diff_by_session(baseline, candidate)
    assert len(result) == 2
    # Second session has no candidate pairs → pair_count == 0.
    assert result[1].pair_count == 0


def test_session_diff_captures_worst_severity_per_session() -> None:
    """A regressed session should report a non-'none' worst_severity."""
    baseline = [_meta(0), _req(0, "sha256:m00"), _resp(0, "sha256:q00", latency=100)]
    # Candidate: latency blown up 10x to trip severity thresholds.
    candidate = [
        _meta(0),
        _req(0, "sha256:m00"),
        _resp(0, "sha256:q00", latency=10_000),
    ]
    result = diff_by_session(baseline, candidate)
    assert len(result) == 1
    # Bootstrap CI with n=1 is technically low_power; we just care
    # that the rollup correctly propagates whatever severity the
    # Rust core computed.
    assert result[0].worst_severity in ("minor", "moderate", "severe")


def test_session_diff_includes_full_diff_report() -> None:
    baseline = [_meta(0), _req(0, "sha256:m00"), _resp(0, "sha256:q00")]
    candidate = [_meta(0), _req(0, "sha256:m00"), _resp(0, "sha256:q00", text="bye")]
    result = diff_by_session(baseline, candidate)
    report = result[0].report
    assert "rows" in report
    assert len(report["rows"]) == 9  # nine axes


# ---- span-level ---------------------------------------------------------


def _resp_blocks(blocks: list[dict[str, Any]], stop_reason: str = "end_turn") -> dict[str, Any]:
    """Bare chat_response payload with a custom content block list."""
    return {
        "model": "x",
        "content": blocks,
        "stop_reason": stop_reason,
        "latency_ms": 100,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }


def test_span_diff_identical_responses_has_zero_changes() -> None:
    r = _resp_blocks([{"type": "text", "text": "hi"}])
    assert span_diff(r, r) == []


def test_span_diff_text_block_change_reports_similarity_and_previews() -> None:
    b = _resp_blocks([{"type": "text", "text": "hello there"}])
    c = _resp_blocks([{"type": "text", "text": "hi friend"}])
    spans = span_diff(b, c)
    assert len(spans) == 1
    assert spans[0].kind == "text_block_changed"
    assert "similarity" in spans[0].summary


def test_span_diff_tool_use_added() -> None:
    b = _resp_blocks([{"type": "text", "text": "done"}])
    c = _resp_blocks(
        [
            {"type": "text", "text": "done"},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
        ]
    )
    spans = span_diff(b, c)
    # Block #1 is a tool_use that didn't exist in baseline → added.
    adds = [s for s in spans if s.kind == "tool_use_added"]
    assert len(adds) == 1
    assert "search" in adds[0].summary


def test_span_diff_tool_use_removed() -> None:
    b = _resp_blocks(
        [
            {"type": "text", "text": "done"},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
        ]
    )
    c = _resp_blocks([{"type": "text", "text": "done"}])
    spans = span_diff(b, c)
    removes = [s for s in spans if s.kind == "tool_use_removed"]
    assert len(removes) == 1


def test_span_diff_tool_use_arg_added() -> None:
    b = _resp_blocks(
        [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
        ]
    )
    c = _resp_blocks(
        [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x", "limit": 10}},
        ]
    )
    spans = span_diff(b, c)
    assert len(spans) == 1
    assert spans[0].kind == "tool_use_args_changed"
    assert "limit" in spans[0].summary
    assert "arg added" in spans[0].summary


def test_span_diff_tool_use_arg_rename_is_remove_plus_add() -> None:
    """customer_id → cid should surface as both remove `customer_id`
    and add `cid` inside the same tool_use block."""
    b = _resp_blocks(
        [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"customer_id": "C42"}},
        ]
    )
    c = _resp_blocks(
        [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"cid": "C42"}},
        ]
    )
    spans = span_diff(b, c)
    assert len(spans) == 1
    assert spans[0].kind == "tool_use_args_changed"
    assert "customer_id" in spans[0].summary
    assert "cid" in spans[0].summary


def test_span_diff_tool_result_is_error_flip() -> None:
    b = _resp_blocks(
        [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "text", "text": "ok"}],
                "is_error": False,
            },
        ]
    )
    c = _resp_blocks(
        [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "text", "text": "err"}],
                "is_error": True,
            },
        ]
    )
    spans = span_diff(b, c)
    assert len(spans) == 1
    assert spans[0].kind == "tool_result_changed"
    assert "is_error" in spans[0].summary


def test_span_diff_stop_reason_flip() -> None:
    b = _resp_blocks([{"type": "text", "text": "hi"}], stop_reason="end_turn")
    c = _resp_blocks([{"type": "text", "text": "hi"}], stop_reason="content_filter")
    spans = span_diff(b, c)
    stop_changes = [s for s in spans if s.kind == "stop_reason_changed"]
    assert len(stop_changes) == 1
    assert "end_turn" in stop_changes[0].summary
    assert "content_filter" in stop_changes[0].summary


def test_span_diff_block_type_swap() -> None:
    """Block #0 type swap text → tool_use."""
    b = _resp_blocks([{"type": "text", "text": "done"}])
    c = _resp_blocks(
        [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
        ]
    )
    spans = span_diff(b, c)
    assert any(s.kind == "block_type_changed" for s in spans)


def test_span_diff_long_list_inserted_block_doesnt_cascade() -> None:
    """The real-world case greedy alignment got wrong: agent turn has
    20 tool calls; candidate adds ONE new tool call at position 5.
    Greedy would report block #5 as changed AND every block after as
    `block_type_changed`. NW alignment should report exactly 1
    tool_use_added and zero other changes.
    """
    baseline_blocks = [
        {"type": "tool_use", "id": f"t{i}", "name": f"tool_{i}", "input": {"arg": i}}
        for i in range(20)
    ]
    # Candidate inserts an extra tool at position 5.
    candidate_blocks = (
        baseline_blocks[:5]
        + [{"type": "tool_use", "id": "NEW", "name": "injected", "input": {"x": 1}}]
        + baseline_blocks[5:]
    )
    b = _resp_blocks(baseline_blocks)
    c = _resp_blocks(candidate_blocks)
    spans = span_diff(b, c)
    # Exactly one tool_use_added; no other real changes.
    adds = [s for s in spans if s.kind == "tool_use_added"]
    type_changes = [s for s in spans if s.kind == "block_type_changed"]
    assert len(adds) == 1, f"expected 1 add, got {len(adds)}; spans: {spans}"
    assert (
        len(type_changes) == 0
    ), f"NW alignment should not cascade into type_changed spans; got {type_changes}"
    assert adds[0].candidate["name"] == "injected"


def test_span_diff_long_list_removed_block_doesnt_cascade() -> None:
    """Mirror of the added-block test — removing a block in the middle
    of a long list should report exactly 1 tool_use_removed, no cascade."""
    baseline_blocks = [
        {"type": "tool_use", "id": f"t{i}", "name": f"tool_{i}", "input": {}} for i in range(20)
    ]
    candidate_blocks = baseline_blocks[:10] + baseline_blocks[11:]  # drop #10
    b = _resp_blocks(baseline_blocks)
    c = _resp_blocks(candidate_blocks)
    spans = span_diff(b, c)
    removes = [s for s in spans if s.kind == "tool_use_removed"]
    type_changes = [s for s in spans if s.kind == "block_type_changed"]
    assert len(removes) == 1
    assert len(type_changes) == 0


def test_span_diff_preserves_block_indices() -> None:
    """The index in the report must match the position in the content
    list — so downstream UIs can click-back."""
    b = _resp_blocks(
        [
            {"type": "text", "text": "same"},
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
        ]
    )
    c = _resp_blocks(
        [
            {"type": "text", "text": "same"},
            {"type": "tool_use", "id": "t1", "name": "b", "input": {}},
        ]
    )
    spans = span_diff(b, c)
    assert spans[0].block_index == 1  # index #0 unchanged, #1 is the diff


# ---- rendering ---------------------------------------------------------


def test_render_session_summary_formats_each_session() -> None:
    diffs = [
        SessionDiff(
            session_index=0,
            baseline_session_id="a",
            candidate_session_id="a",
            pair_count=3,
            worst_severity="severe",
            report={},
        ),
        SessionDiff(
            session_index=1,
            baseline_session_id="b",
            candidate_session_id="b",
            pair_count=2,
            worst_severity="none",
            report={},
        ),
    ]
    out = render_session_summary(diffs)
    assert "session #0" in out and "severe" in out
    assert "session #1" in out and "none" in out


def test_render_spans_empty_input_is_empty() -> None:
    assert render_spans([]) == ""


def test_render_spans_groups_per_line() -> None:
    spans = [
        SpanDiff(
            kind="text_block_changed", block_index=0, baseline={}, candidate={}, summary="text ..."
        ),
        SpanDiff(
            kind="tool_use_added",
            block_index=1,
            baseline=None,
            candidate={},
            summary="tool_use `x` added",
        ),
    ]
    out = render_spans(spans)
    assert "(2)" in out  # count header
    assert "text ..." in out
    assert "tool_use `x` added" in out
