"""Tests for shadow.sdk.Session."""

from __future__ import annotations

from pathlib import Path

from shadow import _core
from shadow.sdk import Session


def test_session_writes_agentlog_with_metadata_root(tmp_path: Path) -> None:
    out = tmp_path / "session.agentlog"
    with Session(output_path=out, tags={"env": "test"}, session_tag="unit-test") as session:
        session.record_chat(
            request={"model": "claude-opus-4-7", "messages": [], "params": {}},
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    assert out.exists()
    records = _core.parse_agentlog(out.read_bytes())
    assert len(records) == 3  # metadata, chat_request, chat_response
    assert records[0]["kind"] == "metadata"
    assert records[0]["parent"] is None
    assert records[1]["kind"] == "chat_request"
    assert records[1]["parent"] == records[0]["id"]
    assert records[2]["kind"] == "chat_response"
    assert records[2]["parent"] == records[1]["id"]
    # session_tag is in envelope meta per the Session contract.
    assert records[0].get("meta", {}).get("session_tag") == "unit-test"


def test_session_redacts_by_default(tmp_path: Path) -> None:
    out = tmp_path / "session.agentlog"
    with Session(output_path=out) as session:
        session.record_chat(
            request={
                "model": "x",
                "messages": [{"role": "user", "content": "alice@example.com please help"}],
                "params": {},
            },
            response={
                "model": "x",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    records = _core.parse_agentlog(out.read_bytes())
    req = records[1]
    user_msg = req["payload"]["messages"][0]["content"]
    assert "alice@example.com" not in user_msg
    assert "[REDACTED:email]" in user_msg
    # The redacted stamp was placed on the request record's meta.
    assert req.get("meta", {}).get("redacted") is True


def test_session_tool_call_records_link_to_response(tmp_path: Path) -> None:
    out = tmp_path / "session.agentlog"
    with Session(output_path=out) as session:
        _req_id, resp_id = session.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [],
                "stop_reason": "tool_use",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
        tool_call_id = session.record_tool_call(
            "search_files", "toolu_01", {"query": "*.py"}, parent_id=resp_id
        )
        session.record_tool_result(
            "toolu_01", "hello.py\nworld.py", is_error=False, latency_ms=2, parent_id=tool_call_id
        )
    records = _core.parse_agentlog(out.read_bytes())
    # metadata, chat_request, chat_response, tool_call, tool_result
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
        "tool_call",
        "tool_result",
    ]


def test_two_tagless_sessions_get_distinct_trace_ids_in_diff_report(
    tmp_path: Path,
) -> None:
    """Regression: before v3.0.5 the diff report's `baseline_trace_id` and
    `candidate_trace_id` came from the first record's content hash,
    which collides whenever two `Session()` calls produce byte-identical
    metadata payloads. That was the case for any default-tagless run
    pair — the metadata payload `{"sdk": {"name": "shadow", ...}}` is
    deterministic.

    The Python `Session` does mint a unique 128-bit hex `trace_id` per
    instance and stamps it on every record's envelope `meta.trace_id`,
    but envelope meta is intentionally not part of the content hash
    (SPEC §6). The fix in `crates/shadow-core/src/diff/mod.rs` changed
    the diff report to prefer envelope `meta.trace_id` over the first
    record's content hash.

    This test exercises the full Python path: two real Sessions with
    no tags, write to disk, parse back, run `compute_diff_report`, and
    assert the two reported trace ids differ.
    """
    baseline_path = tmp_path / "baseline.agentlog"
    candidate_path = tmp_path / "candidate.agentlog"

    with Session(output_path=baseline_path) as s:
        s.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"input_tokens": 5, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    with Session(output_path=candidate_path) as s:
        s.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": "hello"}],  # same response
                "stop_reason": "end_turn",
                "latency_ms": 200,  # different latency to make the diff non-trivial
                "usage": {"input_tokens": 5, "output_tokens": 1, "thinking_tokens": 0},
            },
        )

    baseline_records = _core.parse_agentlog(baseline_path.read_bytes())
    candidate_records = _core.parse_agentlog(candidate_path.read_bytes())

    # Sanity: the metadata payloads are byte-identical (no tags
    # distinguish them), so first-record content ids would collide if
    # the diff fell back to that path.
    assert baseline_records[0]["id"] == candidate_records[0]["id"]
    # But the envelopes carry distinct meta.trace_id values that the
    # Session minted.
    assert baseline_records[0]["meta"]["trace_id"] != candidate_records[0]["meta"]["trace_id"]

    report = _core.compute_diff_report(baseline_records, candidate_records, None, 42)

    assert report["baseline_trace_id"] != report["candidate_trace_id"], (
        "Two Sessions without tags produced colliding trace_ids — "
        f"baseline={report['baseline_trace_id']}, "
        f"candidate={report['candidate_trace_id']}"
    )
    # Each reported trace_id should match the envelope trace_id of the
    # corresponding metadata record, not the content hash.
    assert report["baseline_trace_id"] == baseline_records[0]["meta"]["trace_id"]
    assert report["candidate_trace_id"] == candidate_records[0]["meta"]["trace_id"]
