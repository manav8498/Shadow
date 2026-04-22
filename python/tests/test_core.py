"""Tests for the `shadow._core` PyO3 extension."""

from __future__ import annotations

from typing import Any

from shadow import SPEC_VERSION, _core


def test_spec_vector_roundtrips() -> None:
    payload = {"hello": "world"}
    assert _core.canonical_bytes(payload) == b'{"hello":"world"}'
    assert _core.content_id(payload) == (
        "sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588"
    )


def test_spec_version_is_exposed() -> None:
    assert SPEC_VERSION == "0.1"
    assert _core.SPEC_VERSION == "0.1"


def test_parse_write_roundtrip_preserves_ids() -> None:
    # Build a tiny trace: metadata → chat_request → chat_response.
    meta_payload: dict[str, Any] = {"sdk": {"name": "shadow", "version": "0.1.0"}}
    meta_id = _core.content_id(meta_payload)
    req_payload: dict[str, Any] = {"model": "x", "messages": [], "params": {}}
    req_id = _core.content_id(req_payload)
    resp_payload: dict[str, Any] = {
        "model": "x",
        "content": [{"text": "Hi", "type": "text"}],
        "stop_reason": "end_turn",
        "latency_ms": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }
    resp_id = _core.content_id(resp_payload)

    trace: list[dict[str, Any]] = [
        {
            "version": "0.1",
            "id": meta_id,
            "kind": "metadata",
            "ts": "2026-04-21T10:00:00Z",
            "parent": None,
            "payload": meta_payload,
        },
        {
            "version": "0.1",
            "id": req_id,
            "kind": "chat_request",
            "ts": "2026-04-21T10:00:00.100Z",
            "parent": meta_id,
            "payload": req_payload,
        },
        {
            "version": "0.1",
            "id": resp_id,
            "kind": "chat_response",
            "ts": "2026-04-21T10:00:00.500Z",
            "parent": req_id,
            "payload": resp_payload,
        },
    ]

    wire = _core.write_agentlog(trace)
    assert wire.count(b"\n") == 3
    back = _core.parse_agentlog(wire)
    assert len(back) == 3
    assert back[0]["id"] == meta_id
    assert back[2]["kind"] == "chat_response"


def test_parse_rejects_malformed_bytes() -> None:
    import pytest

    with pytest.raises(ValueError):
        _core.parse_agentlog(b"not-valid-json\n")


def test_compute_diff_report_returns_nine_axes() -> None:
    # Build two tiny traces with different latencies.
    def make_trace(latency: int) -> list[dict[str, Any]]:
        meta_payload = {"sdk": {"name": "shadow", "version": "0.1.0"}}
        meta_id = _core.content_id(meta_payload)
        req_payload = {"model": "x", "messages": [], "params": {}}
        req_id = _core.content_id(req_payload)
        resp_payload = {
            "model": "x",
            "content": [{"text": "hi", "type": "text"}],
            "stop_reason": "end_turn",
            "latency_ms": latency,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        }
        resp_id = _core.content_id(resp_payload)
        return [
            {
                "version": "0.1",
                "id": meta_id,
                "kind": "metadata",
                "ts": "2026-04-21T10:00:00Z",
                "parent": None,
                "payload": meta_payload,
            },
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": "2026-04-21T10:00:00.100Z",
                "parent": meta_id,
                "payload": req_payload,
            },
            {
                "version": "0.1",
                "id": resp_id,
                "kind": "chat_response",
                "ts": "2026-04-21T10:00:00.500Z",
                "parent": req_id,
                "payload": resp_payload,
            },
        ]

    baseline = make_trace(100)
    candidate = make_trace(200)
    report = _core.compute_diff_report(baseline, candidate, None, 42)
    assert len(report["rows"]) == 9
    # Latency axis should have a positive delta.
    latency_row = next(row for row in report["rows"] if row["axis"] == "latency")
    assert latency_row["delta"] > 0
