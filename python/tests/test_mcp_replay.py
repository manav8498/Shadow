"""Tests for ``shadow.mcp_replay`` — protocol-level MCP replay."""

from __future__ import annotations

import pytest

from shadow.mcp_replay import (
    MCPCall,
    MCPCallNotRecorded,
    MCPServerError,
    RecordingIndex,
    ReplayClientSession,
    canonicalize_params,
    index_from_imported_mcp_records,
)

# ---- canonicalize_params ---------------------------------------------


def test_canonicalize_is_key_order_independent() -> None:
    a = canonicalize_params({"name": "lookup_order", "arguments": {"q": "rust", "limit": 10}})
    b = canonicalize_params({"arguments": {"limit": 10, "q": "rust"}, "name": "lookup_order"})
    assert a == b


def test_canonicalize_handles_unicode() -> None:
    out = canonicalize_params({"uri": "file:///tmp/résumé.pdf"})
    assert "résumé" in out  # ensure_ascii=False — survives non-ASCII URIs


# ---- RecordingIndex --------------------------------------------------


def test_lookup_returns_recorded_response() -> None:
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call",
                params={"name": "lookup_order", "arguments": {"q": "rust"}},
                response={"result": "rust source"},
            )
        ]
    )
    out = idx.lookup("tools/call", {"name": "lookup_order", "arguments": {"q": "rust"}})
    assert out is not None
    assert out.response == {"result": "rust source"}


def test_lookup_returns_responses_in_recorded_order_for_repeated_calls() -> None:
    """Repeated calls with identical method+params yield recorded
    responses in original order — preserves "the second list_tools
    came back with one fewer tool" behaviour."""
    idx = RecordingIndex(
        calls=[
            MCPCall(method="tools/list", params={}, response={"tools": ["a", "b", "c"]}),
            MCPCall(method="tools/list", params={}, response={"tools": ["a", "b"]}),
        ]
    )
    first = idx.lookup("tools/list", {})
    second = idx.lookup("tools/list", {})
    third = idx.lookup("tools/list", {})  # past end → fall back to last recorded
    assert first is not None
    assert second is not None
    assert third is not None
    assert first.response == {"tools": ["a", "b", "c"]}
    assert second.response == {"tools": ["a", "b"]}
    assert third.response == {"tools": ["a", "b"]}  # fallback to last


def test_lookup_misses_when_method_not_recorded() -> None:
    idx = RecordingIndex(calls=[MCPCall(method="tools/call", params={"name": "x"}, response="ok")])
    assert idx.lookup("resources/read", {"uri": "file:///x"}) is None


def test_lookup_distinguishes_different_args() -> None:
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call",
                params={"name": "lookup_order", "arguments": {"q": "rust"}},
                response="rust",
            ),
            MCPCall(
                method="tools/call",
                params={"name": "lookup_order", "arguments": {"q": "python"}},
                response="python",
            ),
        ]
    )
    a = idx.lookup("tools/call", {"name": "lookup_order", "arguments": {"q": "rust"}})
    b = idx.lookup("tools/call", {"name": "lookup_order", "arguments": {"q": "python"}})
    assert a is not None and a.response == "rust"
    assert b is not None and b.response == "python"


def test_unconsumed_keys_reports_skipped_recordings() -> None:
    idx = RecordingIndex(
        calls=[
            MCPCall(method="tools/call", params={"name": "a"}, response="A"),
            MCPCall(method="tools/call", params={"name": "b"}, response="B"),
        ]
    )
    idx.lookup("tools/call", {"name": "a"})
    unconsumed = idx.unconsumed_keys()
    keys = {k[0] for k in unconsumed}
    assert keys == {"tools/call"}  # `b` was never looked up


# ---- ReplayClientSession --------------------------------------------


def test_replay_session_call_tool_returns_recorded_output() -> None:
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call",
                params={"name": "lookup_order", "arguments": {"q": "rust"}},
                response={"matches": ["doc-1", "doc-2"]},
            )
        ]
    )
    sess = ReplayClientSession(idx)
    out = sess.call_tool("lookup_order", {"q": "rust"})
    assert out == {"matches": ["doc-1", "doc-2"]}


def test_replay_session_call_tool_strict_raises_on_miss() -> None:
    idx = RecordingIndex(calls=[])
    sess = ReplayClientSession(idx, strict=True)
    with pytest.raises(MCPCallNotRecorded, match="tools/call"):
        sess.call_tool("missing_tool", {})


def test_replay_session_call_tool_non_strict_returns_none_on_miss() -> None:
    idx = RecordingIndex(calls=[])
    sess = ReplayClientSession(idx, strict=False)
    assert sess.call_tool("missing_tool", {}) is None


def test_replay_session_strict_raises_on_overflow() -> None:
    """Strict mode treats over-consumption as drift, not a quiet
    last-response replay. A candidate that calls the same tool more
    times than the baseline should fail loudly under strict."""
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call",
                params={"name": "fetch", "arguments": {}},
                response="first",
            ),
        ]
    )
    sess = ReplayClientSession(idx, strict=True)
    assert sess.call_tool("fetch", {}) == "first"
    with pytest.raises(MCPCallNotRecorded, match="tools/call"):
        sess.call_tool("fetch", {})


def test_replay_session_non_strict_overflow_reuses_last_response() -> None:
    """Non-strict preserves the historical permissive behaviour: the
    last recorded response is replayed for over-consumed calls."""
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call",
                params={"name": "fetch", "arguments": {}},
                response="first",
            ),
        ]
    )
    sess = ReplayClientSession(idx, strict=False)
    assert sess.call_tool("fetch", {}) == "first"
    assert sess.call_tool("fetch", {}) == "first"


def test_replay_session_raises_when_recording_carries_error() -> None:
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call",
                params={"name": "broken", "arguments": {}},
                error={"code": -32000, "message": "tool went bang"},
            )
        ]
    )
    sess = ReplayClientSession(idx)
    with pytest.raises(MCPServerError, match="tool went bang"):
        sess.call_tool("broken", {})


def test_replay_session_read_resource_keys_on_uri() -> None:
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="resources/read",
                params={"uri": "file:///tmp/notes.md"},
                response="markdown content",
            )
        ]
    )
    sess = ReplayClientSession(idx)
    assert sess.read_resource("file:///tmp/notes.md") == "markdown content"


def test_replay_session_initialize_returns_stub_when_not_recorded() -> None:
    sess = ReplayClientSession(RecordingIndex(calls=[]))
    out = sess.initialize()
    assert out["protocolVersion"]
    assert out["serverInfo"]["name"] == "replay-stub"


def test_replay_session_initialize_replays_recording_when_present() -> None:
    custom = {
        "protocolVersion": "1999",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "real-server"},
    }
    idx = RecordingIndex(calls=[MCPCall(method="initialize", params={}, response=custom)])
    sess = ReplayClientSession(idx)
    assert sess.initialize() == custom


def test_replay_session_async_methods() -> None:
    import asyncio

    idx = RecordingIndex(
        calls=[
            MCPCall(method="tools/call", params={"name": "x", "arguments": {}}, response="ok"),
            MCPCall(method="resources/read", params={"uri": "x://"}, response="text"),
        ]
    )
    sess = ReplayClientSession(idx)
    assert asyncio.run(sess.async_call_tool("x", {})) == "ok"
    assert asyncio.run(sess.async_read_resource("x://")) == "text"


# ---- index_from_imported_mcp_records ---------------------------------


def test_index_from_records_pulls_tool_call_pairs() -> None:
    records = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
        {
            "kind": "tool_call",
            "id": "sha256:tc",
            "ts": "t",
            "parent": "sha256:m",
            "payload": {
                "tool_name": "lookup_order",
                "tool_call_id": "t1",
                "arguments": {"q": "rust"},
            },
        },
        {
            "kind": "tool_result",
            "id": "sha256:tr",
            "ts": "t",
            "parent": "sha256:tc",
            "payload": {"tool_call_id": "t1", "output": "rust source", "is_error": False},
        },
    ]
    idx = index_from_imported_mcp_records(records)
    out = idx.lookup("tools/call", {"name": "lookup_order", "arguments": {"q": "rust"}})
    assert out is not None
    assert out.response == "rust source"


def test_index_from_records_marks_errors() -> None:
    records = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
        {
            "kind": "tool_call",
            "id": "sha256:tc",
            "ts": "t",
            "parent": "sha256:m",
            "payload": {"tool_name": "x", "tool_call_id": "t1", "arguments": {}},
        },
        {
            "kind": "tool_result",
            "id": "sha256:tr",
            "ts": "t",
            "parent": "sha256:tc",
            "payload": {"tool_call_id": "t1", "output": "boom", "is_error": True},
        },
    ]
    idx = index_from_imported_mcp_records(records)
    out = idx.lookup("tools/call", {"name": "x", "arguments": {}})
    assert out is not None
    assert out.response is None
    assert out.error is not None
    assert "boom" in out.error["message"]


def test_index_from_records_picks_up_extra_mcp_calls_in_metadata() -> None:
    records = [
        {
            "kind": "metadata",
            "id": "sha256:m",
            "ts": "t",
            "parent": None,
            "payload": {
                "mcp": {
                    "calls": [
                        {
                            "method": "resources/read",
                            "params": {"uri": "file:///tmp/x.md"},
                            "result": "content",
                        }
                    ]
                }
            },
        }
    ]
    idx = index_from_imported_mcp_records(records)
    out = idx.lookup("resources/read", {"uri": "file:///tmp/x.md"})
    assert out is not None
    assert out.response == "content"


def test_replay_session_identifies_unicode_uri() -> None:
    """Non-ASCII resource URIs round-trip through canonicalize_params."""
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="resources/read",
                params={"uri": "file:///tmp/résumé.pdf"},
                response=b"bytes" if False else "fake-content",
            )
        ]
    )
    sess = ReplayClientSession(idx)
    assert sess.read_resource("file:///tmp/résumé.pdf") == "fake-content"
