"""Tests for ``shadow.tools.base`` — the protocol + types layer."""

from __future__ import annotations

from shadow.tools.base import (
    ToolBackend,
    ToolCall,
    ToolResult,
    canonical_args_hash,
)

# ---- ToolCall -----------------------------------------------------------


def test_tool_call_signature_hash_is_deterministic() -> None:
    a = ToolCall(id="c1", name="search", arguments={"query": "rust", "limit": 10})
    b = ToolCall(id="c2", name="search", arguments={"limit": 10, "query": "rust"})
    # Different ids must NOT make the signatures differ; the signature
    # is over (name, args) only — the id is provider-specific.
    assert a.signature_hash() == b.signature_hash()


def test_tool_call_signature_distinguishes_tools() -> None:
    a = ToolCall(id="c", name="search", arguments={"q": "x"})
    b = ToolCall(id="c", name="lookup", arguments={"q": "x"})
    assert a.signature_hash() != b.signature_hash()


def test_tool_call_from_block_round_trips() -> None:
    block = {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "rust"}}
    call = ToolCall.from_block(block)
    assert call.id == "t1"
    assert call.name == "search"
    assert call.arguments == {"q": "rust"}


def test_tool_call_from_record_round_trips() -> None:
    record = {
        "kind": "tool_call",
        "payload": {
            "tool_name": "fetch",
            "tool_call_id": "t9",
            "arguments": {"url": "..."},
        },
    }
    call = ToolCall.from_record(record)
    assert call.name == "fetch"
    assert call.id == "t9"
    assert call.arguments == {"url": "..."}


def test_tool_call_from_block_handles_missing_fields() -> None:
    """A malformed tool_use block must coerce gracefully, not raise."""
    call = ToolCall.from_block({"type": "tool_use"})
    assert call.id == ""
    assert call.name == ""
    assert call.arguments == {}


# ---- ToolResult ---------------------------------------------------------


def test_tool_result_to_record_payload_shape() -> None:
    r = ToolResult(tool_call_id="t1", output="ok", is_error=False, latency_ms=5)
    payload = r.to_record_payload()
    assert payload == {
        "tool_call_id": "t1",
        "output": "ok",
        "is_error": False,
        "latency_ms": 5,
    }


# ---- canonical_args_hash ------------------------------------------------


def test_canonical_args_hash_is_stable_across_key_order() -> None:
    a = canonical_args_hash("search", {"q": "x", "limit": 10})
    b = canonical_args_hash("search", {"limit": 10, "q": "x"})
    assert a == b


def test_canonical_args_hash_format() -> None:
    """Hash matches Shadow's content-id convention so it round-trips
    through every Shadow tool that recognises sha256: prefixed ids."""
    h = canonical_args_hash("search", {"q": "x"})
    assert h.startswith("sha256:")
    # sha256 hex is 64 chars.
    assert len(h) == len("sha256:") + 64


# ---- ToolBackend Protocol -----------------------------------------------


def test_tool_backend_runtime_checkable() -> None:
    """A class with execute + id properties must isinstance-match
    the Protocol (runtime_checkable)."""

    class _MockBackend:
        @property
        def id(self) -> str:
            return "mock"

        async def execute(self, call: ToolCall) -> ToolResult:
            return ToolResult(call.id, "stub")

    assert isinstance(_MockBackend(), ToolBackend)
