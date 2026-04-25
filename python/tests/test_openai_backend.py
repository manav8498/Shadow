"""Tests for the OpenAI backend's wire-shape conversion.

These tests focus on ``_convert_message_to_openai`` — the static
helper that maps Shadow's internal message shape to the shape the
OpenAI Chat Completions API actually accepts on input. They do not
make network calls.

Coverage focus is the multi-turn / tool-using path, which is what
the agent-loop replay engine produces when paired with a real
OpenAI backend.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

from shadow.llm.openai_backend import OpenAILLM  # noqa: E402


def test_assistant_with_string_content_and_tool_calls_forwards_both() -> None:
    """The agent-loop engine emits assistant messages of the form
    ``{role:assistant, content:"", tool_calls:[...]}``. The converter
    must forward ``tool_calls`` to the API; if it drops them, the
    next request fails with 400 "messages with role 'tool' must be
    a response to a preceding message with 'tool_calls'."
    """
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "lookup_order", "arguments": '{"order_id": "1"}'},
            }
        ],
    }
    out = OpenAILLM._convert_message_to_openai(msg)
    assert out["role"] == "assistant"
    assert out["content"] == ""
    assert out["tool_calls"] == msg["tool_calls"]


def test_tool_role_message_forwards_tool_call_id() -> None:
    """``role:tool`` messages must carry ``tool_call_id`` to the API
    or the response fails to thread the call→result pairing."""
    msg = {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": '{"status": "delivered"}',
    }
    out = OpenAILLM._convert_message_to_openai(msg)
    assert out["role"] == "tool"
    assert out["tool_call_id"] == "call_abc"
    assert out["content"] == '{"status": "delivered"}'


def test_plain_user_message_unchanged() -> None:
    msg = {"role": "user", "content": "Hello"}
    assert OpenAILLM._convert_message_to_openai(msg) == msg


def test_assistant_with_only_text_unchanged() -> None:
    msg = {"role": "assistant", "content": "Sure, looking that up."}
    assert OpenAILLM._convert_message_to_openai(msg) == msg


def test_assistant_content_list_with_tool_use_block_routes_to_tool_calls() -> None:
    """The list-of-parts content shape (Anthropic-style) must also
    produce a top-level ``tool_calls`` field for OpenAI."""
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll look that up."},
            {
                "type": "tool_use",
                "id": "call_xyz",
                "name": "lookup_order",
                "input": {"order_id": "ORD-1"},
            },
        ],
    }
    out = OpenAILLM._convert_message_to_openai(msg)
    assert out["role"] == "assistant"
    assert "tool_calls" in out
    assert out["tool_calls"][0]["id"] == "call_xyz"
    assert out["tool_calls"][0]["function"]["name"] == "lookup_order"
