"""Tests for shadow.llm.AnthropicLLM and shadow.llm.OpenAILLM.

No real API calls — we stub the provider SDK clients with unittest.mock
so we can verify the request conversion and response conversion paths
without credentials or network.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _mk_anthropic_response(
    text: str = "hello",
    stop: str = "end_turn",
    input_tokens: int = 12,
    output_tokens: int = 3,
) -> MagicMock:
    content_part = MagicMock()
    content_part.type = "text"
    content_part.text = text
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp = MagicMock()
    resp.model = "claude-opus-4-7"
    resp.content = [content_part]
    resp.stop_reason = stop
    resp.usage = usage
    return resp


def _install_anthropic_stub(monkeypatch: pytest.MonkeyPatch, response: MagicMock) -> MagicMock:
    """Install a fake `anthropic` module with an `AsyncAnthropic` client."""
    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=response)
    fake_module = types.ModuleType("anthropic")
    fake_module.AsyncAnthropic = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
    return fake_client


def test_anthropic_backend_converts_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _mk_anthropic_response(text="echo", input_tokens=50, output_tokens=10)
    fake_client = _install_anthropic_stub(monkeypatch, response)

    from shadow.llm.anthropic_backend import AnthropicLLM

    backend = AnthropicLLM(api_key="fake")
    request = {
        "model": "claude-opus-4-7",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "params": {"temperature": 0.2, "max_tokens": 256, "top_p": 0.9},
        "tools": [{"name": "search", "description": "x", "input_schema": {}}],
    }
    out = asyncio.run(backend.complete(request))

    # Response conversion
    assert out["model"] == "claude-opus-4-7"
    assert out["stop_reason"] == "end_turn"
    assert out["content"][0] == {"type": "text", "text": "echo"}
    assert out["usage"]["input_tokens"] == 50
    assert out["usage"]["output_tokens"] == 10

    # Verify the request we sent to anthropic was shaped correctly
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-7"
    assert call_kwargs["system"] == "be brief"  # system pulled out
    # Messages after system-pull should start at user
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs["max_tokens"] == 256
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["tools"] == request["tools"]


def test_anthropic_backend_handles_tool_use_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_part = MagicMock()
    tool_part.type = "tool_use"
    tool_part.id = "toolu_1"
    tool_part.name = "search"
    tool_part.input = {"query": "abc"}
    text_part = MagicMock()
    text_part.type = "text"
    text_part.text = "looking this up"
    usage = MagicMock(input_tokens=10, output_tokens=5)
    response = MagicMock(
        model="claude-opus-4-7",
        content=[text_part, tool_part],
        stop_reason="tool_use",
        usage=usage,
    )
    _install_anthropic_stub(monkeypatch, response)

    from shadow.llm.anthropic_backend import AnthropicLLM

    backend = AnthropicLLM(api_key="fake")
    out = asyncio.run(backend.complete({"model": "claude-opus-4-7", "messages": [], "params": {}}))
    types_ = [p["type"] for p in out["content"]]
    assert types_ == ["text", "tool_use"]
    assert out["content"][1]["name"] == "search"
    assert out["content"][1]["input"] == {"query": "abc"}


def test_anthropic_backend_missing_sdk_raises_shadow_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Remove any cached anthropic module and block future imports
    sys_mod = __import__("sys").modules
    sys_mod.pop("anthropic", None)
    sys_mod.pop("shadow.llm.anthropic_backend", None)

    import builtins

    real_import = builtins.__import__

    def blocking(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking)
    from shadow.errors import ShadowBackendError

    with pytest.raises(ShadowBackendError, match="anthropic SDK not installed"):
        from shadow.llm.anthropic_backend import AnthropicLLM

        AnthropicLLM()


# --------- OpenAI backend ---------------------------------------------------


def _install_openai_stub(monkeypatch: pytest.MonkeyPatch, response: Any) -> MagicMock:
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=response)
    fake_module = types.ModuleType("openai")
    fake_module.AsyncOpenAI = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    return fake_client


def test_openai_backend_converts_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Response object shape per the openai SDK
    message = MagicMock()
    message.content = "pong"
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 20
    usage.completion_tokens = 4
    usage.completion_tokens_details = MagicMock(reasoning_tokens=0)
    response = MagicMock(model="gpt-5", choices=[choice], usage=usage)

    fake_client = _install_openai_stub(monkeypatch, response)

    from shadow.llm.openai_backend import OpenAILLM

    backend = OpenAILLM(api_key="fake")
    request = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "ping"},
        ],
        "params": {"temperature": 0.5, "max_tokens": 64},
        "tools": [
            {
                "name": "search",
                "description": "search",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ],
    }
    out = asyncio.run(backend.complete(request))

    assert out["model"] == "gpt-5"
    assert out["stop_reason"] == "end_turn"  # "stop" → "end_turn"
    assert out["content"][0] == {"type": "text", "text": "pong"}
    assert out["usage"]["input_tokens"] == 20
    assert out["usage"]["output_tokens"] == 4

    # OpenAI kwargs
    call = fake_client.chat.completions.create.call_args.kwargs
    assert call["model"] == "gpt-5"
    assert call["max_completion_tokens"] == 64
    assert call["temperature"] == 0.5
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    # Tool gets wrapped in OpenAI's function-tool shape
    assert call["tools"][0]["type"] == "function"
    assert call["tools"][0]["function"]["name"] == "search"


def test_openai_backend_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tc_function = MagicMock()
    tc_function.name = "search"
    tc_function.arguments = '{"q": "hello"}'
    tc = MagicMock(id="call_1", function=tc_function)
    message = MagicMock(content=None, tool_calls=[tc])
    choice = MagicMock(message=message, finish_reason="tool_calls")
    usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=5,
        completion_tokens_details=MagicMock(reasoning_tokens=0),
    )
    response = MagicMock(model="gpt-5", choices=[choice], usage=usage)
    _install_openai_stub(monkeypatch, response)

    from shadow.llm.openai_backend import OpenAILLM

    out = asyncio.run(
        OpenAILLM(api_key="k").complete({"model": "gpt-5", "messages": [], "params": {}})
    )
    assert out["stop_reason"] == "tool_use"  # "tool_calls" → "tool_use"
    assert out["content"][0]["type"] == "tool_use"
    assert out["content"][0]["name"] == "search"
    assert out["content"][0]["input"] == {"q": "hello"}
