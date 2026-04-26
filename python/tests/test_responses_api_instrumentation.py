"""Tests for auto-instrumentation of the OpenAI Responses API.

Shadow's v0.1 instrumentor hooked only `openai.resources.chat.completions`.
The Responses API (SDK v1.40+, recommended path for new code) lives at
`openai.resources.responses` and was silently un-instrumented — every
Responses-API call produced no trace.

These tests install a fake `openai.resources.responses` module into
sys.modules, run a Session, make a Responses-shaped call, and verify
the resulting `.agentlog` captures it as a chat_request/chat_response
pair with correct translations.
"""

from __future__ import annotations

import asyncio
import sys
import types
import typing
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.sdk import Session


class _RespUsage:
    def __init__(self) -> None:
        self.input_tokens = 20
        self.output_tokens = 7
        self.output_tokens_details = types.SimpleNamespace(reasoning_tokens=3)
        self.input_tokens_details = types.SimpleNamespace(cached_tokens=10)


class _RespPart:
    def __init__(self, type_: str, text: str) -> None:
        self.type = type_
        self.text = text


class _RespMessage:
    def __init__(self) -> None:
        self.type = "message"
        self.content = [_RespPart("output_text", "hello from responses")]


class _FakeResponse:
    def __init__(self) -> None:
        self.id = "resp_abc123"
        self.model = "gpt-5"
        self.status = "completed"
        self.output = [_RespMessage()]
        self.usage = _RespUsage()


def _install_fake_openai_responses() -> tuple[type, type]:
    class Responses:
        def create(self, **kwargs: Any) -> Any:
            return _FakeResponse()

    class AsyncResponses:
        async def create(self, **kwargs: Any) -> Any:
            return _FakeResponse()

    openai_pkg = types.ModuleType("openai")
    openai_resources = types.ModuleType("openai.resources")
    openai_responses = types.ModuleType("openai.resources.responses")
    openai_responses.Responses = Responses  # type: ignore[attr-defined]
    openai_responses.AsyncResponses = AsyncResponses  # type: ignore[attr-defined]
    openai_resources.responses = openai_responses  # type: ignore[attr-defined]
    openai_pkg.resources = openai_resources  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_pkg
    sys.modules["openai.resources"] = openai_resources
    sys.modules["openai.resources.responses"] = openai_responses
    # Also prevent chat.completions from being picked up (fail soft in the
    # instrumentor) so we can isolate the Responses path.
    return Responses, AsyncResponses


def _cleanup() -> None:
    for n in [
        "openai",
        "openai.resources",
        "openai.resources.responses",
        "openai.resources.chat",
        "openai.resources.chat.completions",
    ]:
        sys.modules.pop(n, None)


def test_responses_api_sync_call_is_recorded(tmp_path: Path) -> None:
    responses_cls, _ = _install_fake_openai_responses()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = responses_cls()
            client.create(
                model="gpt-5",
                input="hello",
                instructions="be concise",
                max_output_tokens=50,
                temperature=0.2,
            )
    finally:
        _cleanup()

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    # Instructions lifted to a system message, input becomes a user message.
    assert req["messages"][0] == {"role": "system", "content": "be concise"}
    assert req["messages"][1] == {"role": "user", "content": "hello"}
    assert req["params"]["max_tokens"] == 50
    assert req["params"]["temperature"] == 0.2
    resp = records[2]["payload"]
    assert resp["model"] == "gpt-5"
    assert resp["content"][0]["text"] == "hello from responses"
    assert resp["stop_reason"] == "end_turn"
    assert resp["usage"]["input_tokens"] == 20
    assert resp["usage"]["output_tokens"] == 7
    assert resp["usage"]["thinking_tokens"] == 3
    assert resp["usage"]["cached_input_tokens"] == 10


def test_responses_api_async_call_is_recorded(tmp_path: Path) -> None:
    _, async_cls = _install_fake_openai_responses()
    out = tmp_path / "t.agentlog"

    async def run() -> None:
        with Session(output_path=out):
            client = async_cls()
            await client.create(model="gpt-5", input="hi")

    try:
        asyncio.run(run())
    finally:
        _cleanup()

    records = _core.parse_agentlog(out.read_bytes())
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_responses_api_input_list_becomes_messages(tmp_path: Path) -> None:
    responses_cls, _ = _install_fake_openai_responses()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = responses_cls()
            client.create(
                model="gpt-5",
                input=[
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                ],
            )
    finally:
        _cleanup()

    records = _core.parse_agentlog(out.read_bytes())
    req = records[1]["payload"]
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_responses_api_function_call_output_translated_to_tool_use(tmp_path: Path) -> None:
    """Model-emitted function_call items in output → tool_use content parts."""

    class _FnCall:
        type = "function_call"
        call_id = "call_xyz"
        name = "search"
        arguments = '{"q": "rust"}'

    class _FnResponse:
        id = "resp_fn"
        model = "gpt-5"
        status = "completed"
        output: typing.ClassVar[list[Any]] = [_FnCall()]
        usage = _RespUsage()

    class Responses:
        def create(self, **kwargs: Any) -> Any:
            return _FnResponse()

    class AsyncResponses:
        async def create(self, **kwargs: Any) -> Any:
            return _FnResponse()

    openai_pkg = types.ModuleType("openai")
    openai_resources = types.ModuleType("openai.resources")
    openai_responses = types.ModuleType("openai.resources.responses")
    openai_responses.Responses = Responses  # type: ignore[attr-defined]
    openai_responses.AsyncResponses = AsyncResponses  # type: ignore[attr-defined]
    openai_resources.responses = openai_responses  # type: ignore[attr-defined]
    openai_pkg.resources = openai_resources  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_pkg
    sys.modules["openai.resources"] = openai_resources
    sys.modules["openai.resources.responses"] = openai_responses

    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = Responses()
            client.create(model="gpt-5", input="find rust docs")
    finally:
        _cleanup()

    records = _core.parse_agentlog(out.read_bytes())
    resp = records[-1]["payload"]
    tool_uses = [c for c in resp["content"] if c.get("type") == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "search"
    assert tool_uses[0]["id"] == "call_xyz"
    assert tool_uses[0]["input"] == {"q": "rust"}


def test_responses_api_tool_schema_translated(tmp_path: Path) -> None:
    """Responses-API tool shape ({type, name, parameters}) → Shadow shape."""
    responses_cls, _ = _install_fake_openai_responses()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = responses_cls()
            client.create(
                model="gpt-5",
                input="hello",
                tools=[
                    {
                        "type": "function",
                        "name": "search",
                        "description": "search docs",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    }
                ],
            )
    finally:
        _cleanup()
    records = _core.parse_agentlog(out.read_bytes())
    req = records[1]["payload"]
    assert req["tools"][0]["name"] == "search"
    assert req["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"


def test_responses_api_omit_sentinels_are_stripped(tmp_path: Path) -> None:
    """Real-world bug from openai-agents dogfood: client.responses.create
    receives openai.Omit / NotGiven sentinels for unset optional params,
    which crashed the canonicaliser ("ValueError: unsupported type Omit").
    The translator now drops them before hashing."""

    class _OmitSentinel:
        """Stand-in for openai.Omit — same name, no special behaviour."""

        def __repr__(self) -> str:  # pragma: no cover — debug aid
            return "Omit()"

    # Ensure the sentinel matches the translator's name-based check.
    _OmitSentinel.__name__ = "Omit"

    responses_cls, _ = _install_fake_openai_responses()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = responses_cls()
            client.create(
                model="gpt-5",
                input="hi",
                instructions=_OmitSentinel(),
                # Nested Omit inside a list and dict — recursive strip.
                tools=[
                    {
                        "type": "function",
                        "name": "noop",
                        "description": _OmitSentinel(),
                        "parameters": {
                            "type": "object",
                            "default": _OmitSentinel(),
                        },
                    }
                ],
                temperature=_OmitSentinel(),
                max_output_tokens=_OmitSentinel(),
            )
    finally:
        _cleanup()
    records = _core.parse_agentlog(out.read_bytes())
    req = records[1]["payload"]
    # Omitted instructions never made it to messages as a system role.
    assert all(m.get("role") != "system" for m in req["messages"])
    # params has no temperature / max_tokens entries (both were Omit).
    assert "temperature" not in req["params"]
    assert "max_tokens" not in req["params"]
    # Tool description didn't get a stringified Omit.
    tool = req["tools"][0]
    assert "description" not in tool or tool["description"] == ""
    assert "default" not in tool["input_schema"]
