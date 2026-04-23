"""Tests for streaming auto-instrumentation.

Fake OpenAI and Anthropic stream shapes → verify that after the consumer
finishes iterating, a chat_request + chat_response pair lands in the
.agentlog with the aggregated content.
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

# ---------------------------------------------------------------------------
# Fake OpenAI streaming: a chunk has .choices[0].delta.content.
# ---------------------------------------------------------------------------


class _OpenAIDelta:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _OpenAIChoice:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.delta = _OpenAIDelta(content=content)
        self.finish_reason = finish_reason


class _OpenAIChunk:
    object = "chat.completion.chunk"

    def __init__(
        self, content: str | None = None, finish_reason: str | None = None, model: str = "gpt-test"
    ) -> None:
        self.model = model
        self.choices = [_OpenAIChoice(content=content, finish_reason=finish_reason)]
        self.usage = None


def _install_fake_openai_streaming() -> tuple[type, type]:
    class Completions:
        def create(self, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return iter(
                    [
                        _OpenAIChunk(content="hello "),
                        _OpenAIChunk(content="world"),
                        _OpenAIChunk(content=None, finish_reason="stop"),
                    ]
                )
            return _OpenAIChunk(content="non-stream")

    class AsyncCompletions:
        async def create(self, **kwargs: Any) -> Any:
            async def agen() -> Any:
                for ch in [
                    _OpenAIChunk(content="hi "),
                    _OpenAIChunk(content="there"),
                    _OpenAIChunk(content=None, finish_reason="stop"),
                ]:
                    yield ch

            if kwargs.get("stream"):
                return agen()
            return _OpenAIChunk(content="non-stream")

    openai_pkg = types.ModuleType("openai")
    openai_resources = types.ModuleType("openai.resources")
    openai_chat = types.ModuleType("openai.resources.chat")
    openai_completions = types.ModuleType("openai.resources.chat.completions")
    openai_completions.Completions = Completions  # type: ignore[attr-defined]
    openai_completions.AsyncCompletions = AsyncCompletions  # type: ignore[attr-defined]
    openai_chat.completions = openai_completions  # type: ignore[attr-defined]
    openai_resources.chat = openai_chat  # type: ignore[attr-defined]
    openai_pkg.resources = openai_resources  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_pkg
    sys.modules["openai.resources"] = openai_resources
    sys.modules["openai.resources.chat"] = openai_chat
    sys.modules["openai.resources.chat.completions"] = openai_completions
    return Completions, AsyncCompletions


# ---------------------------------------------------------------------------
# Fake Anthropic streaming: event.type ∈ {message_start, content_block_start,
#   content_block_delta, content_block_stop, message_delta, message_stop}.
# ---------------------------------------------------------------------------


class _Event:
    def __init__(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)


def _install_fake_anthropic_streaming() -> tuple[type, type]:
    events = [
        _Event(
            type="message_start",
            message=_Event(
                model="claude-test",
                usage=_Event(input_tokens=10),
            ),
        ),
        _Event(type="content_block_start", index=0, content_block=_Event(type="text")),
        _Event(type="content_block_delta", index=0, delta=_Event(text="hello ")),
        _Event(type="content_block_delta", index=0, delta=_Event(text="there")),
        _Event(type="content_block_stop", index=0),
        _Event(
            type="message_delta",
            delta=_Event(stop_reason="end_turn"),
            usage=_Event(output_tokens=2),
        ),
        _Event(type="message_stop"),
    ]

    class _Message:
        model = "claude-test"
        content: typing.ClassVar[list[Any]] = []
        stop_reason = "end_turn"
        usage: typing.ClassVar[_Event] = _Event(
            input_tokens=10, output_tokens=2, cache_read_input_tokens=0
        )

    class Messages:
        def create(self, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return iter(events)
            return _Message()

    class AsyncMessages:
        async def create(self, **kwargs: Any) -> Any:
            async def agen() -> Any:
                for e in events:
                    yield e

            if kwargs.get("stream"):
                return agen()
            return _Message()

    anthropic_pkg = types.ModuleType("anthropic")
    anthropic_resources = types.ModuleType("anthropic.resources")
    anthropic_messages = types.ModuleType("anthropic.resources.messages")
    anthropic_messages.Messages = Messages  # type: ignore[attr-defined]
    anthropic_messages.AsyncMessages = AsyncMessages  # type: ignore[attr-defined]
    anthropic_resources.messages = anthropic_messages  # type: ignore[attr-defined]
    anthropic_pkg.resources = anthropic_resources  # type: ignore[attr-defined]
    sys.modules["anthropic"] = anthropic_pkg
    sys.modules["anthropic.resources"] = anthropic_resources
    sys.modules["anthropic.resources.messages"] = anthropic_messages
    return Messages, AsyncMessages


def _cleanup(names: list[str]) -> None:
    for n in names:
        sys.modules.pop(n, None)


def test_openai_sync_stream_records_aggregated_response(tmp_path: Path) -> None:
    completions_cls, _ = _install_fake_openai_streaming()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = completions_cls()
            stream = client.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            # Consumer drains the stream like a real user.
            list(stream)
    finally:
        _cleanup(
            [
                "openai",
                "openai.resources",
                "openai.resources.chat",
                "openai.resources.chat.completions",
            ]
        )

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    resp_payload = records[2]["payload"]
    assert resp_payload["content"][0]["text"] == "hello world"
    assert resp_payload["stop_reason"] == "end_turn"


def test_openai_async_stream_records_aggregated_response(tmp_path: Path) -> None:
    _, async_cls = _install_fake_openai_streaming()
    out = tmp_path / "t.agentlog"

    async def run() -> None:
        with Session(output_path=out):
            client = async_cls()
            stream = await client.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for _ in stream:
                pass

    try:
        asyncio.run(run())
    finally:
        _cleanup(
            [
                "openai",
                "openai.resources",
                "openai.resources.chat",
                "openai.resources.chat.completions",
            ]
        )

    records = _core.parse_agentlog(out.read_bytes())
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    assert records[2]["payload"]["content"][0]["text"] == "hi there"


def test_anthropic_sync_stream_aggregates_text_blocks(tmp_path: Path) -> None:
    messages_cls, _ = _install_fake_anthropic_streaming()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = messages_cls()
            stream = client.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            list(stream)
    finally:
        _cleanup(
            [
                "anthropic",
                "anthropic.resources",
                "anthropic.resources.messages",
            ]
        )

    records = _core.parse_agentlog(out.read_bytes())
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    payload = records[2]["payload"]
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][0]["text"] == "hello there"
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"]["input_tokens"] == 10
    assert payload["usage"]["output_tokens"] == 2


def test_stream_consumer_partial_iteration_records_when_closed(tmp_path: Path) -> None:
    """If the consumer breaks early and explicitly closes, partial content records."""
    completions_cls, _ = _install_fake_openai_streaming()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = completions_cls()
            stream = client.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            for ch in stream:
                del ch
                break
            # Users who break early should call close() — our finally fires then.
            stream.close()
    finally:
        _cleanup(
            [
                "openai",
                "openai.resources",
                "openai.resources.chat",
                "openai.resources.chat.completions",
            ]
        )

    records = _core.parse_agentlog(out.read_bytes())
    assert any(r["kind"] == "chat_response" for r in records)
