"""Tests for Session auto-instrumentation of anthropic + openai clients.

We install fake SDK modules in `sys.modules` that expose the same
`resources.messages.Messages` / `resources.chat.completions.Completions`
class paths the real SDKs use. The Session monkey-patches `.create` on
those classes; the wrappers record the call into the session.

No network calls; no real SDK installs needed.
"""

from __future__ import annotations

import sys
import types
import typing
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.sdk import Session


def _install_fake_anthropic() -> tuple[type, type]:
    """Create fake anthropic.resources.messages.{Messages,AsyncMessages}."""

    class _Usage:
        def __init__(self, input_tokens: int, output_tokens: int) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cache_read_input_tokens = 0

    class _Part:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Message:
        def __init__(self) -> None:
            self.model = "claude-test"
            self.content = [_Part("hello back")]
            self.stop_reason = "end_turn"
            self.usage = _Usage(10, 3)

    class Messages:
        def create(self, **kwargs: Any) -> _Message:
            return _Message()

    class AsyncMessages:
        async def create(self, **kwargs: Any) -> _Message:
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


def _install_fake_openai() -> tuple[type, type]:
    class _Usage:
        def __init__(self) -> None:
            self.prompt_tokens = 5
            self.completion_tokens = 4
            self.completion_tokens_details = None

    class _Msg:
        content = "hi there"
        tool_calls: typing.ClassVar[list[Any]] = []

    class _Choice:
        message: typing.ClassVar[_Msg] = _Msg()
        finish_reason = "stop"

    class _Completion:
        model = "gpt-test"
        choices: typing.ClassVar[list[_Choice]] = [_Choice()]
        usage: typing.ClassVar[_Usage] = _Usage()

    class Completions:
        def create(self, **kwargs: Any) -> _Completion:
            return _Completion()

    class AsyncCompletions:
        async def create(self, **kwargs: Any) -> _Completion:
            return _Completion()

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


def _cleanup_modules(names: list[str]) -> None:
    for n in names:
        sys.modules.pop(n, None)


def test_auto_instrumentation_records_anthropic_sync_call(tmp_path: Path) -> None:
    messages_cls, _ = _install_fake_anthropic()
    original_create = messages_cls.create
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client_messages = messages_cls()
            client_messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                system="you are helpful",
                messages=[{"role": "user", "content": "hi"}],
            )
        # patch should be undone on __exit__
        assert messages_cls.create is original_create
    finally:
        _cleanup_modules(["anthropic", "anthropic.resources", "anthropic.resources.messages"])

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    # System message gets prepended to messages[] in Shadow shape.
    assert req["messages"][0] == {"role": "system", "content": "you are helpful"}
    assert req["messages"][1]["content"] == "hi"
    assert req["params"]["max_tokens"] == 1024
    resp = records[2]["payload"]
    assert resp["content"][0]["type"] == "text"
    assert resp["content"][0]["text"] == "hello back"
    assert resp["stop_reason"] == "end_turn"


def test_auto_instrumentation_records_openai_sync_call(tmp_path: Path) -> None:
    completions_cls, _ = _install_fake_openai()
    original_create = completions_cls.create
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = completions_cls()
            client.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=256,
                temperature=0.2,
            )
        assert completions_cls.create is original_create
    finally:
        _cleanup_modules(
            [
                "openai",
                "openai.resources",
                "openai.resources.chat",
                "openai.resources.chat.completions",
            ]
        )

    records = _core.parse_agentlog(out.read_bytes())
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["params"]["max_tokens"] == 256
    assert req["params"]["temperature"] == 0.2
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "hi there"
    assert resp["stop_reason"] == "end_turn"


def test_auto_instrumentation_disabled_when_flag_off(tmp_path: Path) -> None:
    messages_cls, _ = _install_fake_anthropic()
    original_create = messages_cls.create
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out, auto_instrument=False):
            pass
        # Without auto-instrument, Messages.create is never patched.
        assert messages_cls.create is original_create
    finally:
        _cleanup_modules(["anthropic", "anthropic.resources", "anthropic.resources.messages"])

    records = _core.parse_agentlog(out.read_bytes())
    # Only the metadata record is present.
    assert [r["kind"] for r in records] == ["metadata"]


def test_auto_instrumentation_passes_stream_calls_through(tmp_path: Path) -> None:
    messages_cls, _ = _install_fake_anthropic()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = messages_cls()
            client.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
    finally:
        _cleanup_modules(["anthropic", "anthropic.resources", "anthropic.resources.messages"])

    records = _core.parse_agentlog(out.read_bytes())
    # Streaming calls should not be recorded.
    assert [r["kind"] for r in records] == ["metadata"]


def test_auto_instrumentation_swallows_recording_errors(tmp_path: Path) -> None:
    """If the translator raises, user's call still returns normally."""
    messages_cls, _ = _install_fake_anthropic()
    out = tmp_path / "t.agentlog"
    try:
        with Session(output_path=out):
            client = messages_cls()
            # Pass a messages shape that would normally crash the translator
            # if it weren't guarded. `messages=None` triggers an error
            # in the translator when iterating.
            result = client.create(model="x", max_tokens=1, messages=None)
            assert result is not None  # user call still succeeds
    finally:
        _cleanup_modules(["anthropic", "anthropic.resources", "anthropic.resources.messages"])
