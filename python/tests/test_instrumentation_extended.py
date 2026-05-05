"""Tests for the extended auto-instrumentation surface added in v3.1.

Covers the four customer-reported gaps:

* LiteLLM module-level patching (`litellm.completion` /
  `litellm.acompletion`).
* LangChain `BaseChatOpenAI._generate` / `_agenerate` patching.
* Sync streaming proxy preserves the context-manager interface.
* Async streaming proxy preserves the async context-manager interface
  (the regression that broke GPT Researcher under shadow record).

The tests use stub fakes rather than real OpenAI / LangChain SDKs so
they run offline. Each fake duck-types exactly the surface the
production wrapper uses.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from shadow.sdk import Session
from shadow.sdk.instrumentation import Instrumentor

# ---- LiteLLM ----------------------------------------------------------------


class _FakeLiteLLMResponse:
    """Duck-type the LiteLLM `ModelResponse` shape (= OpenAI ChatCompletion)."""

    def __init__(self, content: str, tokens_in: int = 7, tokens_out: int = 4) -> None:
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop",
        )
        self.choices = [choice]
        self.model = "gpt-4o-mini"
        self.usage = types.SimpleNamespace(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            total_tokens=tokens_in + tokens_out,
        )


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake `litellm` module so the patcher has something to patch.

    The instrumentor's discovery path is `import litellm`; we install a
    stub into ``sys.modules`` *before* the Session opens, then assert
    after exit that calling `litellm.completion(...)` while the session
    is active records a chat pair.
    """
    fake = types.ModuleType("litellm")

    def completion(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return _FakeLiteLLMResponse(content="hi from litellm")

    async def acompletion(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return _FakeLiteLLMResponse(content="hi from async litellm")

    def text_completion(model: str, prompt: str, **kwargs: Any) -> Any:
        return _FakeLiteLLMResponse(content="text out")

    async def atext_completion(model: str, prompt: str, **kwargs: Any) -> Any:
        return _FakeLiteLLMResponse(content="async text out")

    fake.completion = completion  # type: ignore[attr-defined]
    fake.acompletion = acompletion  # type: ignore[attr-defined]
    fake.text_completion = text_completion  # type: ignore[attr-defined]
    fake.atext_completion = atext_completion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def test_litellm_completion_records_chat_pair(
    tmp_path: Path, fake_litellm: types.ModuleType
) -> None:
    """litellm.completion() under a Session records a chat pair."""
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out) as s:
        result = fake_litellm.completion(  # type: ignore[attr-defined]
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert result.choices[0].message.content == "hi from litellm"
    chat_resps = [r for r in s._records if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1
    assert "hi from litellm" in str(chat_resps[0]["payload"]["content"])


def test_litellm_acompletion_records_chat_pair(
    tmp_path: Path, fake_litellm: types.ModuleType
) -> None:
    """litellm.acompletion() under a Session records a chat pair."""
    out = tmp_path / "trace.agentlog"

    async def _go() -> Any:
        with Session(output_path=out) as s:
            return await fake_litellm.acompletion(  # type: ignore[attr-defined]
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            ), s

    result, session = asyncio.run(_go())
    assert result.choices[0].message.content == "hi from async litellm"
    chat_resps = [r for r in session._records if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1


def test_litellm_request_translator_strips_routing_kwargs(
    tmp_path: Path, fake_litellm: types.ModuleType
) -> None:
    """LiteLLM-specific kwargs (api_key, api_base, etc.) don't leak into the trace."""
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out) as s:
        fake_litellm.completion(  # type: ignore[attr-defined]
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-leaky-secret",
            api_base="https://internal/llm",
            custom_llm_provider="azure",
            metadata={"caller": "test"},
        )
    requests = [r for r in s._records if r["kind"] == "chat_request"]
    assert len(requests) == 1
    payload_str = str(requests[0]["payload"])
    # Loud assertion: none of the routing/security kwargs leak into the payload.
    assert "sk-leaky-secret" not in payload_str
    assert "internal/llm" not in payload_str
    assert "custom_llm_provider" not in payload_str


def test_uninstall_restores_litellm(tmp_path: Path, fake_litellm: types.ModuleType) -> None:
    """After Session exit, `litellm.completion` is the original function again."""
    original = fake_litellm.completion  # type: ignore[attr-defined]
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out):
        # While the session is open, the function is wrapped.
        assert fake_litellm.completion is not original  # type: ignore[attr-defined]
    # After exit, the original is restored.
    assert fake_litellm.completion is original  # type: ignore[attr-defined]


# ---- LangChain ChatOpenAI ---------------------------------------------------


class _BaseMsg:
    def __init__(self, content: str) -> None:
        self.content = content


# Class names match what LangChain's `langchain_core.messages` exports —
# the role detector looks at `type(msg).__name__` so the fakes must be
# named the same.
class HumanMessage(_BaseMsg):
    pass


class AIMessage(_BaseMsg):
    def __init__(self, content: str, usage: dict[str, int] | None = None) -> None:
        super().__init__(content)
        self.tool_calls = []
        self.usage_metadata = usage


class SystemMessage(_BaseMsg):
    pass


class _FakeChatGeneration:
    def __init__(self, message: Any, model: str = "gpt-4o-mini") -> None:
        self.message = message
        self.generation_info = {"model_name": model, "finish_reason": "stop"}


class _FakeChatResult:
    def __init__(self, generations: list[Any]) -> None:
        self.generations = generations
        self.llm_output = {
            "model_name": "gpt-4o-mini",
            "token_usage": {"prompt_tokens": 8, "completion_tokens": 5},
        }


class _FakeBaseChatOpenAI:
    """Tiny stub of langchain_openai.chat_models.base.BaseChatOpenAI."""

    def __init__(self) -> None:
        self.model_name = "gpt-4o-mini"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> _FakeChatResult:
        ai = AIMessage(
            content="response from langchain",
            usage={"input_tokens": 8, "output_tokens": 5},
        )
        return _FakeChatResult([_FakeChatGeneration(ai)])

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> _FakeChatResult:
        ai = AIMessage(
            content="async response from langchain",
            usage={"input_tokens": 9, "output_tokens": 6},
        )
        return _FakeChatResult([_FakeChatGeneration(ai)])


@pytest.fixture
def fake_langchain_openai(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake `langchain_openai.chat_models.base` module.

    Mirrors the import path the patcher uses
    (``from langchain_openai.chat_models.base import BaseChatOpenAI``)
    so the patcher's discovery succeeds against this stub.
    """
    parent = types.ModuleType("langchain_openai")
    chat_models = types.ModuleType("langchain_openai.chat_models")
    base = types.ModuleType("langchain_openai.chat_models.base")
    base.BaseChatOpenAI = _FakeBaseChatOpenAI  # type: ignore[attr-defined]
    chat_models.base = base  # type: ignore[attr-defined]
    parent.chat_models = chat_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", parent)
    monkeypatch.setitem(sys.modules, "langchain_openai.chat_models", chat_models)
    monkeypatch.setitem(sys.modules, "langchain_openai.chat_models.base", base)
    return _FakeBaseChatOpenAI


def test_langchain_chatopenai_generate_records_chat_pair(
    tmp_path: Path, fake_langchain_openai: type
) -> None:
    """ChatOpenAI._generate(...) under a Session records a chat_response."""
    out = tmp_path / "trace.agentlog"
    chat = fake_langchain_openai()
    with Session(output_path=out) as s:
        result = chat._generate(
            messages=[
                SystemMessage("you are helpful"),
                HumanMessage("hi"),
            ]
        )
    assert result.generations[0].message.content == "response from langchain"
    chat_resps = [r for r in s._records if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1
    assert "response from langchain" in str(chat_resps[0]["payload"]["content"])


def test_langchain_chatopenai_agenerate_records_chat_pair(
    tmp_path: Path, fake_langchain_openai: type
) -> None:
    """ChatOpenAI._agenerate(...) under a Session records a chat_response."""
    out = tmp_path / "trace.agentlog"

    async def _go() -> Any:
        chat = fake_langchain_openai()
        with Session(output_path=out) as s:
            r = await chat._agenerate(
                messages=[HumanMessage("hi")],
            )
            return r, s

    result, session = asyncio.run(_go())
    assert result.generations[0].message.content == "async response from langchain"
    chat_resps = [r for r in session._records if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1


def test_langchain_request_carries_user_role(tmp_path: Path, fake_langchain_openai: type) -> None:
    """LangChain HumanMessage → 'user' role; SystemMessage → 'system'."""
    out = tmp_path / "trace.agentlog"
    chat = fake_langchain_openai()
    with Session(output_path=out) as s:
        chat._generate(
            messages=[
                SystemMessage("you are helpful"),
                HumanMessage("hi"),
            ]
        )
    requests = [r for r in s._records if r["kind"] == "chat_request"]
    assert len(requests) == 1
    msgs = requests[0]["payload"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


# ---- Async streaming proxy --------------------------------------------------
#
# Regression test for the customer-reported issue:
#   `TypeError: 'async_generator' object does not support the asynchronous
#   context manager protocol`
# under shadow record + LangChain ChatOpenAI(streaming=True).astream(...)
# ----------------------------------------------------------------------------


class _FakeAsyncStream:
    """Duck-type the OpenAI async Stream that's both iterable AND async ctxmgr."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._idx = 0
        self.entered = False
        self.exited = False
        self.aclosed = False

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    async def __aenter__(self) -> Any:
        self.entered = True
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        self.exited = True
        return False

    async def aclose(self) -> None:
        self.aclosed = True


def test_async_stream_proxy_supports_async_with(tmp_path: Path) -> None:
    """The wrapper preserves `async with`. THIS IS THE GPT-RESEARCHER REGRESSION FIX.

    Before the fix: the async wrapper returned a bare async generator,
    which has no ``__aenter__`` / ``__aexit__``. LangChain's
    ``ChatOpenAI(streaming=True).astream(...)`` does ``async with ...``
    on the response, which surfaced as ``TypeError: 'async_generator'
    object does not support the asynchronous context manager protocol``.

    After the fix, the wrapper is a proxy class with `__aenter__` /
    `__aexit__` that delegate to the underlying stream. This test
    asserts that contract directly without needing real LangChain.
    """
    from shadow.sdk.instrumentation import _AsyncStreamProxy

    out = tmp_path / "trace.agentlog"
    chunks = [
        types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="hel", tool_calls=None),
                    finish_reason=None,
                )
            ],
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            usage=None,
        ),
        types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="lo", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            usage=None,
        ),
    ]

    async def _go() -> dict[str, Any]:
        with Session(output_path=out) as session:
            from shadow.sdk.instrumentation import (
                _openai_req_from_kwargs,
                _openai_resp,
            )

            stream = _FakeAsyncStream(chunks)
            proxy = _AsyncStreamProxy(
                stream,
                session,
                _openai_req_from_kwargs,
                _openai_resp,
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                start=0.0,
            )

            collected: list[str] = []
            # The whole point: this exact pattern crashed before the fix.
            async with proxy as iter_proxy:
                async for ch in iter_proxy:
                    collected.append(ch.choices[0].delta.content or "")

            assert stream.entered, "underlying __aenter__ was not delegated"
            assert stream.exited, "underlying __aexit__ was not delegated"
            assert "".join(collected) == "hello"
            return {
                "records": list(session._records),
                "stream": stream,
            }

    result = asyncio.run(_go())
    chat_resps = [r for r in result["records"] if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1, "exactly one aggregated chat_response should be recorded"
    payload = chat_resps[0]["payload"]
    assert any(part.get("text") == "hello" for part in payload["content"])


def test_sync_stream_proxy_supports_with(tmp_path: Path) -> None:
    """Sync sibling: `with stream:` survives the wrapper."""
    from shadow.sdk.instrumentation import _SyncStreamProxy

    class _FakeSyncStream:
        def __init__(self, chunks: list[Any]) -> None:
            self._chunks = list(chunks)
            self._idx = 0
            self.entered = False
            self.exited = False

        def __iter__(self) -> Any:
            return self

        def __next__(self) -> Any:
            if self._idx >= len(self._chunks):
                raise StopIteration
            c = self._chunks[self._idx]
            self._idx += 1
            return c

        def __enter__(self) -> Any:
            self.entered = True
            return self

        def __exit__(self, *exc: Any) -> bool:
            self.exited = True
            return False

    out = tmp_path / "trace.agentlog"
    with Session(output_path=out) as session:
        from shadow.sdk.instrumentation import _openai_req_from_kwargs, _openai_resp

        stream = _FakeSyncStream(
            [
                types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="hi", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    usage=None,
                ),
            ]
        )
        proxy = _SyncStreamProxy(
            stream,
            session,
            _openai_req_from_kwargs,
            _openai_resp,
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            start=0.0,
        )
        with proxy as iter_proxy:
            collected = [ch.choices[0].delta.content for ch in iter_proxy]

    assert stream.entered
    assert stream.exited
    assert collected == ["hi"]
    chat_resps = [r for r in session._records if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1


# ---- Instrumentor.install + uninstall idempotence ---------------------------


def test_install_uninstall_does_not_break_when_no_sdks_installed(tmp_path: Path) -> None:
    """Smoke test: instrumentor cleanly handles missing SDKs."""
    out = tmp_path / "trace.agentlog"
    # No litellm / langchain in sys.modules — patcher should silently skip.
    with Session(output_path=out) as s:
        s.record_chat(
            request={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "params": {},
            },
            response={
                "model": "gpt-4o-mini",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    chat_resps = [r for r in s._records if r["kind"] == "chat_response"]
    assert len(chat_resps) == 1


def test_instrumentor_uninstall_is_idempotent(tmp_path: Path) -> None:
    """Calling uninstall twice is safe."""
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out) as s:
        instr = Instrumentor(s)
        instr.install()
        instr.uninstall()
        instr.uninstall()  # second call is a no-op
