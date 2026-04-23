"""Resilience tests: 429, 500, network errors, timeouts don't break Shadow.

Every test asserts two invariants:
  1. Shadow never raises an exception that the caller didn't see already.
  2. Whatever successfully completed BEFORE the failure is preserved.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from shadow import _core
from shadow.bisect.corner_scorer import replay_with_config
from shadow.errors import ShadowBackendError
from shadow.sdk import Session

# ---------------------------------------------------------------------------
# Fake backends that simulate various failure modes.
# ---------------------------------------------------------------------------


class _RateLimitBackend:
    """Emits a 429 Too Many Requests on every call."""

    @property
    def id(self) -> str:
        return "rate-limit"

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        raise ShadowBackendError("429 Too Many Requests")


class _FlakyBackend:
    """Fails every other call with a 500; succeeds otherwise."""

    def __init__(self) -> None:
        self.n = 0

    @property
    def id(self) -> str:
        return "flaky"

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.n += 1
        if self.n % 2 == 0:
            raise ShadowBackendError("500 Internal Server Error")
        return {
            "model": "x",
            "content": [{"type": "text", "text": f"call {self.n}"}],
            "stop_reason": "end_turn",
            "latency_ms": 10,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        }


class _TimeoutBackend:
    """Raises asyncio.TimeoutError — a real client-side timeout."""

    @property
    def id(self) -> str:
        return "timeout"

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("request timed out")


# ---------------------------------------------------------------------------
# Replay-with-config under failure modes.
# ---------------------------------------------------------------------------


def _baseline_with_pairs(tmp_path: Path, n: int) -> list[dict[str, Any]]:
    path = tmp_path / "b.agentlog"
    with Session(output_path=path, auto_instrument=False) as s:
        for i in range(n):
            s.record_chat(
                request={
                    "model": "x",
                    "messages": [{"role": "user", "content": f"t{i}"}],
                    "params": {},
                },
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 10,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )
    return _core.parse_agentlog(path.read_bytes())


def test_replay_with_rate_limited_backend_emits_error_records(tmp_path: Path) -> None:
    baseline = _baseline_with_pairs(tmp_path, 3)
    config = {"model": "x", "prompt": {"system": "hi"}, "params": {}, "tools": []}
    result = asyncio.run(replay_with_config(baseline, config, _RateLimitBackend()))
    # Each failed call becomes an `error` record; the replay doesn't crash.
    errors = [r for r in result if r["kind"] == "error"]
    assert len(errors) == 3
    for e in errors:
        assert "429" in e["payload"]["message"]


def test_replay_with_flaky_backend_records_partial_progress(tmp_path: Path) -> None:
    baseline = _baseline_with_pairs(tmp_path, 4)
    config = {"model": "x", "prompt": {"system": "hi"}, "params": {}, "tools": []}
    result = asyncio.run(replay_with_config(baseline, config, _FlakyBackend()))
    # 4 requests, 4 successes/errors — every call recorded one way or the other.
    req_count = sum(1 for r in result if r["kind"] == "chat_request")
    resp_count = sum(1 for r in result if r["kind"] == "chat_response")
    err_count = sum(1 for r in result if r["kind"] == "error")
    assert req_count == 4
    assert resp_count + err_count == 4
    assert resp_count >= 1 and err_count >= 1  # mix of both


def test_replay_with_timeout_backend_raises_unknown_error_as_general_error(
    tmp_path: Path,
) -> None:
    """Non-ShadowBackendError exceptions should propagate (they're real bugs,
    not just transient failures)."""
    baseline = _baseline_with_pairs(tmp_path, 2)
    config = {"model": "x", "prompt": {"system": "hi"}, "params": {}, "tools": []}
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(replay_with_config(baseline, config, _TimeoutBackend()))


# ---------------------------------------------------------------------------
# Auto-instrumentation under provider exceptions.
# ---------------------------------------------------------------------------


def _install_fake_openai_raising(
    exc: Exception,
) -> tuple[type, type]:
    class Completions:
        def create(self, **kwargs: Any) -> Any:
            raise exc

    class AsyncCompletions:
        async def create(self, **kwargs: Any) -> Any:
            raise exc

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


def _cleanup_openai() -> None:
    for n in [
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
    ]:
        sys.modules.pop(n, None)


def test_instrumentor_does_not_swallow_provider_exception(tmp_path: Path) -> None:
    """If the provider raises, the user sees it. Shadow doesn't hide errors."""

    class FakeRateLimitError(Exception):
        pass

    completions_cls, _ = _install_fake_openai_raising(FakeRateLimitError("429"))
    try:
        with Session(output_path=tmp_path / "t.agentlog"):
            client = completions_cls()
            with pytest.raises(FakeRateLimitError):
                client.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                )
    finally:
        _cleanup_openai()

    # The session should still write a valid .agentlog (metadata only — no
    # successful chat to record).
    records = _core.parse_agentlog((tmp_path / "t.agentlog").read_bytes())
    assert records[0]["kind"] == "metadata"


def test_instrumentor_recording_never_leaks_its_own_exceptions(tmp_path: Path) -> None:
    """If our translator blows up (e.g. malformed response), the user's
    call result is still returned intact."""

    # Response shape that intentionally breaks our translator.
    class _BrokenResponse:
        pass

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            return _BrokenResponse()

    class AsyncCompletions:
        async def create(self, **kwargs: Any) -> Any:
            return _BrokenResponse()

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

    try:
        with Session(output_path=tmp_path / "t.agentlog"):
            client = Completions()
            # The malformed response shape will make our translator throw;
            # the user's call should still succeed and return the object.
            result = client.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
            )
            assert isinstance(result, _BrokenResponse)
    finally:
        _cleanup_openai()


# ---------------------------------------------------------------------------
# Redactor under pathological inputs (fuzz-ish).
# ---------------------------------------------------------------------------


def test_redactor_handles_pathological_inputs() -> None:
    """Redactor should never crash on weird inputs."""
    from shadow.redact import Redactor

    r = Redactor()
    # Tricky strings.
    for s in [
        "",
        "a" * 100_000,
        "email@@@@@x@y.com",
        "\x00\x01\x02 mixed control chars",
        "🔥🔥🔥 emoji stress 🔥🔥🔥",
        "credit 4111-1111-1111-1111 in middle",
        {"nested": {"email": "a@b.com", "phone": "+14155551234"}},
        [[[{"sk-ant": "sk-ant-api03-deadbeef"}]]],
    ]:
        r.redact_value(s)  # just asserts no exception
