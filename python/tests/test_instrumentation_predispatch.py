"""Tests for v2.2 auto-instrument-layer pre-dispatch enforcement.

The Instrumentor wraps `.create` on the OpenAI / Anthropic SDK
classes. When the active session is an EnforcedSession (has
`_enforcer`), each non-streaming response is probed for tool_use
blocks BEFORE the response is returned to user code. Violating
calls raise PolicyViolationError at the auto-instrument layer.

We mock the SDK classes — no real HTTP, no provider keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shadow.hierarchical import load_policy
from shadow.policy_runtime import EnforcedSession, PolicyEnforcer, PolicyViolationError
from shadow.sdk.instrumentation import _enforce_pre_dispatch
from shadow.sdk.session import Session

# ---- helpers -----------------------------------------------------------


def _no_call_rules(tool: str) -> list[Any]:
    return load_policy(
        [{"id": f"no-{tool}", "kind": "no_call", "params": {"tool": tool}, "severity": "error"}]
    )


def _resp_with_tool_use(tool_name: str = "delete_user") -> dict[str, Any]:
    """Shadow chat_response payload with a single tool_use block."""
    return {
        "model": "gpt-4o-mini",
        "content": [
            {
                "type": "tool_use",
                "id": "call_abc",
                "name": tool_name,
                "input": {"user_id": "u-1"},
            }
        ],
        "stop_reason": "tool_use",
        "latency_ms": 100,
        "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
    }


def _resp_no_tool() -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "latency_ms": 100,
        "usage": {"input_tokens": 5, "output_tokens": 1, "thinking_tokens": 0},
    }


def _identity_resp_translator(result: Any, _latency: int) -> dict[str, Any]:
    """Translator that just returns the dict it was handed — keeps the
    test free of SDK-specific response object construction."""
    return result if isinstance(result, dict) else {}


def _identity_req_translator(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": kwargs.get("model", "x"),
        "messages": kwargs.get("messages", []),
        "params": {},
    }


# ---- _enforce_pre_dispatch direct unit tests --------------------------


def test_enforce_predispatch_no_op_on_plain_session(tmp_path: Path) -> None:
    """A plain Session (no _enforcer) is a complete no-op — the
    auto-instrument layer must not regress for users who never opted
    into runtime enforcement."""
    with Session(output_path=tmp_path / "plain.agentlog", auto_instrument=False) as s:
        # Should not raise even on a tool_use response.
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            _identity_resp_translator,
            {"model": "x"},
            _resp_with_tool_use("delete_user"),
            100,
        )


def test_enforce_predispatch_no_op_on_response_without_tool_use(tmp_path: Path) -> None:
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="raise")
    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "no-tool.agentlog",
        auto_instrument=False,
    ) as s:
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            _identity_resp_translator,
            {"model": "x"},
            _resp_no_tool(),
            100,
        )  # no exception expected


def test_enforce_predispatch_raises_on_violating_tool_use(tmp_path: Path) -> None:
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="raise")
    with (
        EnforcedSession(
            enforcer=enforcer,
            output_path=tmp_path / "raise.agentlog",
            auto_instrument=False,
        ) as s,
        pytest.raises(PolicyViolationError, match="delete_user"),
    ):
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            _identity_resp_translator,
            {"model": "x"},
            _resp_with_tool_use("delete_user"),
            100,
        )


def test_enforce_predispatch_replace_mode_also_raises(tmp_path: Path) -> None:
    """Replace mode at the auto-instrument layer is approximated by raise
    (modifying the SDK's response object across versions is fragile).
    Document and verify."""
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="replace")
    with (
        EnforcedSession(
            enforcer=enforcer,
            output_path=tmp_path / "replace.agentlog",
            auto_instrument=False,
        ) as s,
        pytest.raises(PolicyViolationError),
    ):
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            _identity_resp_translator,
            {"model": "x"},
            _resp_with_tool_use("delete_user"),
            100,
        )


def test_enforce_predispatch_warn_mode_passes_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="warn")
    with (
        EnforcedSession(
            enforcer=enforcer,
            output_path=tmp_path / "warn.agentlog",
            auto_instrument=False,
        ) as s,
        caplog.at_level("WARNING", logger="shadow.policy_runtime"),
    ):
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            _identity_resp_translator,
            {"model": "x"},
            _resp_with_tool_use("delete_user"),
            100,
        )
    assert any("would be blocked but warn" in r.message for r in caplog.records)


def test_enforce_predispatch_allows_non_violating_tool_use(tmp_path: Path) -> None:
    """An allowed tool (not in the no_call list) must pass through
    silently — the wrapper must not block legitimate tool calls."""
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="raise")
    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "allow.agentlog",
        auto_instrument=False,
    ) as s:
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            _identity_resp_translator,
            {"model": "x"},
            _resp_with_tool_use("lookup_order"),  # NOT in no_call
            100,
        )


def test_enforce_predispatch_does_not_pollute_enforcer_state(tmp_path: Path) -> None:
    """Probe is non-mutating — repeated blocked calls leave the enforcer
    state clean. Mirrors the wrap_tools regression test."""
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="raise")
    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "clean.agentlog",
        auto_instrument=False,
    ) as s:
        for _ in range(5):
            with pytest.raises(PolicyViolationError):
                _enforce_pre_dispatch(
                    s,
                    _identity_req_translator,
                    _identity_resp_translator,
                    {"model": "x"},
                    _resp_with_tool_use("delete_user"),
                    100,
                )
        # _known should still be empty — probe didn't mutate.
        assert len(enforcer._known) == 0


def test_enforce_predispatch_handles_translator_error_gracefully(tmp_path: Path) -> None:
    """If the response translator raises (SDK-version drift, malformed
    response), pre-dispatch must NOT raise — fall through to normal
    recording, which is best-effort."""
    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="raise")

    def broken_translator(_result: Any, _latency: int) -> dict[str, Any]:
        raise RuntimeError("translator broke")

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "broken-trans.agentlog",
        auto_instrument=False,
    ) as s:
        # No exception — translator failure is silently absorbed at this
        # layer (the recording layer logs/skips on its own path).
        _enforce_pre_dispatch(
            s,
            _identity_req_translator,
            broken_translator,
            {"model": "x"},
            {"anything": "here"},
            100,
        )


# ---- end-to-end via Instrumentor + a fake patched class ---------------


class _FakeOpenAIChoice:
    def __init__(self) -> None:
        self.message = type("M", (), {})()
        self.message.content = ""
        self.message.refusal = None
        self.message.tool_calls = []
        self.finish_reason = "tool_calls"


class _FakeOpenAIResponse:
    def __init__(self) -> None:
        self.choices = [_FakeOpenAIChoice()]
        self.usage = type(
            "U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        )()
        self.model = "gpt-4o-mini"


class _FakeCompletions:
    def create(self, **kwargs: Any) -> _FakeOpenAIResponse:
        # Simulate the OpenAI SDK returning a response with a tool_calls
        # entry. We construct just enough of the shape that the
        # auto-instrument's response translator can consume.
        resp = _FakeOpenAIResponse()
        choice = resp.choices[0]
        # Build a synthetic tool_call entry on the message.
        tc = type("TC", (), {})()
        tc.id = "call_xyz"
        tc.type = "function"
        tc.function = type("F", (), {"name": "delete_user", "arguments": '{"user_id": "u-1"}'})()
        choice.message.tool_calls = [tc]
        return resp


def test_end_to_end_auto_instrument_pre_dispatch_with_fake_sdk(tmp_path: Path) -> None:
    """End-to-end: install the Instrumentor on a fake OpenAI Completions
    class, fire a `.create` call, and verify the enforcer raises
    PolicyViolationError before the response is handed to the caller.
    Uses the real translator so the integration is realistic."""
    from shadow.sdk.instrumentation import Instrumentor

    enforcer = PolicyEnforcer(_no_call_rules("delete_user"), on_violation="raise")
    sess = EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "e2e.agentlog",
        auto_instrument=False,
    )
    sess.__enter__()
    try:
        # Manually install on the fake class.
        instr = Instrumentor(sess)
        instr._install_sync(
            _FakeCompletions,
            "create",
            _import_openai_req_translator(),
            _import_openai_resp_translator(),
        )
        try:
            client = _FakeCompletions()
            with pytest.raises(PolicyViolationError, match="delete_user"):
                client.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "delete user u-1"}],
                )
        finally:
            instr.uninstall()
    finally:
        sess.__exit__(None, None, None)


def _import_openai_req_translator() -> Any:
    from shadow.sdk.instrumentation import _openai_req_from_kwargs

    return _openai_req_from_kwargs


def _import_openai_resp_translator() -> Any:
    from shadow.sdk.instrumentation import _openai_resp

    return _openai_resp
