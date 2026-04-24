"""Tests for the counterfactual replay module.

Counterfactual replay isolates one config delta at a time and
re-runs the trace through a backend. These tests lock down:

1. Each counterfactual kind produces the expected mutation on a
   `chat_request` payload.
2. Unknown or ill-formed counterfactuals raise `ShadowConfigError`
   at construction time, not deep inside a backend call.
3. `run_counterfactual` preserves the baseline metadata link and
   tags the new trace with a `counterfactual` marker.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from shadow.counterfactual import (
    Counterfactual,
    apply_counterfactual_to_request,
    run_counterfactual,
)
from shadow.errors import ShadowConfigError


def _req(
    *,
    model: str = "claude-opus-4-7",
    temperature: float = 0.2,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages
        or [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
        "params": {"temperature": temperature},
        "tools": tools or [],
    }


# ---- apply_counterfactual_to_request ------------------------------------


def test_model_counterfactual_swaps_model() -> None:
    out = apply_counterfactual_to_request(
        _req(model="claude-opus-4-7"),
        Counterfactual(kind="model", model="claude-haiku-4-5-20251001"),
    )
    assert out["model"] == "claude-haiku-4-5-20251001"
    # Everything else unchanged.
    assert out["params"]["temperature"] == 0.2


def test_temperature_counterfactual_overrides_only_temperature() -> None:
    out = apply_counterfactual_to_request(
        _req(temperature=0.0),
        Counterfactual(kind="temperature", temperature=0.7),
    )
    assert out["params"]["temperature"] == 0.7
    assert out["model"] == "claude-opus-4-7"


def test_top_p_counterfactual() -> None:
    out = apply_counterfactual_to_request(
        _req(),
        Counterfactual(kind="top_p", top_p=0.9),
    )
    assert out["params"]["top_p"] == 0.9


def test_max_tokens_counterfactual() -> None:
    out = apply_counterfactual_to_request(
        _req(),
        Counterfactual(kind="max_tokens", max_tokens=256),
    )
    assert out["params"]["max_tokens"] == 256


def test_system_prompt_counterfactual_replaces_existing_system() -> None:
    req = _req()
    out = apply_counterfactual_to_request(
        req, Counterfactual(kind="system_prompt", system_prompt="NEW INSTRUCTIONS")
    )
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][0]["content"] == "NEW INSTRUCTIONS"
    assert out["messages"][1] == {"role": "user", "content": "Hi"}


def test_system_prompt_counterfactual_prepends_when_no_system() -> None:
    req = _req(messages=[{"role": "user", "content": "Hi"}])
    out = apply_counterfactual_to_request(
        req, Counterfactual(kind="system_prompt", system_prompt="NEW")
    )
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][0]["content"] == "NEW"
    assert out["messages"][1]["role"] == "user"


def test_tools_counterfactual_replaces_tool_list() -> None:
    req = _req(tools=[{"name": "old_tool"}])
    new_tools = [{"name": "new_tool", "description": "fresh"}]
    out = apply_counterfactual_to_request(req, Counterfactual(kind="tools", tools=new_tools))
    assert out["tools"] == new_tools


def test_counterfactual_does_not_mutate_input() -> None:
    """Defensive: running the same counterfactual twice must not leave
    the second invocation with leaked state from the first."""
    req = _req()
    _ = apply_counterfactual_to_request(req, Counterfactual(kind="temperature", temperature=0.9))
    # Original is still at 0.2.
    assert req["params"]["temperature"] == 0.2


# ---- error paths --------------------------------------------------------


def test_missing_required_field_raises_config_error() -> None:
    """Missing the payload for the kind should raise at apply time
    with an actionable message."""
    with pytest.raises(ShadowConfigError, match="requires model"):
        apply_counterfactual_to_request(_req(), Counterfactual(kind="model", model=None))
    with pytest.raises(ShadowConfigError, match="requires temperature"):
        apply_counterfactual_to_request(
            _req(), Counterfactual(kind="temperature", temperature=None)
        )


def test_unknown_kind_raises() -> None:
    """An unknown `kind` — typed or runtime — should not slip through."""
    cf = Counterfactual(kind="temperature", temperature=0.5)  # type: ignore[arg-type]
    cf.kind = "not-a-real-kind"  # type: ignore[assignment]
    with pytest.raises(ShadowConfigError, match="unknown counterfactual kind"):
        apply_counterfactual_to_request(_req(), cf)


# ---- run_counterfactual (end-to-end) -----------------------------------


class _FakeBackend:
    """Minimal backend that returns canned responses per request."""

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    @property
    def id(self) -> str:
        return "fake"

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "model": request.get("model", ""),
            "content": [{"type": "text", "text": self._reply}],
            "stop_reason": "end_turn",
            "latency_ms": 1,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        }


def test_run_counterfactual_preserves_dag_and_tags_metadata() -> None:
    baseline = [
        {
            "version": "0.1",
            "id": "sha256:meta0",
            "kind": "metadata",
            "ts": "t",
            "parent": None,
            "payload": {"sdk": {"name": "shadow"}},
        },
        {
            "version": "0.1",
            "id": "sha256:req0",
            "kind": "chat_request",
            "ts": "t",
            "parent": "sha256:meta0",
            "payload": _req(model="claude-opus-4-7"),
        },
        {
            "version": "0.1",
            "id": "sha256:resp0",
            "kind": "chat_response",
            "ts": "t",
            "parent": "sha256:req0",
            "payload": {"model": "claude-opus-4-7", "content": [], "stop_reason": "end_turn"},
        },
    ]
    cf = Counterfactual(kind="model", model="claude-haiku-4-5-20251001")
    backend = _FakeBackend(reply="different")

    result = asyncio.run(run_counterfactual(baseline, cf, backend))

    # Metadata must carry the counterfactual marker.
    assert result[0]["kind"] == "metadata"
    meta_payload = result[0]["payload"]
    assert "counterfactual" in meta_payload
    assert meta_payload["counterfactual"]["kind"] == "model"
    assert meta_payload["counterfactual"]["baseline_of"] == "sha256:meta0"

    # Exactly one new chat_request + one chat_response.
    kinds = [r["kind"] for r in result]
    assert kinds == ["metadata", "chat_request", "chat_response"]

    # The counterfactual's model override made it to the backend.
    assert len(backend.calls) == 1
    assert backend.calls[0]["model"] == "claude-haiku-4-5-20251001"


def test_run_counterfactual_wraps_backend_errors_as_error_records() -> None:
    """If the backend fails, we produce an `error` record — not a
    panic — so the trace still represents the full run."""

    class _BrokenBackend:
        id = "broken"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("backend down")

    baseline = [
        {"version": "0.1", "id": "m", "kind": "metadata", "ts": "t", "parent": None, "payload": {}},
        {
            "version": "0.1",
            "id": "q",
            "kind": "chat_request",
            "ts": "t",
            "parent": "m",
            "payload": _req(),
        },
    ]
    result = asyncio.run(
        run_counterfactual(
            baseline,
            Counterfactual(kind="model", model="x"),
            _BrokenBackend(),  # type: ignore[arg-type]
        )
    )
    kinds = [r["kind"] for r in result]
    assert kinds == ["metadata", "chat_request", "error"]
    err_payload = result[-1]["payload"]
    assert "backend down" in err_payload["message"]


def test_run_counterfactual_rejects_empty_baseline() -> None:
    with pytest.raises(ShadowConfigError, match="baseline is empty"):
        asyncio.run(run_counterfactual([], Counterfactual(kind="model", model="x"), _FakeBackend()))


def test_run_counterfactual_rejects_non_metadata_root() -> None:
    bad = [
        {
            "version": "0.1",
            "id": "x",
            "kind": "chat_request",
            "ts": "t",
            "parent": None,
            "payload": {},
        }
    ]
    with pytest.raises(ShadowConfigError, match="baseline root"):
        asyncio.run(
            run_counterfactual(bad, Counterfactual(kind="model", model="x"), _FakeBackend())
        )
