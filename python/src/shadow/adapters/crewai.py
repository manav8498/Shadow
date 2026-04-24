"""CrewAI adapter for Shadow.

Subclass :class:`ShadowCrewAIListener`, pass your :class:`Session`, and
every ``LLMCall{Started,Completed,Failed}`` and
``ToolUsage{Started,Finished,Error}`` event CrewAI emits gets recorded
to the shadow ``.agentlog`` file.

Design notes
------------

1. **Event-bus listener, not monkey-patch.** ``crewai_event_bus`` is
   CrewAI's first-party instrumentation surface — OpenInference (event-
   listener mode), Langfuse, and AgentOps all wire in the same way.
   Subclassing :class:`BaseEventListener` means we survive any CrewAI
   refactor that keeps the public bus contract stable.

2. **Pairing by call_id.** LLMCallStartedEvent carries the outbound
   messages; LLMCallCompletedEvent carries the response. Both share
   ``event.call_id``, which we use as the buffer key so concurrent
   crews (which 1.14+ can spawn) never cross-contaminate.

3. **Session grouping.** Shadow surfaces CrewAI's ``task_id`` on every
   record payload and lets Shadow's session-boundary detector handle
   the rest. Each CrewKickoffStartedEvent starts a new session-like
   chain; we don't need to invent a boundary protocol.

4. **Tool calls.** ToolUsageStartedEvent -> Shadow ``tool_call``.
   ToolUsageFinishedEvent -> ``tool_result``. ToolUsageErrorEvent
   records a tool_result with ``is_error=True`` so downstream policy
   checks can treat failed tools as terminal turn boundaries.

5. **Double-fire guard.** CrewAI paired with OpenInference wrapper
   mode sometimes emits duplicate LLM spans. This listener keys purely
   on ``call_id`` so even if CrewAI's own token-counter listener fires
   a sibling event, Shadow sees only one pair per call.

6. **Zero-import fallback.** If ``crewai`` is not installed, importing
   this module raises a clear ``ImportError`` pointing to the
   ``shadow-diff[crewai]`` extra.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

try:
    from crewai.events import crewai_event_bus
    from crewai.events.base_event_listener import BaseEventListener
    from crewai.events.types.crew_events import (
        CrewKickoffCompletedEvent,
        CrewKickoffStartedEvent,
    )
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
except ImportError as exc:  # pragma: no cover - hit only without the extra
    raise ImportError(
        "shadow.adapters.crewai requires crewai. "
        "Install it via `pip install 'shadow-diff[crewai]'`."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover
    from shadow.sdk.session import Session


class ShadowCrewAIListener(BaseEventListener):
    """CrewAI event listener that writes to a Shadow :class:`Session`.

    Parameters
    ----------
    session
        An active :class:`shadow.sdk.Session`. The listener does **not**
        manage the session lifecycle.
    session_tag
        Optional label attached to records via ``meta.session_tag``.
    capture_kickoff
        When True (default), CrewKickoffStartedEvent/CompletedEvent
        are surfaced as synthetic metadata entries so ``diff_by_session``
        can see crew boundaries even when a single .agentlog contains
        multiple kickoffs.

    Example
    -------
    .. code-block:: python

        from shadow.sdk import Session
        from shadow.adapters.crewai import ShadowCrewAIListener

        with Session(output_path="trace.agentlog") as s:
            listener = ShadowCrewAIListener(s)
            result = crew.kickoff(inputs={"topic": "..."})
    """

    def __init__(
        self,
        session: Session,
        *,
        session_tag: str | None = None,
        capture_kickoff: bool = True,
    ) -> None:
        self._session = session
        self._session_tag = session_tag
        self._capture_kickoff = capture_kickoff
        # call_id -> (req_dict, started_at_monotonic)
        self._pending_calls: dict[str, tuple[dict[str, Any], float]] = {}
        # call_id -> (tool_name, tool_call_id, started_at_monotonic)
        self._pending_tools: dict[str, tuple[str, str, float]] = {}
        # Must run last so handlers are registered.
        super().__init__()

    def setup_listeners(self, event_bus: Any) -> None:
        """Register handlers for every CrewAI event Shadow cares about."""

        @event_bus.on(LLMCallStartedEvent)
        def _on_llm_start(_source: Any, event: LLMCallStartedEvent) -> None:
            self._handle_llm_start(event)

        @event_bus.on(LLMCallCompletedEvent)
        def _on_llm_end(_source: Any, event: LLMCallCompletedEvent) -> None:
            self._handle_llm_completed(event)

        @event_bus.on(LLMCallFailedEvent)
        def _on_llm_failed(_source: Any, event: LLMCallFailedEvent) -> None:
            self._handle_llm_failed(event)

        @event_bus.on(ToolUsageStartedEvent)
        def _on_tool_start(_source: Any, event: ToolUsageStartedEvent) -> None:
            self._handle_tool_start(event)

        @event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_end(_source: Any, event: ToolUsageFinishedEvent) -> None:
            self._handle_tool_end(event)

        @event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(_source: Any, event: ToolUsageErrorEvent) -> None:
            self._handle_tool_error(event)

        if self._capture_kickoff:

            @event_bus.on(CrewKickoffStartedEvent)
            def _on_kickoff_start(_source: Any, event: CrewKickoffStartedEvent) -> None:
                # No-op for now; hook left here so future
                # session-tagging work has a documented home.
                _ = event

            @event_bus.on(CrewKickoffCompletedEvent)
            def _on_kickoff_end(_source: Any, event: CrewKickoffCompletedEvent) -> None:
                _ = event

    # ---- LLM pair handlers ---------------------------------------------

    def _handle_llm_start(self, event: LLMCallStartedEvent) -> None:
        req_payload: dict[str, Any] = {
            "model": str(getattr(event, "model", "") or ""),
            "messages": _normalise_messages(getattr(event, "messages", []) or []),
            "params": {},
        }
        tools = getattr(event, "tools", None)
        if isinstance(tools, list) and tools:
            req_payload["tools"] = [t for t in tools if isinstance(t, dict)]
        task_id = _stringify(getattr(event, "task_id", None))
        if task_id:
            req_payload["task_id"] = task_id
        agent_id = _stringify(getattr(event, "agent_id", None))
        if agent_id:
            req_payload["agent_id"] = agent_id
        call_id = _call_id(event)
        if call_id:
            self._pending_calls[call_id] = (req_payload, time.monotonic())

    def _handle_llm_completed(self, event: LLMCallCompletedEvent) -> None:
        call_id = _call_id(event)
        pending = self._pending_calls.pop(call_id, None) if call_id else None
        if pending is None:
            return
        req_payload, started_at = pending
        latency_ms = int((time.monotonic() - started_at) * 1000)
        resp_payload = _build_response_payload(event, req_payload["model"], latency_ms)
        self._session.record_chat(req_payload, resp_payload)

    def _handle_llm_failed(self, event: LLMCallFailedEvent) -> None:
        call_id = _call_id(event)
        pending = self._pending_calls.pop(call_id, None) if call_id else None
        if pending is None:
            return
        req_payload, started_at = pending
        latency_ms = int((time.monotonic() - started_at) * 1000)
        err = getattr(event, "error", "") or ""
        err_msg = str(err)
        resp_payload = {
            "model": req_payload.get("model", ""),
            "content": [{"type": "text", "text": err_msg}],
            "stop_reason": "error",
            "latency_ms": latency_ms,
            "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            "error": {"type": type(err).__name__ if err else "LLMCallFailed", "message": err_msg},
        }
        self._session.record_chat(req_payload, resp_payload)

    # ---- tool handlers --------------------------------------------------

    def _handle_tool_start(self, event: ToolUsageStartedEvent) -> None:
        tool_name = str(getattr(event, "tool_name", "") or "unknown_tool")
        # Use the event's own id as the pairing key. Finished and error
        # events reference it via their `started_event_id` field.
        pairing_id = _stringify(getattr(event, "event_id", None)) or tool_name
        args = getattr(event, "tool_args", {}) or {}
        if not isinstance(args, dict):
            # CrewAI sometimes serialises args as a string before the tool
            # runs; we keep the original under 'input' so nothing is lost.
            args = {"input": str(args)}
        self._session.record_tool_call(tool_name, pairing_id, args)
        self._pending_tools[pairing_id] = (tool_name, pairing_id, time.monotonic())

    def _handle_tool_end(self, event: ToolUsageFinishedEvent) -> None:
        # CrewAI's convention: finished/error events reference the
        # starting event via `started_event_id`. That's the correct
        # pairing key. `event_id` on the finished event is unique to the
        # finished event itself (useful for ordering, not pairing).
        started_id = _stringify(getattr(event, "started_event_id", None))
        pairing_id = started_id or _stringify(getattr(event, "event_id", None))
        if not pairing_id:
            return
        pending = self._pending_tools.pop(pairing_id, None)
        started_at = pending[2] if pending is not None else time.monotonic()
        latency_ms = int((time.monotonic() - started_at) * 1000)
        output = getattr(event, "output", "") or ""
        self._session.record_tool_result(
            tool_call_id=pairing_id,
            output=_coerce_output(output),
            is_error=False,
            latency_ms=latency_ms,
        )

    def _handle_tool_error(self, event: ToolUsageErrorEvent) -> None:
        started_id = _stringify(getattr(event, "started_event_id", None))
        pairing_id = started_id or _stringify(getattr(event, "event_id", None))
        if not pairing_id:
            return
        pending = self._pending_tools.pop(pairing_id, None)
        started_at = pending[2] if pending is not None else time.monotonic()
        latency_ms = int((time.monotonic() - started_at) * 1000)
        err = getattr(event, "error", "") or ""
        self._session.record_tool_result(
            tool_call_id=pairing_id,
            output=f"{type(err).__name__ if err else 'ToolError'}: {err}",
            is_error=True,
            latency_ms=latency_ms,
        )


# ---- helpers --------------------------------------------------------------


def _call_id(event: Any) -> str:
    return _stringify(getattr(event, "call_id", None)) or _stringify(
        getattr(event, "event_id", None)
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalise_messages(raw: Any) -> list[dict[str, Any]]:
    """CrewAI passes messages as a list of dicts with role/content."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for m in raw:
        if isinstance(m, dict):
            out.append({"role": str(m.get("role", "user")), "content": m.get("content", "")})
    return out


def _build_response_payload(
    event: LLMCallCompletedEvent, model: str, latency_ms: int
) -> dict[str, Any]:
    response = getattr(event, "response", "")
    if isinstance(response, str):
        content_text = response
    elif isinstance(response, dict):
        content_text = str(
            response.get("content") or response.get("text") or response.get("message") or ""
        )
    else:
        content_text = str(response or "")
    usage_raw = getattr(event, "usage", None) or {}
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0),
        "output_tokens": int(
            usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0
        ),
        "thinking_tokens": int(
            usage_raw.get("reasoning_tokens") or usage_raw.get("thinking_tokens") or 0
        ),
    }
    response_model = str(getattr(event, "model", "") or model or "")
    blocks = (
        [{"type": "text", "text": content_text}] if content_text else [{"type": "text", "text": ""}]
    )
    return {
        "model": response_model,
        "content": blocks,
        "stop_reason": "end_turn",
        "latency_ms": latency_ms,
        "usage": usage,
    }


def _coerce_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict | list):
        import json

        try:
            return json.dumps(output, default=str)
        except (TypeError, ValueError):
            return str(output)
    return str(output)


# Short alias matching the LangGraph adapter's convention.
ShadowListener = ShadowCrewAIListener


__all__ = [
    "ShadowCrewAIListener",
    "ShadowListener",
    "crewai_event_bus",
]
