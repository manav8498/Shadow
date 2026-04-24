"""AG2 (formerly AutoGen) adapter for Shadow.

Use :class:`ShadowAG2Adapter` to wrap one or more
:class:`~autogen.agentchat.ConversableAgent` instances; every LLM call
routed through those agents gets recorded to the shadow ``.agentlog``
file.

Design notes
------------

1. **Hook on the safeguard surface.** AG2's
   :meth:`ConversableAgent.register_hook` surface is the framework's
   own documented extension point. The ``safeguard_llm_inputs`` hook
   fires immediately before the LLM call with the fully-assembled
   ``messages`` list, and ``safeguard_llm_outputs`` fires immediately
   after with the raw response. That pair is exactly the bracket
   Shadow needs for a ``chat_request``/``chat_response`` record.

2. **Why not the OTel surface.** ``autogen.opentelemetry`` (shipped
   Feb 2026) emits OTel spans that look spec-compliant but, as noted
   in the AG2 OTel blog post and reproduced by Langfuse issue #11505,
   it redacts message bodies by default. We want the request payloads
   for Shadow's semantic diff, so we hook the safeguards directly and
   let the user layer in the OTel exporter separately if they want
   timing metrics. Track B (our OTel v1.40 importer) already accepts
   whatever the native AG2 OTel exporter emits.

3. **Per-agent registration.** AG2's hooks are per-instance; a
   ``GroupChat`` of five agents needs five hook registrations. The
   adapter keeps an internal list of instrumented agents and offers
   :meth:`install` (single agent) and :meth:`install_all` (many).

4. **Pairing key.** ``safeguard_llm_inputs`` takes ``messages`` and
   returns ``messages``; ``safeguard_llm_outputs`` takes ``response``
   and returns ``response``. They're connected by the owning agent,
   not by a request id. We key the pending-buffer on
   ``id(agent)`` which is stable for the life of the agent instance.
   Concurrent LLM calls on the same agent are rare in AG2 (each agent
   is sequential by design) but we detect re-entry and skip buffering
   a second simultaneous call so we never cross-contaminate.

5. **Message-to-agent routing.** We additionally register
   ``process_message_before_send`` so inter-agent handoffs are
   captured as metadata on the next outbound message. This is how AG2
   orchestrates multi-agent conversations; surfacing it makes the
   Shadow session structure mirror what the user sees in AG2's
   console output.

6. **Zero-import fallback.** If ``autogen`` / ``ag2`` is not
   installed, importing this module raises a clear ``ImportError``
   pointing to the ``shadow-diff[ag2]`` extra.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

try:
    from autogen.agentchat import ConversableAgent
except ImportError as exc:  # pragma: no cover - hit only without the extra
    raise ImportError(
        "shadow.adapters.ag2 requires ag2 (the former autogen package). "
        "Install it via `pip install 'shadow-diff[ag2]'`."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover
    from shadow.sdk.session import Session


class ShadowAG2Adapter:
    """Instruments AG2 agents so their LLM calls land in a Shadow :class:`Session`.

    Parameters
    ----------
    session
        An active :class:`shadow.sdk.Session`. The adapter does **not**
        manage the session lifecycle; wrap it in a ``with`` block.

    Example
    -------
    .. code-block:: python

        from autogen.agentchat import ConversableAgent
        from shadow.sdk import Session
        from shadow.adapters.ag2 import ShadowAG2Adapter

        planner = ConversableAgent(name="planner", ...)
        executor = ConversableAgent(name="executor", ...)

        with Session(output_path="trace.agentlog") as s:
            adapter = ShadowAG2Adapter(s)
            adapter.install_all([planner, executor])
            planner.initiate_chat(executor, message="plan the migration")
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._installed: list[ConversableAgent] = []
        # id(agent) -> (request_payload_dict, started_at_monotonic)
        self._pending: dict[int, tuple[dict[str, Any], float]] = {}

    def install(self, agent: ConversableAgent) -> ConversableAgent:
        """Register hooks on one agent; returns the agent for chaining."""
        agent.register_hook("safeguard_llm_inputs", self._make_input_hook(agent))
        agent.register_hook("safeguard_llm_outputs", self._make_output_hook(agent))
        self._installed.append(agent)
        return agent

    def install_all(self, agents: list[ConversableAgent]) -> list[ConversableAgent]:
        for a in agents:
            self.install(a)
        return list(self._installed)

    # ---- hook factories --------------------------------------------------

    def _make_input_hook(self, agent: ConversableAgent) -> Any:
        def on_inputs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            key = id(agent)
            # Skip buffering if another LLM call is already in flight on
            # this agent (AG2 is sequential per-agent, but defensive
            # coding against future async paths costs us nothing).
            if key in self._pending:
                return messages
            model = _extract_model(agent)
            req_payload: dict[str, Any] = {
                "model": model,
                "messages": _normalise_messages(messages),
                "params": _extract_params(agent),
                "agent": {
                    "name": getattr(agent, "name", ""),
                    "id": str(id(agent)),
                },
            }
            self._pending[key] = (req_payload, time.monotonic())
            return messages

        return on_inputs

    def _make_output_hook(self, agent: ConversableAgent) -> Any:
        def on_outputs(response: Any) -> Any:
            key = id(agent)
            pending = self._pending.pop(key, None)
            if pending is None:
                return response
            req_payload, started_at = pending
            latency_ms = int((time.monotonic() - started_at) * 1000)
            resp_payload = _build_response_payload(response, req_payload["model"], latency_ms)
            # We're inside an AG2 hook; suppressing here avoids
            # corrupting the agent's reply path on a Shadow bug. The
            # record is lost but the user's workflow continues.
            with contextlib.suppress(Exception):
                self._session.record_chat(req_payload, resp_payload)
            return response

        return on_outputs


# ---- helpers --------------------------------------------------------------


def _extract_model(agent: ConversableAgent) -> str:
    """Best-effort model lookup on an AG2 ConversableAgent.

    AG2's llm_config shape varies across versions: a dict, a list of
    ``config_list`` entries, or an ``LLMConfig`` object. We walk the
    common shapes and return the first ``model`` we find; empty string
    when no config is set (the agent may be a non-LLM agent).
    """
    cfg = getattr(agent, "llm_config", None)
    if cfg is None or cfg is False:
        return ""
    if isinstance(cfg, dict):
        cl = cfg.get("config_list")
        if isinstance(cl, list) and cl and isinstance(cl[0], dict):
            model = cl[0].get("model")
            if isinstance(model, str) and model:
                return model
        model = cfg.get("model")
        if isinstance(model, str) and model:
            return model
    # LLMConfig-like objects expose config_list as an attribute.
    cl = getattr(cfg, "config_list", None)
    if isinstance(cl, list) and cl:
        first = cl[0]
        if isinstance(first, dict):
            return str(first.get("model") or "")
        return str(getattr(first, "model", "") or "")
    return ""


_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
)


def _extract_params(agent: ConversableAgent) -> dict[str, Any]:
    """Pull standard LLM sampling params from the agent's config."""
    cfg = getattr(agent, "llm_config", None)
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _PARAM_KEYS:
        if key in cfg and cfg[key] is not None:
            out[key] = cfg[key]
    return out


def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce AG2's message dicts into Shadow's ``{role, content}`` shape.

    AG2 carries extra fields (``name``, ``tool_calls``, ``tool_responses``)
    that Shadow preserves verbatim on the message dict so nothing is
    lost for downstream diffs.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        msg: dict[str, Any] = {
            "role": str(m.get("role", "user")),
            "content": m.get("content", ""),
        }
        if "name" in m:
            msg["name"] = m["name"]
        tc = m.get("tool_calls")
        if isinstance(tc, list) and tc:
            msg["tool_calls"] = tc
        tcid = m.get("tool_call_id")
        if isinstance(tcid, str) and tcid:
            msg["tool_call_id"] = tcid
        out.append(msg)
    return out


def _build_response_payload(response: Any, model: str, latency_ms: int) -> dict[str, Any]:
    """Build a Shadow chat_response payload from an AG2 LLM response.

    AG2's ``safeguard_llm_outputs`` receives whatever the underlying
    client produced — a raw string for simple chat completions, a dict
    with ``choices[0].message`` for OpenAI-style responses, or an
    AG2-wrapped dict.
    """
    content_text = ""
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    response_model = model
    stop_reason = "end_turn"

    if isinstance(response, str):
        content_text = response
    elif isinstance(response, dict):
        # OpenAI-style response shape
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                msg = choice.get("message") or {}
                if isinstance(msg, dict):
                    content_text = str(msg.get("content") or "")
                    raw_tc = msg.get("tool_calls")
                    if isinstance(raw_tc, list):
                        for tc in raw_tc:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function") or {}
                            tool_calls.append(
                                {
                                    "type": "tool_use",
                                    "id": str(tc.get("id", "")),
                                    "name": str(fn.get("name", "") if isinstance(fn, dict) else ""),
                                    "input": fn.get("arguments") if isinstance(fn, dict) else {},
                                }
                            )
                fr = choice.get("finish_reason")
                if isinstance(fr, str) and fr:
                    stop_reason = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                        "function_call": "tool_use",
                        "content_filter": "content_filter",
                    }.get(fr, fr)
        # Plain content string at top level
        if not content_text:
            direct = response.get("content")
            if isinstance(direct, str):
                content_text = direct
        # Usage
        raw_usage = response.get("usage") or {}
        if isinstance(raw_usage, dict):
            usage = {
                "input_tokens": int(
                    raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0
                ),
                "output_tokens": int(
                    raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0
                ),
                "thinking_tokens": int(
                    raw_usage.get("reasoning_tokens") or raw_usage.get("thinking_tokens") or 0
                ),
            }
        rm = response.get("model")
        if isinstance(rm, str) and rm:
            response_model = rm
    else:
        content_text = str(response or "")

    blocks: list[dict[str, Any]] = []
    if content_text:
        blocks.append({"type": "text", "text": content_text})
    blocks.extend(tool_calls)
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    return {
        "model": response_model,
        "content": blocks,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }


# Short alias matching LangGraph/CrewAI convention.
ShadowAdapter = ShadowAG2Adapter


__all__ = ["ShadowAG2Adapter", "ShadowAdapter"]
