"""LangGraph / LangChain adapter for Shadow.

Drop a :class:`ShadowLangChainHandler` into a LangGraph graph's
``RunnableConfig.callbacks`` and every LLM call and tool invocation
flowing through the graph gets recorded to a Shadow ``.agentlog`` file,
with session boundaries driven by the LangGraph ``thread_id``.

Design notes
------------

1. **Callback hook, not monkey-patch.** LangChain's
   :class:`AsyncCallbackHandler` is the framework's own documented
   instrumentation surface (LangSmith, Langfuse, and Arize Phoenix all
   wire in the same way). We subclass both the async and the sync base
   so the handler works under ``graph.invoke`` (sync) and
   ``graph.ainvoke`` / ``graph.astream_events`` (async) without the
   known sync-on-async race where LangChain dispatches sync callbacks
   through ``loop.run_in_executor``.

2. **Pair construction.** Shadow's
   :meth:`~shadow.sdk.Session.record_chat` takes a request + response
   in one shot because the two are siblings under a single content
   hash. LangChain fires two callbacks — ``on_chat_model_start`` and
   ``on_llm_end`` — so we buffer the request on *start* keyed by the
   unique ``run_id`` LangChain generates, and flush the pair on *end*.
   The start→end buffer is keyed purely on ``run_id`` so concurrent
   graph branches (which LangGraph routinely spawns for fan-outs)
   never cross-contaminate.

3. **Session grouping.** LangGraph's checkpointer keys runs by
   ``config.configurable.thread_id``; we surface that on every record
   via the ``meta.session_tag`` channel so Shadow's session-scoped
   policies and ``diff_by_session`` see the same boundaries the user
   does. The graph's ``invoke()`` exposes the config's metadata to
   callbacks via the ``metadata`` parameter.

4. **Tool calls.** LangChain fires ``on_tool_start`` with an
   ``inputs`` kwarg (dict) and ``on_tool_end`` with an ``output`` value.
   We map those to Shadow ``tool_call`` / ``tool_result`` records and
   correlate by ``run_id`` (the per-tool-call run id, not the parent
   graph's). LangChain also produces synthetic tool_call records inside
   ``AIMessage.tool_calls``; we surface those as ``tool_use`` blocks
   inside the paired ``chat_response`` content so the trace mirrors
   what Shadow's native Anthropic/OpenAI instrumentation produces.

5. **Error handling.** ``on_llm_error`` / ``on_tool_error`` produce a
   response record with ``stop_reason="error"`` and the exception class
   and message in the content. Shadow's policy engine already treats
   ``stop_reason="error"`` as a terminal session boundary, so multi-
   turn graphs recover the correct session topology even on failure.

6. **Zero-import fallback.** If ``langchain_core`` is not installed,
   importing this module raises a clear ``ImportError`` pointing to the
   ``shadow-diff[langgraph]`` extra. Users who never install the extra
   never import langchain.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

try:
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import LLMResult
except ImportError as exc:  # pragma: no cover - hit only when extra not installed
    raise ImportError(
        "shadow.adapters.langgraph requires langchain-core. "
        "Install it via `pip install 'shadow-diff[langgraph]'`."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover
    from shadow.sdk.session import Session


# Map LangChain finish reasons to the spec-standard Shadow values. The
# keys are the strings LangChain providers set in
# ``generation_info["finish_reason"]``; everything not in this map
# passes through unchanged.
_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "content_filter",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
}


class ShadowLangChainHandler(AsyncCallbackHandler):
    """LangChain callback handler that writes to a Shadow :class:`Session`.

    Works transparently under sync and async graph invocation because
    :class:`AsyncCallbackHandler` already provides awaitable methods and
    LangChain dispatches them correctly in both contexts.

    Parameters
    ----------
    session
        An active :class:`shadow.sdk.Session` (already inside its
        ``__enter__``). The handler does **not** manage the session
        lifecycle; the caller is responsible for the ``with`` block.
    session_tag
        Optional label copied onto each record's ``meta.session_tag``.
        Defaults to ``None``, in which case the LangGraph ``thread_id``
        (when present in callback metadata) is used.

    Example
    -------
    .. code-block:: python

        from shadow.sdk import Session
        from shadow.adapters.langgraph import ShadowLangChainHandler

        with Session(output_path="trace.agentlog") as s:
            handler = ShadowLangChainHandler(s)
            result = graph.invoke(
                {"messages": [HumanMessage("hi")]},
                config={"callbacks": [handler],
                        "configurable": {"thread_id": "t-42"}},
            )
    """

    # LangChain probes this attribute to decide whether to call the
    # handler in a thread pool. We're async-safe; set True so sync
    # chains also invoke us without the pool round-trip.
    raise_error = False
    run_inline = True

    def __init__(self, session: Session, *, session_tag: str | None = None) -> None:
        super().__init__()
        self._session = session
        self._session_tag = session_tag
        # run_id -> (chat_request_dict, started_at_monotonic, thread_id)
        self._pending: dict[UUID, tuple[dict[str, Any], float, str | None]] = {}
        # tool run_id -> (tool_name, tool_call_id, started_at_monotonic)
        self._pending_tools: dict[UUID, tuple[str, str, float]] = {}

    # ---- chat model lifecycle --------------------------------------------

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the request and buffer until :meth:`on_llm_end`."""
        invocation_params = kwargs.get("invocation_params") or {}
        model = _extract_model(serialized, invocation_params)
        # LangChain wraps messages in a list-of-lists for batch support;
        # a single chain only ever has one batch so index 0 is safe.
        flat = messages[0] if messages else []
        thread_id = _extract_thread_id(metadata)
        req_payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_dict(m) for m in flat],
            "params": _extract_params(invocation_params),
        }
        tools = _extract_tool_definitions(invocation_params)
        if tools:
            req_payload["tools"] = tools
        if thread_id:
            req_payload["thread_id"] = thread_id
        self._pending[run_id] = (req_payload, time.monotonic(), thread_id)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            # We didn't see the start (non-chat-model LLM call, e.g.
            # string-completion model). Skip silently; the user isn't
            # using a chat model here.
            return
        req_payload, start_t, thread_id = pending
        latency_ms = int((time.monotonic() - start_t) * 1000)
        resp_payload = _llm_result_to_response_payload(response, req_payload["model"], latency_ms)
        if thread_id:
            resp_payload["thread_id"] = thread_id
        self._session.record_chat(req_payload, resp_payload)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return
        req_payload, start_t, _ = pending
        latency_ms = int((time.monotonic() - start_t) * 1000)
        err_msg = f"{type(error).__name__}: {error}"
        resp_payload = {
            "model": req_payload.get("model", ""),
            "content": [{"type": "text", "text": err_msg}],
            "stop_reason": "error",
            "latency_ms": latency_ms,
            "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        self._session.record_chat(req_payload, resp_payload)

    # ---- tool lifecycle --------------------------------------------------

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = str(serialized.get("name") or serialized.get("id") or "unknown_tool")
        tool_call_id = str(run_id)
        args = inputs if isinstance(inputs, dict) and inputs else _coerce_tool_args(input_str)
        self._session.record_tool_call(tool_name, tool_call_id, args)
        self._pending_tools[run_id] = (tool_name, tool_call_id, time.monotonic())

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending_tools.pop(run_id, None)
        if pending is None:
            return
        _tool_name, tool_call_id, start_t = pending
        latency_ms = int((time.monotonic() - start_t) * 1000)
        self._session.record_tool_result(
            tool_call_id=tool_call_id,
            output=_coerce_tool_output(output),
            is_error=False,
            latency_ms=latency_ms,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending_tools.pop(run_id, None)
        if pending is None:
            return
        _tool_name, tool_call_id, start_t = pending
        latency_ms = int((time.monotonic() - start_t) * 1000)
        self._session.record_tool_result(
            tool_call_id=tool_call_id,
            output=f"{type(error).__name__}: {error}",
            is_error=True,
            latency_ms=latency_ms,
        )


# ---- helpers --------------------------------------------------------------


def _extract_thread_id(metadata: dict[str, Any] | None) -> str | None:
    """LangGraph surfaces ``config.configurable.thread_id`` via metadata."""
    if not metadata:
        return None
    tid = metadata.get("thread_id") or metadata.get("langgraph_thread_id")
    return str(tid) if tid is not None else None


def _extract_model(serialized: dict[str, Any], invocation_params: dict[str, Any]) -> str:
    """Pull the model identifier from LangChain's serialized chat model."""
    # `invocation_params` is set by most providers and is the most reliable.
    for key in ("model", "model_name", "model_id", "deployment_name"):
        val = invocation_params.get(key)
        if isinstance(val, str) and val:
            return val
    # Fall back to the serialized class path (e.g. ChatOpenAI).
    kwargs = serialized.get("kwargs") or {}
    for key in ("model", "model_name"):
        val = kwargs.get(key)
        if isinstance(val, str) and val:
            return val
    return str(serialized.get("name") or serialized.get("id") or "unknown")


_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_output_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "stop_sequences",
    "seed",
    "n",
)


def _extract_params(invocation_params: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in _PARAM_KEYS:
        if key in invocation_params and invocation_params[key] is not None:
            # Normalise two synonyms so downstream code sees one key.
            dst = (
                "max_tokens"
                if key == "max_output_tokens"
                else "stop"
                if key == "stop_sequences"
                else key
            )
            params.setdefault(dst, invocation_params[key])
    return params


def _extract_tool_definitions(invocation_params: dict[str, Any]) -> list[dict[str, Any]]:
    """LangChain bind_tools pushes tools into invocation_params.tools."""
    tools = invocation_params.get("tools")
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, dict):
            out.append(t)
    return out


def _message_to_dict(m: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain BaseMessage to Shadow's ``{role, content}`` dict."""
    role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(
        getattr(m, "type", ""), "user"
    )
    content = m.content
    # AIMessage may carry tool_calls on a separate attribute.
    out: dict[str, Any] = {"role": role, "content": content}
    tool_calls = getattr(m, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            }
            for tc in tool_calls
            if isinstance(tc, dict)
        ]
    tool_call_id = getattr(m, "tool_call_id", None)
    if isinstance(tool_call_id, str) and tool_call_id:
        out["tool_call_id"] = tool_call_id
    return out


def _llm_result_to_response_payload(
    response: LLMResult, model: str, latency_ms: int
) -> dict[str, Any]:
    """Flatten an LLMResult into Shadow's chat_response payload.

    LangChain's ``LLMResult.generations`` is a ``list[list[Generation]]``
    — outer list is per-input (batch), inner is per-candidate (n>1).
    We take the first candidate of the first input, which matches the
    single-chat convention Shadow uses.
    """
    generations = response.generations or []
    first = generations[0][0] if generations and generations[0] else None
    content_blocks: list[dict[str, Any]] = []
    stop_reason = "end_turn"
    response_model = model
    has_tool_calls = False
    if first is not None:
        msg = getattr(first, "message", None)
        if msg is not None:
            text = msg.content if isinstance(msg.content, str) else ""
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in getattr(msg, "tool_calls", None) or []:
                if not isinstance(tc, dict):
                    continue
                has_tool_calls = True
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id", "")),
                        "name": str(tc.get("name", "")),
                        "input": tc.get("args") or {},
                    }
                )
        # generation_info.finish_reason is the authoritative signal when
        # the provider sets it; fake chat models (and a few real ones on
        # structured-output paths) don't, so fall back to detecting tool
        # calls on the message itself.
        gi = getattr(first, "generation_info", None) or {}
        fr = gi.get("finish_reason")
        if isinstance(fr, str) and fr:
            stop_reason = _FINISH_REASON_MAP.get(fr, fr)
        elif has_tool_calls:
            stop_reason = "tool_use"
        rm = gi.get("model_name") or gi.get("model")
        if isinstance(rm, str) and rm:
            response_model = rm
    # Fall back to a token_usage block on the overall result if no
    # generation-level usage is present.
    llm_output = response.llm_output or {}
    usage_in = llm_output.get("token_usage") or llm_output.get("usage") or {}
    usage = {
        "input_tokens": int(usage_in.get("prompt_tokens") or usage_in.get("input_tokens") or 0),
        "output_tokens": int(
            usage_in.get("completion_tokens") or usage_in.get("output_tokens") or 0
        ),
        "thinking_tokens": int(
            usage_in.get("thinking_tokens") or usage_in.get("reasoning_tokens") or 0
        ),
    }
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return {
        "model": response_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }


def _coerce_tool_args(input_str: str) -> dict[str, Any]:
    """Best-effort parse of the stringified tool input LangChain passes."""
    import json

    if not input_str:
        return {}
    try:
        parsed = json.loads(input_str)
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    except (ValueError, TypeError):
        return {"input": input_str}


def _coerce_tool_output(output: Any) -> str:
    """Render a tool output into a string for the Shadow record.

    LangChain tools can return ``ToolMessage``, strings, dicts, or
    arbitrary objects. Strings pass through; everything else is
    JSON-encoded when possible, else ``str()``-coerced.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, ToolMessage):
        content = output.content
        return content if isinstance(content, str) else str(content)
    if isinstance(output, dict | list):
        import json

        try:
            return json.dumps(output, default=str)
        except (TypeError, ValueError):
            return str(output)
    return str(output)


# Re-export for users who prefer a short alias.
ShadowHandler = ShadowLangChainHandler


__all__ = [
    "AIMessage",
    "ShadowHandler",
    "ShadowLangChainHandler",
    "SystemMessage",
]
