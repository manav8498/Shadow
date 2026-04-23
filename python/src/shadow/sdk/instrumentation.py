"""Auto-instrumentation for anthropic + openai Python clients.

When a `Session` enters, we monkey-patch the `.create` method on the
message/completion resource classes of whichever SDKs are installed.
The wrappers:

1. Time the call (wall-clock latency_ms).
2. Call the original method, passing through *args/**kwargs unchanged.
3. Convert kwargs → Shadow `chat_request` payload and response →
   Shadow `chat_response` payload, and push both into the active
   `Session` via `session.record_chat`.
4. Streaming calls (`stream=True` on OpenAI, or `.stream()` on
   Anthropic) are passed through un-recorded — proper streaming
   capture is scoped to a later release.

On `Session.__exit__` we restore the originals. The instrumentor is
best-effort: if an SDK isn't importable, or its internal resource path
has shifted, we silently skip that SDK rather than breaking user code.
Exceptions raised *inside* the recording path are swallowed for the
same reason (user calls must keep working).
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadow.sdk.session import Session


class Instrumentor:
    """Install/restore monkey-patches around anthropic + openai `.create`."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._patches: list[tuple[Any, str, Any]] = []

    def install(self) -> None:
        self._patch_anthropic()
        self._patch_openai()

    def uninstall(self) -> None:
        for cls, attr, original in self._patches:
            with contextlib.suppress(Exception):
                setattr(cls, attr, original)
        self._patches.clear()

    # ---- anthropic -------------------------------------------------------

    def _patch_anthropic(self) -> None:
        try:
            from anthropic.resources.messages import (  # type: ignore[import-not-found, unused-ignore]
                AsyncMessages,
                Messages,
            )
        except Exception:
            return
        self._install_sync(Messages, "create", _anthropic_req_from_kwargs, _anthropic_resp)
        self._install_async(AsyncMessages, "create", _anthropic_req_from_kwargs, _anthropic_resp)

    # ---- openai ----------------------------------------------------------

    def _patch_openai(self) -> None:
        try:
            from openai.resources.chat.completions import (  # type: ignore[import-not-found, unused-ignore]
                AsyncCompletions,
                Completions,
            )
        except Exception:
            pass
        else:
            self._install_sync(Completions, "create", _openai_req_from_kwargs, _openai_resp)
            self._install_async(AsyncCompletions, "create", _openai_req_from_kwargs, _openai_resp)
        # OpenAI Responses API (SDK v1.40+, the recommended path for new
        # code). Separate request/response shapes — translators below.
        try:
            from openai.resources.responses import (  # type: ignore[import-not-found, unused-ignore]
                AsyncResponses,
                Responses,
            )
        except Exception:
            return
        self._install_sync(
            Responses, "create", _openai_responses_req_from_kwargs, _openai_responses_resp
        )
        self._install_async(
            AsyncResponses, "create", _openai_responses_req_from_kwargs, _openai_responses_resp
        )

    # ---- core patch machinery -------------------------------------------

    def _install_sync(
        self,
        cls: Any,
        attr: str,
        req_translator: Any,
        resp_translator: Any,
    ) -> None:
        original = getattr(cls, attr, None)
        if original is None:
            return
        session = self._session

        def wrapper(client_self: Any, *args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream") is True
            start = time.perf_counter()
            result = original(client_self, *args, **kwargs)
            if is_stream:
                return _wrap_sync_stream(
                    result, session, req_translator, resp_translator, kwargs, start
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            _record_safely(session, req_translator, resp_translator, kwargs, result, latency_ms)
            return result

        self._patches.append((cls, attr, original))
        setattr(cls, attr, wrapper)

    def _install_async(
        self,
        cls: Any,
        attr: str,
        req_translator: Any,
        resp_translator: Any,
    ) -> None:
        original = getattr(cls, attr, None)
        if original is None:
            return
        session = self._session

        async def wrapper(client_self: Any, *args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream") is True
            start = time.perf_counter()
            result = await original(client_self, *args, **kwargs)
            if is_stream:
                return _wrap_async_stream(
                    result, session, req_translator, resp_translator, kwargs, start
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            _record_safely(session, req_translator, resp_translator, kwargs, result, latency_ms)
            return result

        self._patches.append((cls, attr, original))
        setattr(cls, attr, wrapper)


def _record_safely(
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    kwargs: dict[str, Any],
    result: Any,
    latency_ms: int,
) -> None:
    """Never let the recording layer break the caller's LLM call."""
    try:
        shadow_req = req_translator(kwargs)
        shadow_resp = resp_translator(result, latency_ms)
        session.record_chat(shadow_req, shadow_resp)
    except Exception:
        # Best-effort: recording failures must not propagate.
        return


# ---------------------------------------------------------------------------
# Translators: SDK kwargs / responses → Shadow chat_request / chat_response.
# ---------------------------------------------------------------------------


def _anthropic_req_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """anthropic.messages.create(**kwargs) → Shadow chat_request payload."""
    messages = [dict(m) for m in kwargs.get("messages", [])]
    system = kwargs.get("system")
    if system is not None:
        messages = [{"role": "system", "content": system}, *messages]
    params: dict[str, Any] = {}
    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ):
        if src in kwargs:
            params[dst] = kwargs[src]
    out: dict[str, Any] = {
        "model": kwargs.get("model", ""),
        "messages": messages,
        "params": params,
    }
    tools = kwargs.get("tools")
    if tools:
        out["tools"] = [dict(t) for t in tools]
    return out


def _anthropic_resp(response: Any, latency_ms: int) -> dict[str, Any]:
    """anthropic Message object → Shadow chat_response payload."""
    from shadow.llm.anthropic_backend import AnthropicLLM

    return AnthropicLLM._from_provider(response, latency_ms)


def _openai_req_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """openai.chat.completions.create(**kwargs) → Shadow chat_request payload."""
    messages = [dict(m) for m in kwargs.get("messages", [])]
    params: dict[str, Any] = {}
    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("max_completion_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop"),
    ):
        if src in kwargs:
            params[dst] = kwargs[src]
    out: dict[str, Any] = {
        "model": kwargs.get("model", ""),
        "messages": messages,
        "params": params,
    }
    tools = kwargs.get("tools")
    if tools:
        shadow_tools: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function":
                fn = t.get("function", {})
                shadow_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    }
                )
            else:
                shadow_tools.append(dict(t))
        out["tools"] = shadow_tools
    return out


def _openai_resp(response: Any, latency_ms: int) -> dict[str, Any]:
    """openai ChatCompletion → Shadow chat_response payload."""
    from shadow.llm.openai_backend import OpenAILLM

    return OpenAILLM._from_provider(response, latency_ms)


# ---------------------------------------------------------------------------
# OpenAI Responses API translators (SDK v1.40+).
#
# The Responses API has a flatter shape than Chat Completions:
#   request:  {model, input: str|[items...], instructions?, tools?, ...}
#   response: {id, model, output: [items...], usage, status, ...}
# where each `item` is {type: "message"|"function_call"|...}. We translate
# to the same Shadow chat_request/chat_response envelope the Chat API uses,
# so downstream axes don't need to care which API produced the record.
# ---------------------------------------------------------------------------


def _openai_responses_req_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """openai.responses.create(**kwargs) → Shadow chat_request payload."""
    messages: list[dict[str, Any]] = []
    instructions = kwargs.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    input_val = kwargs.get("input")
    if isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})
    elif isinstance(input_val, list):
        for item in input_val:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            content = item.get("content")
            messages.append({"role": role, "content": content})
    params: dict[str, Any] = {}
    for src, dst in (
        ("max_output_tokens", "max_tokens"),
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        if src in kwargs and kwargs[src] is not None:
            params[dst] = kwargs[src]
    out: dict[str, Any] = {
        "model": kwargs.get("model", ""),
        "messages": messages,
        "params": params,
    }
    tools = kwargs.get("tools")
    if tools:
        shadow_tools: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                # Responses API tool shape: {type: "function", name, parameters, ...}
                # (flat, unlike Chat's nested {type:"function", function:{...}})
                if t.get("type") == "function" and "function" not in t:
                    shadow_tools.append(
                        {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "input_schema": t.get("parameters", {}),
                        }
                    )
                else:
                    shadow_tools.append(dict(t))
        out["tools"] = shadow_tools
    return out


def _openai_responses_resp(response: Any, latency_ms: int) -> dict[str, Any]:
    """openai Response → Shadow chat_response payload."""
    content: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "message":
            for part in getattr(item, "content", None) or []:
                ptype = getattr(part, "type", None)
                if ptype in ("output_text", "text"):
                    text = getattr(part, "text", "") or ""
                    if text:
                        content.append({"type": "text", "text": text})
                elif ptype == "refusal":
                    content.append({"type": "refusal", "text": getattr(part, "refusal", "")})
        elif itype in ("function_call", "tool_use"):
            # Responses API function calls have {name, arguments (json str), call_id}.
            import json as _json

            args_raw = getattr(item, "arguments", "") or ""
            try:
                parsed = _json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (ValueError, TypeError):
                parsed = {"_raw": args_raw}
            content.append(
                {
                    "type": "tool_use",
                    "id": getattr(item, "call_id", "") or getattr(item, "id", ""),
                    "name": getattr(item, "name", ""),
                    "input": parsed,
                }
            )

    usage = getattr(response, "usage", None)
    thinking_tokens = 0
    cached_input_tokens = 0
    if usage is not None:
        # Responses API usage shape: input_tokens, output_tokens,
        # output_tokens_details.reasoning_tokens, input_tokens_details.cached_tokens
        out_details = getattr(usage, "output_tokens_details", None)
        if out_details is not None:
            thinking_tokens = getattr(out_details, "reasoning_tokens", 0) or 0
        in_details = getattr(usage, "input_tokens_details", None)
        if in_details is not None:
            cached_input_tokens = getattr(in_details, "cached_tokens", 0) or 0

    status = getattr(response, "status", None)
    stop_reason = (
        {
            "completed": "end_turn",
            "incomplete": "max_tokens",
            "failed": "error",
        }.get(str(status), "end_turn")
        if status
        else "end_turn"
    )

    usage_out: dict[str, Any] = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "thinking_tokens": thinking_tokens,
    }
    if cached_input_tokens:
        usage_out["cached_input_tokens"] = cached_input_tokens
    return {
        "model": getattr(response, "model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage_out,
    }


# ---------------------------------------------------------------------------
# Streaming support: intercept the iterator, accumulate chunks, and after the
# consumer finishes iterating emit a single chat_response record with the
# aggregated content. We never buffer the full stream in memory beyond the
# accumulator; each chunk passes through to the consumer immediately.
# ---------------------------------------------------------------------------


def _wrap_sync_stream(
    stream: Any,
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    kwargs: dict[str, Any],
    start: float,
) -> Any:
    """Wrap a sync stream iterator; emit a chat_response once exhausted."""
    chunks: list[Any] = []

    def gen() -> Any:
        try:
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            _record_stream_safely(
                session, req_translator, resp_translator, kwargs, chunks, latency_ms
            )

    return gen()


def _wrap_async_stream(
    stream: Any,
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    kwargs: dict[str, Any],
    start: float,
) -> Any:
    """Wrap an async stream; emit a chat_response once exhausted."""
    chunks: list[Any] = []

    async def gen() -> Any:
        try:
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            _record_stream_safely(
                session, req_translator, resp_translator, kwargs, chunks, latency_ms
            )

    return gen()


def _record_stream_safely(
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    kwargs: dict[str, Any],
    chunks: list[Any],
    latency_ms: int,
) -> None:
    """Best-effort: aggregate chunks into a synthetic response and record."""
    try:
        # The translators are chat-shaped so we need to build a ChatCompletion/
        # Message-shaped object from the chunks. We detect shape from the chunk
        # attributes; fall back to stringifying if we can't.
        aggregated = _aggregate_chunks(chunks)
        shadow_req = req_translator(kwargs)
        if aggregated is None:
            # Couldn't shape it — record as a raw text response.
            shadow_resp = _raw_text_response(chunks, latency_ms)
        else:
            shadow_resp = resp_translator(aggregated, latency_ms)
        session.record_chat(shadow_req, shadow_resp)
    except Exception:
        return


def _aggregate_chunks(chunks: list[Any]) -> Any | None:
    """Rebuild a provider-shaped response from streamed deltas.

    Handles:
      - OpenAI ChatCompletionChunk: chunks have `.choices[0].delta.{content,tool_calls}`
      - Anthropic: chunks have `.type == "content_block_delta"` etc.
    Returns an object with the same attribute shape the non-streaming response
    has, so the existing `_from_provider` paths work unchanged.
    """
    if not chunks:
        return None
    first = chunks[0]
    if _looks_like_openai_chunk(first):
        return _aggregate_openai(chunks)
    if _looks_like_anthropic_event(first):
        return _aggregate_anthropic(chunks)
    return None


def _looks_like_openai_chunk(chunk: Any) -> bool:
    return (
        hasattr(chunk, "choices")
        and getattr(chunk, "object", None)
        in (
            "chat.completion.chunk",
            None,
        )
        and hasattr(chunk, "choices")
    )


def _looks_like_anthropic_event(chunk: Any) -> bool:
    return hasattr(chunk, "type") and isinstance(getattr(chunk, "type", None), str)


class _Aggregated:
    """Duck-typed object that mimics the provider response shape."""

    def __init__(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)


def _aggregate_openai(chunks: list[Any]) -> Any:
    text_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    model = ""
    usage = None
    for chunk in chunks:
        model = getattr(chunk, "model", model) or model
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            text_parts.append(content)
        tc_deltas = getattr(delta, "tool_calls", None) or []
        for td in tc_deltas:
            idx = getattr(td, "index", 0) or 0
            existing = tool_calls_by_index.setdefault(
                idx,
                {"id": "", "function_name": "", "function_arguments": ""},
            )
            if getattr(td, "id", None):
                existing["id"] = td.id
            fn = getattr(td, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    existing["function_name"] = fn.name
                if getattr(fn, "arguments", None):
                    existing["function_arguments"] += fn.arguments
        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish_reason = fr

    tool_calls = [
        _Aggregated(
            id=tc["id"],
            type="function",
            function=_Aggregated(name=tc["function_name"], arguments=tc["function_arguments"]),
        )
        for tc in tool_calls_by_index.values()
    ]
    message = _Aggregated(content="".join(text_parts) or None, tool_calls=tool_calls or None)
    choice_obj = _Aggregated(message=message, finish_reason=finish_reason)
    return _Aggregated(model=model, choices=[choice_obj], usage=usage)


def _aggregate_anthropic(events: list[Any]) -> Any:
    # Anthropic event types: message_start, content_block_start, content_block_delta,
    # content_block_stop, message_delta, message_stop. Deltas are in
    # event.delta.{text,partial_json}.
    model = ""
    stop_reason = "end_turn"
    parts: dict[int, dict[str, Any]] = {}  # index → {"type", "text"/"input"}
    usage_in = 0
    usage_out = 0
    # Anthropic streaming emits `cache_read_input_tokens` on `message_start.usage`
    # (input counts) and `message_delta.usage` (final tallies). Route to
    # Shadow's `cached_input_tokens` field so the cost axis prices cache
    # reads at the cheap cache-read rate, not the reasoning/output rate.
    cached_input_tokens = 0
    for ev in events:
        etype = getattr(ev, "type", "")
        if etype == "message_start":
            msg = getattr(ev, "message", None)
            if msg is not None:
                model = getattr(msg, "model", model) or model
                u = getattr(msg, "usage", None)
                if u is not None:
                    usage_in = getattr(u, "input_tokens", 0) or usage_in
                    cached_input_tokens = (
                        getattr(u, "cache_read_input_tokens", 0) or cached_input_tokens
                    )
        elif etype == "content_block_start":
            idx = getattr(ev, "index", 0)
            block = getattr(ev, "content_block", None)
            btype = getattr(block, "type", "text") if block else "text"
            parts[idx] = {"type": btype, "text": "", "name": "", "input": ""}
            if btype == "tool_use" and block is not None:
                parts[idx]["name"] = getattr(block, "name", "")
                parts[idx]["id"] = getattr(block, "id", "")
        elif etype == "content_block_delta":
            idx = getattr(ev, "index", 0)
            delta = getattr(ev, "delta", None)
            if delta is None:
                continue
            if hasattr(delta, "text") and getattr(delta, "text", None):
                parts.setdefault(idx, {"type": "text", "text": ""})["text"] += delta.text
            elif hasattr(delta, "partial_json"):
                parts.setdefault(idx, {"type": "tool_use", "input": ""})["input"] += (
                    delta.partial_json or ""
                )
            elif hasattr(delta, "thinking"):
                parts.setdefault(idx, {"type": "thinking", "text": ""})["text"] += (
                    delta.thinking or ""
                )
        elif etype == "message_delta":
            delta = getattr(ev, "delta", None)
            if delta is not None and getattr(delta, "stop_reason", None):
                stop_reason = delta.stop_reason
            u = getattr(ev, "usage", None)
            if u is not None:
                usage_out = getattr(u, "output_tokens", 0) or usage_out
                cached_input_tokens = (
                    getattr(u, "cache_read_input_tokens", 0) or cached_input_tokens
                )

    content: list[Any] = []
    for idx in sorted(parts.keys()):
        p = parts[idx]
        ptype = p.get("type", "text")
        if ptype == "text":
            content.append(_Aggregated(type="text", text=p.get("text", "")))
        elif ptype == "thinking":
            content.append(_Aggregated(type="thinking", text=p.get("text", "")))
        elif ptype == "tool_use":
            import json as _json

            try:
                parsed = _json.loads(p.get("input", "") or "{}")
            except (ValueError, TypeError):
                parsed = {"_raw": p.get("input", "")}
            content.append(
                _Aggregated(
                    type="tool_use",
                    id=p.get("id", ""),
                    name=p.get("name", ""),
                    input=parsed,
                )
            )
    usage = _Aggregated(
        input_tokens=usage_in,
        output_tokens=usage_out,
        cache_read_input_tokens=cached_input_tokens,
    )
    return _Aggregated(model=model, content=content, stop_reason=stop_reason, usage=usage)


def _raw_text_response(chunks: list[Any], latency_ms: int) -> dict[str, Any]:
    """Fallback when we can't identify the chunk shape."""
    texts: list[str] = []
    for ch in chunks:
        if isinstance(ch, str):
            texts.append(ch)
        elif isinstance(ch, bytes):
            texts.append(ch.decode("utf-8", errors="replace"))
    return {
        "model": "",
        "content": [{"type": "text", "text": "".join(texts)}],
        "stop_reason": "end_turn",
        "latency_ms": latency_ms,
        "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
    }


__all__ = ["Instrumentor"]
