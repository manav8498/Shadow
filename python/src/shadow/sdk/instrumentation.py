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
        self._patch_litellm()
        self._patch_langchain_openai()

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

    # ---- litellm ---------------------------------------------------------

    def _patch_litellm(self) -> None:
        """Patch litellm.completion / acompletion / text_completion / atext_completion.

        LiteLLM is a module-level routing layer used by mini-SWE-agent,
        OpenHands, Skyvern, and many other agent frameworks. It accepts
        OpenAI-shape kwargs (`model`, `messages`, `tools`, `stream`, etc.)
        and returns an OpenAI-shape `ModelResponse`, so the existing
        OpenAI request/response translators apply directly.

        We patch as MODULE attributes (not class methods) because that's
        how the public surface is exposed. `litellm.completion(...)` is
        called with the function looked up off the module each time, so
        re-binding the module attribute catches every call site.
        """
        try:
            import litellm  # type: ignore[import-not-found, unused-ignore]
        except Exception:
            return

        for attr_name in ("completion", "text_completion"):
            self._install_module_sync(
                litellm,
                attr_name,
                _litellm_req_from_kwargs,
                _openai_resp,
            )
        for attr_name in ("acompletion", "atext_completion"):
            self._install_module_async(
                litellm,
                attr_name,
                _litellm_req_from_kwargs,
                _openai_resp,
            )

    # ---- langchain_openai ChatOpenAI -------------------------------------

    def _patch_langchain_openai(self) -> None:
        """Patch langchain_openai.ChatOpenAI._generate / _agenerate.

        LangChain's ChatOpenAI internally constructs an `openai.OpenAI`
        client at import time and stores a BOUND METHOD reference to
        `client.chat.completions.create`. That bound method captures
        the unpatched function before our `Session.__enter__` patches
        the class — so monkey-patching `Completions.create` doesn't
        catch LangChain calls that go through the captured reference.

        Patching at the LangChain layer (the `_generate` / `_agenerate`
        methods on `BaseChatOpenAI`) sidesteps that capture. The methods
        accept `(self, messages, stop=None, run_manager=None, **kwargs)`
        and return a `ChatResult`. We translate at this layer using a
        LangChain-shaped translator pair.

        Open SWE works because it uses the LangGraph adapter (via
        callback handlers), which is a separate path. This patcher
        catches the *direct* `ChatOpenAI(...).invoke / .ainvoke` path.
        """
        try:
            from langchain_openai.chat_models.base import (  # type: ignore[import-not-found, unused-ignore]
                BaseChatOpenAI,
            )
        except Exception:
            return

        # _generate (sync) and _agenerate (async) are the two convergence
        # points every ChatOpenAI call funnels through (whether invoked
        # via .invoke, .ainvoke, .batch, .abatch, .stream, .astream).
        # Streaming paths re-enter via _generate too; they buffer chunks
        # internally and produce a final ChatResult.
        self._install_sync(
            BaseChatOpenAI,
            "_generate",
            _langchain_chat_req_from_args,
            _langchain_chat_resp,
            arg_passthrough=True,
        )
        self._install_async(
            BaseChatOpenAI,
            "_agenerate",
            _langchain_chat_req_from_args,
            _langchain_chat_resp,
            arg_passthrough=True,
        )

    # ---- core patch machinery -------------------------------------------

    def _install_sync(
        self,
        cls: Any,
        attr: str,
        req_translator: Any,
        resp_translator: Any,
        *,
        arg_passthrough: bool = False,
    ) -> None:
        """Wrap a class-level sync method.

        ``arg_passthrough`` controls how the request translator is
        called: by default it expects only ``kwargs``; when True the
        wrapper passes ``(args, kwargs)`` so translators can read
        positional arguments (e.g. LangChain's ``_generate(messages,
        stop, run_manager, **kwargs)`` signature).
        """
        original = getattr(cls, attr, None)
        if original is None:
            return
        session = self._session

        def wrapper(client_self: Any, *args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream") is True
            start = time.perf_counter()
            result = original(client_self, *args, **kwargs)
            translator_input = (args, kwargs) if arg_passthrough else kwargs
            if is_stream:
                return _wrap_sync_stream(
                    result, session, req_translator, resp_translator, translator_input, start
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            # Pre-dispatch enforcement: probe the enforcer with the
            # response's tool_use blocks BEFORE recording. May raise
            # PolicyViolationError; the caller's handler sees it.
            _enforce_pre_dispatch(
                session, req_translator, resp_translator, translator_input, result, latency_ms
            )
            _record_safely(
                session, req_translator, resp_translator, translator_input, result, latency_ms
            )
            return result

        self._patches.append((cls, attr, original))
        setattr(cls, attr, wrapper)

    def _install_async(
        self,
        cls: Any,
        attr: str,
        req_translator: Any,
        resp_translator: Any,
        *,
        arg_passthrough: bool = False,
    ) -> None:
        original = getattr(cls, attr, None)
        if original is None:
            return
        session = self._session

        async def wrapper(client_self: Any, *args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream") is True
            start = time.perf_counter()
            result = await original(client_self, *args, **kwargs)
            translator_input = (args, kwargs) if arg_passthrough else kwargs
            if is_stream:
                return _wrap_async_stream(
                    result, session, req_translator, resp_translator, translator_input, start
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            _enforce_pre_dispatch(
                session, req_translator, resp_translator, translator_input, result, latency_ms
            )
            _record_safely(
                session, req_translator, resp_translator, translator_input, result, latency_ms
            )
            return result

        self._patches.append((cls, attr, original))
        setattr(cls, attr, wrapper)

    def _install_module_sync(
        self,
        module: Any,
        attr: str,
        req_translator: Any,
        resp_translator: Any,
    ) -> None:
        """Wrap a module-level sync function (e.g. ``litellm.completion``).

        Module-level functions don't have an implicit ``self`` argument
        — the wrapper signature is ``(*args, **kwargs)`` directly. The
        patch is installed on the module attribute and removed by
        restoring it on uninstall, same as class-level patches.
        """
        original = getattr(module, attr, None)
        if original is None or not callable(original):
            return
        session = self._session

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream") is True
            start = time.perf_counter()
            result = original(*args, **kwargs)
            if is_stream:
                return _wrap_sync_stream(
                    result, session, req_translator, resp_translator, kwargs, start
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            _enforce_pre_dispatch(
                session, req_translator, resp_translator, kwargs, result, latency_ms
            )
            _record_safely(session, req_translator, resp_translator, kwargs, result, latency_ms)
            return result

        self._patches.append((module, attr, original))
        setattr(module, attr, wrapper)

    def _install_module_async(
        self,
        module: Any,
        attr: str,
        req_translator: Any,
        resp_translator: Any,
    ) -> None:
        """Wrap a module-level async function (e.g. ``litellm.acompletion``)."""
        original = getattr(module, attr, None)
        if original is None or not callable(original):
            return
        session = self._session

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            is_stream = kwargs.get("stream") is True
            start = time.perf_counter()
            result = await original(*args, **kwargs)
            if is_stream:
                return _wrap_async_stream(
                    result, session, req_translator, resp_translator, kwargs, start
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            _enforce_pre_dispatch(
                session, req_translator, resp_translator, kwargs, result, latency_ms
            )
            _record_safely(session, req_translator, resp_translator, kwargs, result, latency_ms)
            return result

        self._patches.append((module, attr, original))
        setattr(module, attr, wrapper)


def _record_safely(
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    translator_input: Any,
    result: Any,
    latency_ms: int,
) -> None:
    """Never let the recording layer break the caller's LLM call.

    ``translator_input`` is either a kwargs dict (default for
    OpenAI / Anthropic / LiteLLM) or an ``(args, kwargs)`` tuple
    (for LangChain whose translator needs positional ``messages``).
    """
    try:
        shadow_req = req_translator(translator_input)
        shadow_resp = resp_translator(result, latency_ms)
        session.record_chat(shadow_req, shadow_resp)
    except Exception:
        # Best-effort: recording failures must not propagate.
        return


def _enforce_pre_dispatch(
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    translator_input: Any,
    result: Any,
    latency_ms: int,
) -> None:
    """Auto-instrument-layer pre-dispatch enforcement.

    Called AFTER the original ``.create`` returned but BEFORE the
    response is handed back to user code. Translates the response
    into Shadow's ``chat_response`` payload, extracts any ``tool_use``
    blocks, synthesises candidate ``tool_call`` records, and probes
    the session's enforcer.

    On any violation, behaves per the enforcer's ``on_violation``:
      - ``raise``: raises ``PolicyViolationError`` immediately. The
        recording layer never sees the response; the caller's
        exception handler observes the block.
      - ``replace``: same as ``raise`` for v2.2 — auto-instrument
        replace-mode is approximated by raise. Modifying the SDK's
        own response object across SDK versions is fragile, so
        :func:`shadow.policy_runtime.wrap_tools` remains the
        recommended path when callers want a synthetic-refusal
        replacement of individual tool calls. The error message
        mentions wrap_tools for callers who want finer control.
      - ``warn``: logs the violation and lets the response through
        (the recording layer still records the underlying response).

    No-op when the session isn't an :class:`EnforcedSession` (no
    ``_enforcer`` attribute) or when the response carries no
    ``tool_use`` blocks.
    """
    enforcer = getattr(session, "_enforcer", None)
    if enforcer is None:
        return
    try:
        from shadow.policy_runtime import PolicyViolationError
    except Exception:  # pragma: no cover — policy_runtime ships in-tree
        return
    try:
        shadow_resp = resp_translator(result, latency_ms)
    except Exception:
        # If we can't translate the response, we can't probe. Fall
        # through to normal recording — at worst the user pays
        # post-response enforcement instead of pre-dispatch.
        return

    tool_uses = [
        b
        for b in shadow_resp.get("content") or []
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    if not tool_uses:
        return  # no tool calls; nothing to pre-dispatch

    from shadow import _core

    records = list(session._records)
    parent_id = records[-1]["id"] if records else "sha256:none"
    probes: list[dict[str, Any]] = []
    for block in tool_uses:
        payload = {
            "tool_name": str(block.get("name") or ""),
            "tool_call_id": str(block.get("id") or ""),
            "arguments": dict(block.get("input") or {}),
        }
        probes.append(
            {
                "version": "0.1",
                "id": _core.content_id(payload),
                "kind": "tool_call",
                "ts": "1970-01-01T00:00:00.000Z",
                "parent": parent_id,
                "payload": payload,
            }
        )
    verdict = enforcer.probe([*records, *probes])
    if verdict.allow:
        return
    mode = enforcer.on_violation
    if mode == "warn":
        import logging

        logging.getLogger("shadow.policy_runtime").warning(
            "auto-instrument: %d tool_use block(s) would be blocked but "
            "warn mode is set; passing through. Violations: %s",
            len(tool_uses),
            verdict.reason,
        )
        return
    # raise + replace both surface as PolicyViolationError at the
    # auto-instrument layer. The error carries every violation so
    # caller's handler can branch.
    raise PolicyViolationError(verdict.violations)


# ---------------------------------------------------------------------------
# Translators: SDK kwargs / responses → Shadow chat_request / chat_response.
# ---------------------------------------------------------------------------


def _is_omitted(value: Any) -> bool:
    """Return True for the OpenAI / Anthropic SDK 'unset' sentinels.

    Both SDKs ship a placeholder type to distinguish "field not provided"
    from "field is null" — ``openai.Omit``, ``openai.NotGiven`` (and
    matching internal ``_types`` aliases), and ``anthropic.NotGiven``.
    These leak into the kwargs dict for any optional parameter the
    caller didn't pass, and the canonical-JSON layer in the Rust core
    rejects them with ``ValueError: unsupported type Omit``. Strip
    them at the translator boundary.
    """
    cls_name = type(value).__name__
    if cls_name in {"Omit", "NotGiven"}:
        return True
    # Some SDK builds expose the same sentinels under module-private
    # paths (e.g. openai._types.Omit). Match by qualname suffix as a
    # safety net so the check survives SDK refactors.
    qualname = getattr(type(value), "__qualname__", "") or ""
    return qualname.endswith(".Omit") or qualname.endswith(".NotGiven")


def _strip_omitted(value: Any) -> Any:
    """Recursively drop ``Omit`` / ``NotGiven`` sentinels from a value.

    Dict keys whose value is omitted are dropped entirely. List items
    that are omitted are filtered out. Anything else passes through.
    """
    if _is_omitted(value):
        return _OMIT_MARKER
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            cleaned = _strip_omitted(v)
            if cleaned is _OMIT_MARKER:
                continue
            out[k] = cleaned
        return out
    if isinstance(value, list):
        cleaned_list = [_strip_omitted(v) for v in value]
        return [v for v in cleaned_list if v is not _OMIT_MARKER]
    return value


_OMIT_MARKER = object()


def _anthropic_req_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """anthropic.messages.create(**kwargs) → Shadow chat_request payload."""
    kwargs = _strip_omitted(kwargs)
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
    kwargs = _strip_omitted(kwargs)
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
    """openai ChatCompletion → Shadow chat_response payload.

    LiteLLM's ``ModelResponse`` is intentionally shape-compatible with
    ``openai.ChatCompletion`` (same ``.choices[0].message.content``
    layout, same ``.usage.prompt_tokens`` etc.), so this translator
    serves both backends.
    """
    from shadow.llm.openai_backend import OpenAILLM

    return OpenAILLM._from_provider(response, latency_ms)


# ---------------------------------------------------------------------------
# LiteLLM translators.
#
# LiteLLM presents the OpenAI-shape kwargs/response, so the existing
# `_openai_resp` translator works for the response side. For the request
# side we use a thin wrapper that strips LiteLLM-specific kwargs that
# would surface as canonical-JSON garbage downstream (api_base, api_key,
# custom_llm_provider, fallbacks, model_list, …) before delegating to
# the OpenAI translator.
# ---------------------------------------------------------------------------

_LITELLM_KWARG_DENYLIST: frozenset[str] = frozenset(
    {
        # Routing / config — not behaviour-relevant for diff/replay.
        "api_base",
        "api_key",
        "api_version",
        "custom_llm_provider",
        "fallbacks",
        "model_list",
        "router",
        "metadata",
        "extra_headers",
        "request_timeout",
        "timeout",
        "litellm_call_id",
        "litellm_logging_obj",
        "litellm_session_id",
        "user",
        # Caching layer config — also not behaviour-relevant.
        "caching",
        "cache",
        # Logging / callback hooks.
        "logger_fn",
        "logger",
    }
)


def _litellm_req_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """litellm.completion(**kwargs) → Shadow chat_request payload."""
    cleaned = {k: v for k, v in kwargs.items() if k not in _LITELLM_KWARG_DENYLIST}
    return _openai_req_from_kwargs(cleaned)


# ---------------------------------------------------------------------------
# LangChain ChatOpenAI translators.
#
# LangChain's BaseChatOpenAI._generate(messages, stop=None,
# run_manager=None, **kwargs) takes positional `messages` (a list of
# langchain_core.messages.BaseMessage instances) and returns a
# ChatResult containing one or more ChatGenerations whose .message is
# an AIMessage. We translate at this layer (rather than at the
# underlying OpenAI client) because LangChain captures bound-method
# references at import time, bypassing class-level monkey-patches on
# `Completions.create`.
# ---------------------------------------------------------------------------


def _langchain_chat_req_from_args(translator_input: Any) -> dict[str, Any]:
    """LangChain BaseChatOpenAI._generate args → Shadow chat_request payload.

    ``translator_input`` is the ``(args, kwargs)`` tuple captured by the
    ``arg_passthrough`` wrapper. ``args[0]`` is the BaseMessage list,
    ``args[1]`` is optional ``stop``. Model + sampling params live on
    the bound ``self`` instance, but our wrapper doesn't have access
    to ``self`` from the translator boundary; we read what we can from
    ``kwargs`` (which carries any per-call overrides) and pull the rest
    from any ``invocation_params``-shaped fallback.
    """
    args, kwargs = (
        translator_input if isinstance(translator_input, tuple) else ((), translator_input)
    )
    messages_in = args[0] if args else kwargs.get("messages") or []
    messages: list[dict[str, Any]] = []
    for msg in messages_in:
        role = _langchain_role_for(msg)
        content = _langchain_content_for(msg)
        if role:
            messages.append({"role": role, "content": content})
    params: dict[str, Any] = {}
    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("max_completion_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop"),
    ):
        if src in kwargs and kwargs[src] is not None:
            params[dst] = kwargs[src]
    # When LangChain invokes _generate, the model id is on the bound
    # instance (self.model_name / self.model). The wrapper doesn't pass
    # `self` through to the translator (translator gets only args/kwargs),
    # so `model` is best-effort: use the kwargs override if present, else
    # leave empty so the canonical-JSON path emits "" rather than crashing.
    out: dict[str, Any] = {
        "model": kwargs.get("model") or "",
        "messages": messages,
        "params": params,
    }
    return out


def _langchain_chat_resp(result: Any, latency_ms: int) -> dict[str, Any]:
    """LangChain ChatResult → Shadow chat_response payload."""
    # ChatResult.generations is list[ChatGeneration]; .message is AIMessage.
    content: list[dict[str, Any]] = []
    model = ""
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    thinking_tokens = 0

    generations = getattr(result, "generations", None) or []
    if generations:
        gen0 = generations[0]
        msg = getattr(gen0, "message", None)
        if msg is not None:
            text = getattr(msg, "content", "") or ""
            if isinstance(text, str) and text:
                content.append({"type": "text", "text": text})
            elif isinstance(text, list):
                # Newer LangChain emits multimodal content as list of parts.
                for part in text:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content.append({"type": "text", "text": part.get("text", "")})
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                        "name": tc.get("name", "")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", ""),
                        "input": tc.get("args", {})
                        if isinstance(tc, dict)
                        else getattr(tc, "args", {}),
                    }
                )
            usage_md = getattr(msg, "usage_metadata", None) or {}
            if isinstance(usage_md, dict):
                input_tokens = int(usage_md.get("input_tokens") or 0)
                output_tokens = int(usage_md.get("output_tokens") or 0)
                thinking_tokens = int(
                    usage_md.get("output_token_details", {}).get("reasoning", 0) or 0
                )
        gen_info = getattr(gen0, "generation_info", None) or {}
        if isinstance(gen_info, dict):
            model = gen_info.get("model_name") or gen_info.get("model") or ""
            fr = gen_info.get("finish_reason") or ""
            if fr:
                stop_reason = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "function_call": "tool_use",
                    "content_filter": "content_filter",
                }.get(fr, fr)

    # llm_output (older LangChain layout) carries token usage at the
    # ChatResult level — fall back to it when usage_metadata is absent.
    if input_tokens == 0 and output_tokens == 0:
        llm_out = getattr(result, "llm_output", None) or {}
        token_usage = (llm_out.get("token_usage") if isinstance(llm_out, dict) else None) or {}
        input_tokens = int(token_usage.get("prompt_tokens") or 0)
        output_tokens = int(token_usage.get("completion_tokens") or 0)
        if isinstance(llm_out, dict) and not model:
            model = llm_out.get("model_name") or llm_out.get("model") or ""

    return {
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
        },
    }


def _langchain_role_for(msg: Any) -> str:
    """Map a langchain_core.messages.BaseMessage to a Shadow role string."""
    cls_name = type(msg).__name__
    return {
        "SystemMessage": "system",
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "AIMessageChunk": "assistant",
        "ToolMessage": "tool",
        "FunctionMessage": "tool",
        "ChatMessage": getattr(msg, "role", "user") or "user",
    }.get(cls_name, "user")


def _langchain_content_for(msg: Any) -> Any:
    """Extract content from a langchain BaseMessage; preserve list shape if multimodal."""
    content = getattr(msg, "content", "")
    # Tool messages carry a tool_call_id; preserve it inline so policy
    # rules that match on tool-result text still see it.
    return content if content is not None else ""


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
    kwargs = _strip_omitted(kwargs)
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
    translator_input: Any,
    start: float,
) -> Any:
    """Wrap a sync stream; preserve the underlying object's full surface.

    The OpenAI SDK's ``Stream`` is BOTH iterable AND a context manager
    (``with stream: ...`` closes the underlying httpx connection).
    Returning a plain generator from this wrapper drops the context-
    manager methods, which breaks LangChain (and any caller using
    ``with client.chat.completions.create(stream=True): ...``).

    The :class:`_SyncStreamProxy` proxies every attribute access to the
    original stream and adds chunk capture on iteration. Identity
    (``isinstance``, ``type()``) checks are not preserved, but the
    ``async/sync iterator + context manager`` duck-typed surface is.
    """
    return _SyncStreamProxy(
        stream, session, req_translator, resp_translator, translator_input, start
    )


def _wrap_async_stream(
    stream: Any,
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    translator_input: Any,
    start: float,
) -> Any:
    """Wrap an async stream; preserve full surface (iterator + context mgr).

    See :func:`_wrap_sync_stream`. LangChain's ``ChatOpenAI(streaming=True)
    .astream(...)`` does ``async with ...`` on the streaming response.
    Before this fix, the wrapper returned an async generator which has
    no ``__aenter__``/``__aexit__`` and surfaced as
    ``TypeError: 'async_generator' object does not support the
    asynchronous context manager protocol`` — now-fixed.
    """
    return _AsyncStreamProxy(
        stream, session, req_translator, resp_translator, translator_input, start
    )


class _SyncStreamProxy:
    """Capturing proxy around a sync streaming response.

    Preserves: iteration (``for x in stream``), context-manager
    (``with stream``), and arbitrary attribute access on the
    underlying object. Iteration accumulates chunks; on close /
    iterator exhaustion / context-manager exit, the aggregated
    chat_response record is written to the active session.
    """

    def __init__(
        self,
        stream: Any,
        session: Session,
        req_translator: Any,
        resp_translator: Any,
        translator_input: Any,
        start: float,
    ) -> None:
        # Use object.__setattr__ to avoid triggering our own
        # __setattr__ proxy below before _stream is set.
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_req_translator", req_translator)
        object.__setattr__(self, "_resp_translator", resp_translator)
        object.__setattr__(self, "_translator_input", translator_input)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_chunks", [])
        object.__setattr__(self, "_recorded", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._stream, name, value)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._record_once()
            raise
        self._chunks.append(chunk)
        return chunk

    def __enter__(self) -> Any:
        # Delegate to the underlying object's context-manager methods
        # if it has them (OpenAI Stream does); otherwise return self
        # so `with stream:` still works on plain iterators.
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc: Any) -> Any:
        # Record first (best-effort, never raises into the user's
        # `with` block). Then delegate to the underlying object's
        # __exit__ if present, returning whatever it returns so
        # exception suppression semantics are preserved.
        with contextlib.suppress(Exception):
            self._record_once()
        exit_fn = getattr(self._stream, "__exit__", None)
        if exit_fn is not None:
            return exit_fn(*exc)
        return False

    def close(self) -> Any:
        with contextlib.suppress(Exception):
            self._record_once()
        close_fn = getattr(self._stream, "close", None)
        if close_fn is not None:
            return close_fn()
        return None

    def _record_once(self) -> None:
        if self._recorded:
            return
        object.__setattr__(self, "_recorded", True)
        latency_ms = int((time.perf_counter() - self._start) * 1000)
        _record_stream_safely(
            self._session,
            self._req_translator,
            self._resp_translator,
            self._translator_input,
            self._chunks,
            latency_ms,
        )


class _AsyncStreamProxy:
    """Capturing proxy around an async streaming response.

    Async equivalent of :class:`_SyncStreamProxy`. Preserves iteration
    (``async for``), async context-manager (``async with``), and
    arbitrary attribute access on the underlying object.
    """

    def __init__(
        self,
        stream: Any,
        session: Session,
        req_translator: Any,
        resp_translator: Any,
        translator_input: Any,
        start: float,
    ) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_req_translator", req_translator)
        object.__setattr__(self, "_resp_translator", resp_translator)
        object.__setattr__(self, "_translator_input", translator_input)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_chunks", [])
        object.__setattr__(self, "_recorded", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._stream, name, value)

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            await self._record_once()
            raise
        self._chunks.append(chunk)
        return chunk

    async def __aenter__(self) -> Any:
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        # Best-effort recording first; suppress recording-layer failures
        # so they never break the caller's `async with` block. Then
        # delegate to the wrapped stream's __aexit__ to preserve its
        # close behaviour and exception-suppression semantics.
        await self._record_once_safe()
        exit_fn = getattr(self._stream, "__aexit__", None)
        if exit_fn is not None:
            return await exit_fn(*exc)
        return False

    async def aclose(self) -> Any:
        await self._record_once_safe()
        close_fn = getattr(self._stream, "aclose", None)
        if close_fn is not None:
            return await close_fn()
        return None

    async def _record_once_safe(self) -> None:
        """``_record_once`` wrapped to never propagate exceptions.

        Recording must never break the caller's `async with` block —
        a recording-layer failure is silenced and the user's stream
        keeps working.
        """
        with contextlib.suppress(Exception):
            await self._record_once()

    async def _record_once(self) -> None:
        if self._recorded:
            return
        object.__setattr__(self, "_recorded", True)
        latency_ms = int((time.perf_counter() - self._start) * 1000)
        _record_stream_safely(
            self._session,
            self._req_translator,
            self._resp_translator,
            self._translator_input,
            self._chunks,
            latency_ms,
        )


def _record_stream_safely(
    session: Session,
    req_translator: Any,
    resp_translator: Any,
    translator_input: Any,
    chunks: list[Any],
    latency_ms: int,
) -> None:
    """Best-effort: aggregate chunks into a synthetic response and record."""
    try:
        # The translators are chat-shaped so we need to build a ChatCompletion/
        # Message-shaped object from the chunks. We detect shape from the chunk
        # attributes; fall back to stringifying if we can't.
        aggregated = _aggregate_chunks(chunks)
        shadow_req = req_translator(translator_input)
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
