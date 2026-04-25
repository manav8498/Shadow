"""Live OpenAI backend.

Wraps the `openai` Python SDK (Chat Completions shape). Gated behind
the `[openai]` extra. Env var `OPENAI_API_KEY` is read by the SDK;
the backend also accepts an explicit `api_key=`.

Shadow's `.agentlog` request/response shape is closer to Anthropic's
than OpenAI's, so this backend does more reshaping:

- Shadow `tool_use` / `tool_result` content parts ↔ OpenAI
  `tool_calls` / `role: tool` messages.
- `params.max_tokens` → `max_completion_tokens`.
- OpenAI's `content` is either a string or a list of parts; we always
  produce the list form on output (SPEC §4.2).
- `usage.completion_tokens_details.reasoning_tokens` → `thinking_tokens`
  when present (GPT-5+ exposes it).
"""

from __future__ import annotations

import time
from typing import Any

from shadow.errors import ShadowBackendError


class OpenAILLM:
    """Live OpenAI backend — implements the `shadow.llm.LlmBackend` Protocol."""

    # Default model used when neither the caller nor the request payload
    # carries a model name. gpt-4o-mini is the cheapest + fastest usable
    # model right now; callers who care about quality should override.
    DEFAULT_MODEL: str = "gpt-4o-mini"

    def __init__(
        self,
        model_override: str | None = None,
        api_key: str | None = None,
        backend_id: str = "openai",
    ) -> None:
        try:
            import openai  # type: ignore[import-not-found, unused-ignore]
        except ImportError as e:
            raise ShadowBackendError(
                "openai SDK not installed\n"
                "hint: pip install 'shadow[openai]' (or add openai==1.58.1)"
            ) from e
        self._openai = openai
        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        # Explicit timeout — see AnthropicLLM for rationale.
        kwargs.setdefault("timeout", 60.0)
        self._client = openai.AsyncOpenAI(**kwargs)
        self._model_override = model_override
        self._id = backend_id

    @property
    def id(self) -> str:
        return self._id

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        provider_request = self._to_provider(request)
        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**provider_request)
        except Exception as e:
            raise ShadowBackendError(f"openai API error: {e}") from e
        latency_ms = int((time.perf_counter() - start) * 1000)
        return self._from_provider(response, latency_ms)

    def _to_provider(self, request: dict[str, Any]) -> dict[str, Any]:
        # Model resolution: explicit override beats request payload; if
        # both are empty we fall back to DEFAULT_MODEL rather than sending
        # model="" to the API (which rejects with invalid_request_error).
        model = self._model_override or request.get("model") or self.DEFAULT_MODEL
        provider_messages: list[dict[str, Any]] = []
        for m in request.get("messages", []):
            provider_messages.append(self._convert_message_to_openai(m))
        params = request.get("params") or {}
        tools = request.get("tools") or []

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": provider_messages,
        }
        if "max_tokens" in params:
            kwargs["max_completion_tokens"] = params["max_tokens"]
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            kwargs["top_p"] = params["top_p"]
        if params.get("stop") is not None:
            kwargs["stop"] = params["stop"]
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]
        return kwargs

    @staticmethod
    def _convert_message_to_openai(m: dict[str, Any]) -> dict[str, Any]:
        """Map a Shadow/Anthropic-style message to OpenAI format.

        Shadow messages use `role ∈ {system, user, assistant, tool}`.
        Content is either a string or a list of parts. OpenAI accepts a
        similar shape but tool results are separate messages, not
        embedded content parts.
        """
        out: dict[str, Any] = {"role": m["role"]}
        content = m.get("content")
        # Top-level tool_calls / tool_call_id must always forward — the
        # agent-loop engine emits assistant messages with string content
        # AND a tool_calls field (the OpenAI wire shape), and tool-role
        # messages carry tool_call_id. Both are required by the API on
        # follow-up requests; dropping them produces a 400 "messages
        # with role 'tool' must be a response to a preceding message
        # with 'tool_calls'."
        if "tool_calls" in m:
            out["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m:
            out["tool_call_id"] = m["tool_call_id"]
        if isinstance(content, str):
            out["content"] = content
            return out
        if not isinstance(content, list):
            out["content"] = content
            return out
        # Content is a list of parts. Separate text from tool_use / tool_result.
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_result: dict[str, Any] | None = None
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else None
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype == "tool_use":
                tool_calls.append(
                    {
                        "id": part.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": _dump_json(part.get("input", {})),
                        },
                    }
                )
            elif ptype == "tool_result":
                # tool_result gets turned into a separate `role: tool`
                # message by the caller; we signal it here.
                tool_result = part
        if tool_result is not None:
            # Overrides: convert this whole message to a tool role
            return {
                "role": "tool",
                "tool_call_id": tool_result.get("tool_use_id", ""),
                "content": str(tool_result.get("content", "")),
            }
        if tool_calls:
            out["tool_calls"] = tool_calls
        if text_parts:
            out["content"] = "\n".join(text_parts)
        elif "content" not in out:
            out["content"] = ""
        return out

    @staticmethod
    def _from_provider(response: Any, latency_ms: int) -> dict[str, Any]:
        choice = response.choices[0]
        message = choice.message
        content: list[dict[str, Any]] = []
        msg_content = getattr(message, "content", None)
        if isinstance(msg_content, str) and msg_content:
            content.append({"type": "text", "text": msg_content})
        elif isinstance(msg_content, list):
            # Newer OpenAI APIs (vision, structured output) return an array
            # of content parts. Preserve text parts; keep the rest as-is.
            for part in msg_content:
                if isinstance(part, dict):
                    content.append(dict(part))
        # `message.refusal` (gpt-4o+) carries the model's refusal text when it
        # declines to answer. Preserve it so safety/conformance axes can see
        # the refusal instead of treating the response as empty.
        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal:
            content.append({"type": "refusal", "text": refusal})
        for tc in getattr(message, "tool_calls", None) or []:
            fn = tc.function
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": fn.name,
                    "input": _parse_json(fn.arguments),
                }
            )
        usage = getattr(response, "usage", None)
        thinking_tokens = 0
        cached_input_tokens = 0
        if usage is not None:
            comp_details = getattr(usage, "completion_tokens_details", None)
            if comp_details is not None:
                thinking_tokens = getattr(comp_details, "reasoning_tokens", 0) or 0
            # OpenAI's automatic prompt caching (gpt-4o+, prompts >1024 tok)
            # reports cache hits in prompt_tokens_details.cached_tokens.
            # Without routing this to cached_input_tokens, the cost axis
            # bills every cached call at the full uncached rate — same class
            # of bug Anthropic just had.
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        stop_reason_raw = getattr(choice, "finish_reason", "stop")
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
        }.get(stop_reason_raw, stop_reason_raw)
        usage_out: dict[str, Any] = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
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


def _dump_json(obj: Any) -> str:
    import json

    return json.dumps(obj)


def _parse_json(s: str | dict[str, Any]) -> dict[str, Any]:
    import json

    if isinstance(s, dict):
        return s
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else {"_raw": s}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": s}
