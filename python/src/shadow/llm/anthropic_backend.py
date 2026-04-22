"""Live Anthropic backend.

Wraps the `anthropic` Python SDK. Gated behind the `[anthropic]` extra
so the default install stays dep-light. Env var `ANTHROPIC_API_KEY`
is read by the SDK; the backend also accepts an explicit `api_key=`.

The backend converts between Shadow's `.agentlog` `chat_request` /
`chat_response` payload shape and Anthropic's `messages.create` API:

- The system message (if present at position 0 of `messages`) is
  pulled out and passed to Anthropic as the `system=` parameter (the
  Anthropic API takes system as a top-level arg, not a message).
- `params.temperature`, `params.top_p`, `params.max_tokens`,
  `params.stop` are forwarded.
- `tools` are passed through verbatim; Anthropic's schema matches
  Shadow's.
- `usage.input_tokens` / `usage.output_tokens` come straight from
  the response.
- `latency_ms` is wall-clock on the caller side.
"""

from __future__ import annotations

import time
from typing import Any

from shadow.errors import ShadowBackendError


class AnthropicLLM:
    """Live Anthropic backend — implements the `shadow.llm.LlmBackend` Protocol."""

    def __init__(
        self,
        model_override: str | None = None,
        api_key: str | None = None,
        backend_id: str = "anthropic",
    ) -> None:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ShadowBackendError(
                "anthropic SDK not installed\n"
                "hint: pip install 'shadow[anthropic]' (or add anthropic==0.40.0)"
            ) from e
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model_override = model_override
        self._id = backend_id

    @property
    def id(self) -> str:
        return self._id

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        provider_request = self._to_provider(request)
        start = time.perf_counter()
        try:
            response = await self._client.messages.create(**provider_request)
        except Exception as e:
            raise ShadowBackendError(f"anthropic API error: {e}") from e
        latency_ms = int((time.perf_counter() - start) * 1000)
        return self._from_provider(response, latency_ms)

    def _to_provider(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert a Shadow chat_request payload → anthropic.messages.create kwargs."""
        model = self._model_override or request["model"]
        messages = list(request.get("messages", []))
        # Anthropic takes `system` as a top-level parameter, not a message.
        system = None
        if messages and messages[0].get("role") == "system":
            system = messages[0].get("content")
            messages = messages[1:]
        params = request.get("params") or {}
        tools = request.get("tools") or []

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": params.get("max_tokens", 1024),
        }
        if system is not None:
            kwargs["system"] = system
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            kwargs["top_p"] = params["top_p"]
        if params.get("stop") is not None:
            kwargs["stop_sequences"] = params["stop"]
        if tools:
            kwargs["tools"] = tools
        return kwargs

    @staticmethod
    def _from_provider(response: Any, latency_ms: int) -> dict[str, Any]:
        """Convert an anthropic response → Shadow chat_response payload."""
        content: list[dict[str, Any]] = []
        for part in response.content:
            ptype = getattr(part, "type", None)
            if ptype == "text":
                content.append({"type": "text", "text": part.text})
            elif ptype == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": part.id,
                        "name": part.name,
                        "input": part.input,
                    }
                )
            elif ptype == "thinking":
                content.append(
                    {
                        "type": "thinking",
                        "text": getattr(part, "text", getattr(part, "thinking", "")),
                    }
                )
            # Unknown part types are dropped; Shadow's SPEC allows the
            # four types above.
        usage = getattr(response, "usage", None)
        thinking_tokens = 0
        if usage is not None:
            thinking_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        return {
            "model": getattr(response, "model", ""),
            "content": content,
            "stop_reason": getattr(response, "stop_reason", "end_turn"),
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "thinking_tokens": thinking_tokens,
            },
        }
