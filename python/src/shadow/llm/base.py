"""`LlmBackend` Protocol — the Python analogue of the Rust LlmBackend trait."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LlmBackend(Protocol):
    """An async backend that turns chat_request payloads into chat_response payloads.

    See SPEC §10 for the replay semantics. Implementations:
    - `shadow.llm.MockLLM` (deterministic file-backed replayer)
    - `shadow.llm.AnthropicLLM` (live API, requires the `anthropic` extra)
    - `shadow.llm.OpenAILLM`  (live API, requires the `openai` extra)
    """

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        """Map a chat_request payload to a chat_response payload."""
        ...

    @property
    def id(self) -> str:
        """Stable identifier, e.g. "mock" or "anthropic"."""
        ...
