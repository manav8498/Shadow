"""Pluggable LLM backends for shadow.

- `MockLLM` — deterministic file-backed replayer (content-id lookup).
- `PositionalMockLLM` — positional replay for demos/tests where request
  payloads don't match by content id.
- `AnthropicLLM` — live Anthropic API backend (requires the `[anthropic]`
  extra). Import lazily via `shadow.llm.get_anthropic()` so the default
  `pip install shadow` doesn't require the `anthropic` SDK.
- `OpenAILLM` — live OpenAI API backend (requires the `[openai]` extra).

Direct import paths (`from shadow.llm import AnthropicLLM`) also work
when the respective extra is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shadow.llm.base import LlmBackend
from shadow.llm.mock import MockLLM
from shadow.llm.positional import PositionalMockLLM

if TYPE_CHECKING:
    from shadow.llm.anthropic_backend import AnthropicLLM
    from shadow.llm.openai_backend import OpenAILLM


def __getattr__(name: str) -> Any:
    """Lazy-import live backends so missing optional deps don't break
    `from shadow.llm import MockLLM`.
    """
    if name == "AnthropicLLM":
        from shadow.llm.anthropic_backend import AnthropicLLM

        return AnthropicLLM
    if name == "OpenAILLM":
        from shadow.llm.openai_backend import OpenAILLM

        return OpenAILLM
    raise AttributeError(f"module 'shadow.llm' has no attribute {name!r}")


def get_backend(name: str, **kwargs: Any) -> LlmBackend:
    """Return a live backend instance by name. Raises ShadowBackendError if
    the corresponding extra isn't installed.

    Supported names: `"anthropic"`, `"openai"`, `"mock"` (returns an
    empty MockLLM), `"positional"` (requires `reference_path=` kwarg).
    """
    if name == "anthropic":
        from shadow.llm.anthropic_backend import AnthropicLLM

        return AnthropicLLM(**kwargs)
    if name == "openai":
        from shadow.llm.openai_backend import OpenAILLM

        return OpenAILLM(**kwargs)
    if name == "mock":
        return MockLLM({})
    if name == "positional":
        reference_path = kwargs.pop("reference_path", None)
        if reference_path is None:
            from shadow.errors import ShadowConfigError

            raise ShadowConfigError(
                "positional backend requires reference_path=<path-to-candidate.agentlog>"
            )
        return PositionalMockLLM.from_path(reference_path, **kwargs)
    from shadow.errors import ShadowConfigError

    raise ShadowConfigError(
        f"unknown backend {name!r}; supported: anthropic, openai, mock, positional"
    )


__all__ = [
    "AnthropicLLM",
    "LlmBackend",
    "MockLLM",
    "OpenAILLM",
    "PositionalMockLLM",
    "get_backend",
]
