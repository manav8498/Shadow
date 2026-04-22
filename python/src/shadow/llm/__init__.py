"""Pluggable LLM backends for shadow.

- `MockLLM` — deterministic file-backed replayer (content-id lookup).
- `PositionalMockLLM` — positional replay for demos/tests where request
  payloads don't match by content id.
- `AnthropicLLM` / `OpenAILLM` are deferred to v0.2; v0.1 records and
  replays against mocks, and users can wrap their own clients with the
  `shadow.sdk.Session` instrumentation.
"""

from shadow.llm.base import LlmBackend
from shadow.llm.mock import MockLLM
from shadow.llm.positional import PositionalMockLLM

__all__ = ["LlmBackend", "MockLLM", "PositionalMockLLM"]
