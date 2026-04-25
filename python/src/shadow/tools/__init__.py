"""Tool execution backends for shadow's agent-loop replay.

This module is the analog of :mod:`shadow.llm` for the *other* side
of an agent loop: the tool-call side. Existing replay walks a
recorded trace in lock-step (request → ``LlmBackend`` → response,
copy-through tools); the agent-loop replay engine drives the loop
forward instead, dispatching candidate model decisions to a real
``ToolBackend`` so we can answer "what would the candidate have
done?" without touching production.

Three backends ship:

- :class:`~shadow.tools.replay.ReplayToolBackend` — deterministic.
  Indexes the baseline's ``tool_call`` / ``tool_result`` pairs by
  ``(tool_name, canonical_args_hash)`` and serves the recorded
  result. The default for `shadow replay --agent-loop`.
- :class:`~shadow.tools.sandbox.SandboxedToolBackend` — wraps the
  user's actual tool functions but blocks side effects (network,
  filesystem writes, subprocess) via monkey-patching. Best-effort
  isolation for replay determinism, not a security boundary.
- :class:`~shadow.tools.stub.StubToolBackend` — generic mock for
  tests. Returns deterministic placeholder results.

The protocol mirrors :class:`shadow.llm.base.LlmBackend`: a single
async ``execute`` method plus a stable ``id`` property. Implementing
your own backend is a thirty-line module away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shadow.tools.base import ToolBackend, ToolCall, ToolResult, canonical_args_hash
from shadow.tools.replay import ReplayToolBackend
from shadow.tools.stub import StubToolBackend

if TYPE_CHECKING:  # pragma: no cover
    from shadow.tools.sandbox import SandboxedToolBackend


def __getattr__(name: str) -> Any:
    """Lazy-import the sandbox backend so the import-time cost of its
    monkey-patching machinery isn't paid by users who never touch it.
    """
    if name == "SandboxedToolBackend":
        from shadow.tools.sandbox import SandboxedToolBackend

        return SandboxedToolBackend
    raise AttributeError(f"module 'shadow.tools' has no attribute {name!r}")


def get_tool_backend(name: str, **kwargs: Any) -> ToolBackend:
    """Return a tool backend instance by name.

    Supported: ``"replay"``, ``"sandbox"``, ``"stub"``.
    """
    if name == "replay":
        return ReplayToolBackend(**kwargs)
    if name == "stub":
        return StubToolBackend(**kwargs)
    if name == "sandbox":
        from shadow.tools.sandbox import SandboxedToolBackend

        return SandboxedToolBackend(**kwargs)
    from shadow.errors import ShadowConfigError

    raise ShadowConfigError(f"unknown tool backend {name!r}; supported: replay, sandbox, stub")


__all__ = [
    "ReplayToolBackend",
    "SandboxedToolBackend",
    "StubToolBackend",
    "ToolBackend",
    "ToolCall",
    "ToolResult",
    "canonical_args_hash",
    "get_tool_backend",
]
