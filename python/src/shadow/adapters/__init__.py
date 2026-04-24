"""First-class adapters for agent frameworks.

Each submodule exposes a tracing hook tailored to a specific framework's
instrumentation surface — LangGraph callbacks, CrewAI event listeners,
AG2 hooks — and routes the captured data through
``shadow.sdk.Session.record_chat`` / ``record_tool_call`` /
``record_tool_result``.

Adapters are imported lazily (only when the user installs the matching
extra, e.g. ``pip install 'shadow-diff[langgraph]'``) so the main SDK
stays dependency-light.
"""

from __future__ import annotations

__all__ = ["__doc__"]
