"""MockLLM — deterministic Python backend that replays recorded responses.

Same semantics as the Rust `shadow_core::replay::mock::MockLlm`:
build from one or more traces, index responses by their parent chat_request's
content id, look them up on `.complete()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow import _core
from shadow.errors import ShadowBackendError


class MockLLM:
    """File-backed mock backend."""

    def __init__(self, responses: dict[str, dict[str, Any]], backend_id: str = "mock") -> None:
        self._responses = responses
        self._id = backend_id

    @property
    def id(self) -> str:
        return self._id

    def __len__(self) -> int:
        return len(self._responses)

    @classmethod
    def from_trace(cls, trace: list[dict[str, Any]], backend_id: str = "mock") -> MockLLM:
        """Build from a single parsed trace (a list of record dicts)."""
        responses: dict[str, dict[str, Any]] = {}
        for record in trace:
            if record.get("kind") == "chat_response" and record.get("parent"):
                responses[record["parent"]] = record["payload"]
        return cls(responses, backend_id=backend_id)

    @classmethod
    def from_traces(cls, traces: list[list[dict[str, Any]]], backend_id: str = "mock") -> MockLLM:
        """Build from multiple traces (a trace set). Later traces override earlier."""
        responses: dict[str, dict[str, Any]] = {}
        for trace in traces:
            for record in trace:
                if record.get("kind") == "chat_response" and record.get("parent"):
                    responses[record["parent"]] = record["payload"]
        return cls(responses, backend_id=backend_id)

    @classmethod
    def from_path(cls, path: Path | str, backend_id: str = "mock") -> MockLLM:
        """Load a single `.agentlog` file from disk."""
        data = Path(path).read_bytes()
        trace = _core.parse_agentlog(data)
        return cls.from_trace(trace, backend_id=backend_id)

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = _core.content_id(request)
        try:
            return dict(self._responses[request_id])
        except KeyError as exc:
            raise ShadowBackendError(
                f"MockLLM has no recorded response for request id {request_id}\n"
                "hint: either re-record the baseline or switch to --backend live"
            ) from exc
