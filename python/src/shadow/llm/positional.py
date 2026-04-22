"""PositionalMockLLM — variant of MockLLM that matches by request *order*.

Used by the demo: the baseline trace and the reference trace were recorded
with different configs, so the candidate's requests have different content
ids from the baseline's. Strict `content_id` lookup would miss every
request. PositionalMockLLM instead maps the i-th request it sees (in call
order) to the i-th recorded response in the reference trace.

This backend is strictly intended for DEMOS and integration tests that
pre-recorded both sides of a shadow run. Production replays should use
the content-id-keyed `MockLLM` against a baseline whose request payloads
will actually match, or a live backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow import _core
from shadow.errors import ShadowBackendError


class PositionalMockLLM:
    """Return responses in the order they were recorded, regardless of request shape."""

    def __init__(
        self, responses: list[dict[str, Any]], backend_id: str = "positional-mock"
    ) -> None:
        self._responses = list(responses)
        self._id = backend_id
        self._cursor = 0

    @property
    def id(self) -> str:
        return self._id

    def __len__(self) -> int:
        return len(self._responses)

    @classmethod
    def from_trace(
        cls, trace: list[dict[str, Any]], backend_id: str = "positional-mock"
    ) -> PositionalMockLLM:
        responses = [record["payload"] for record in trace if record.get("kind") == "chat_response"]
        return cls(responses, backend_id=backend_id)

    @classmethod
    def from_path(cls, path: Path | str, backend_id: str = "positional-mock") -> PositionalMockLLM:
        data = Path(path).read_bytes()
        trace = _core.parse_agentlog(data)
        return cls.from_trace(trace, backend_id=backend_id)

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._cursor >= len(self._responses):
            raise ShadowBackendError(
                "PositionalMockLLM exhausted: more requests than recorded responses\n"
                "hint: record a longer reference trace, or switch to MockLLM with a full baseline"
            )
        response = dict(self._responses[self._cursor])
        self._cursor += 1
        return response
