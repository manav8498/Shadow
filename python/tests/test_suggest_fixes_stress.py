"""Retry / scale / concurrency stress tests for suggest_fixes.

Fills the gaps from the v1.2.0 stress pass:
- Retry/backoff on transient errors (429/503/timeout)
- Scale behaviour on 10k-pair traces (evidence-truncation must bound prompt)
- Concurrent async callers share backend state safely
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from shadow.suggest_fixes import (
    _complete_with_retry,
    _is_transient,
    suggest_fixes,
)

# ---- helpers --------------------------------------------------------------


def _report_with_recs(n_recs: int = 2) -> dict[str, Any]:
    recs = []
    for i in range(n_recs):
        recs.append(
            {
                "id": f"rec-{i}",
                "title": f"Fix thing {i}",
                "severity": "error" if i % 2 == 0 else "warning",
                "axis": "trajectory",
                "turn_index": i,
                "action": "review",
                "rationale": "test",
            }
        )
    return {
        "pair_count": 10,
        "rows": [
            {
                "axis": "trajectory",
                "severity": "severe",
                "baseline_median": 0,
                "candidate_median": 1,
                "delta": 1,
            }
        ],
        "recommendations": recs,
        "first_divergence": {"turn_index": 0, "kind": "structural_drift"},
    }


def _synthetic_records(turns: int, text_size: int = 200) -> list[dict[str, Any]]:
    """Build `turns` chat_request/chat_response pairs. `text_size` chars of text each."""
    body = "x" * text_size
    recs: list[dict[str, Any]] = [{"kind": "metadata", "id": "meta", "parent": None, "payload": {}}]
    for i in range(turns):
        req_id = f"req{i}"
        resp_id = f"resp{i}"
        recs.append(
            {
                "kind": "chat_request",
                "id": req_id,
                "parent": recs[-1]["id"],
                "payload": {
                    "model": "m",
                    "messages": [{"role": "user", "content": f"q{i}: {body}"}],
                    "params": {},
                },
            }
        )
        recs.append(
            {
                "kind": "chat_response",
                "id": resp_id,
                "parent": req_id,
                "payload": {
                    "model": "m",
                    "content": [{"type": "text", "text": f"resp{i}: {body}"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 10,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            }
        )
    return recs


class _FlakyBackend:
    """Backend that raises `fail_count` transient errors then returns a fixed payload."""

    id = "flaky"

    def __init__(
        self,
        fail_count: int,
        exc_cls: type[BaseException] = RuntimeError,
        exc_msg: str = "429 rate limit exceeded",
    ) -> None:
        self._fail_count = fail_count
        self._exc_cls = exc_cls
        self._exc_msg = exc_msg
        self.call_count = 0

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise self._exc_cls(self._exc_msg)
        return {
            "model": "m",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "suggestions": [
                                {
                                    "anchor": "rec-0",
                                    "proposal": "do a thing",
                                    "snippet": None,
                                    "confidence": 0.9,
                                    "rationale": "fine",
                                }
                            ]
                        }
                    ),
                }
            ],
            "stop_reason": "end_turn",
            "latency_ms": 1,
            "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
        }


class _ConcurrentBackend:
    """Backend that records concurrency — every call increments an in-flight
    counter, sleeps briefly, then decrements. Lets us assert parallelism."""

    id = "concurrent"

    def __init__(self, sleep_s: float = 0.05) -> None:
        self._sleep = sleep_s
        self.in_flight = 0
        self.max_in_flight = 0
        self.total = 0
        self._lock = asyncio.Lock()

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self.in_flight += 1
            if self.in_flight > self.max_in_flight:
                self.max_in_flight = self.in_flight
            self.total += 1
        try:
            await asyncio.sleep(self._sleep)
            return {
                "model": "m",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "suggestions": [
                                    {
                                        "anchor": "rec-0",
                                        "proposal": f"fix {self.total}",
                                        "snippet": None,
                                        "confidence": 0.8,
                                        "rationale": "",
                                    }
                                ]
                            }
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "latency_ms": int(self._sleep * 1000),
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }
        finally:
            async with self._lock:
                self.in_flight -= 1


# ---- retry classifier -----------------------------------------------------


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("429 rate limit exceeded", True),
        ("RateLimit: 429", True),
        ("503 Service Unavailable", True),
        ("upstream is overloaded", True),
        ("connection reset by peer", True),
        ("Request timeout after 60s", True),
        ("502 Bad Gateway", True),
        ("401 Unauthorized — invalid api key", False),
        ("400 Bad Request: messages[0].role must be ...", False),
        ("403 Forbidden", False),
        ("some mysterious error", False),  # default: don't retry
    ],
)
def test_is_transient_classifier(msg: str, expected: bool) -> None:
    assert _is_transient(Exception(msg)) is expected


# ---- retry loop -----------------------------------------------------------


def test_retry_succeeds_after_transient_failures() -> None:
    backend = _FlakyBackend(fail_count=2)
    # Override backoff to ~0 so the test is fast.
    out = asyncio.run(
        _complete_with_retry(
            backend,
            {"model": "m", "messages": []},
            max_retries=4,
            initial_backoff=0.001,
            backoff_mult=1.1,
            jitter=0.0,
        )
    )
    assert out["content"][0]["type"] == "text"
    assert backend.call_count == 3  # 2 failures + 1 success


def test_retry_gives_up_after_max_retries() -> None:
    backend = _FlakyBackend(fail_count=10)
    with pytest.raises(RuntimeError, match="rate limit"):
        asyncio.run(
            _complete_with_retry(
                backend,
                {"model": "m", "messages": []},
                max_retries=3,
                initial_backoff=0.001,
                backoff_mult=1.0,
                jitter=0.0,
            )
        )
    assert backend.call_count == 4  # initial + 3 retries


def test_non_transient_error_raises_immediately() -> None:
    backend = _FlakyBackend(fail_count=10, exc_msg="401 Unauthorized")
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(
            _complete_with_retry(
                backend,
                {"model": "m", "messages": []},
                max_retries=5,
                initial_backoff=0.001,
            )
        )
    assert backend.call_count == 1  # no retries


def test_retry_on_retry_callback_invoked() -> None:
    backend = _FlakyBackend(fail_count=2)
    events: list[tuple[int, float, str]] = []
    asyncio.run(
        _complete_with_retry(
            backend,
            {"model": "m", "messages": []},
            max_retries=4,
            initial_backoff=0.001,
            backoff_mult=1.1,
            jitter=0.0,
            on_retry=lambda n, d, e: events.append((n, d, str(e))),
        )
    )
    assert len(events) == 2
    assert events[0][0] == 1 and "rate limit" in events[0][2]


def test_on_retry_exception_never_crashes_caller() -> None:
    """Observability callback is isolated — if it explodes, retry still proceeds."""
    backend = _FlakyBackend(fail_count=1)

    def broken_callback(n: int, d: float, e: BaseException) -> None:
        raise ValueError("telemetry is broken")

    # Should NOT propagate; the retry should still succeed.
    out = asyncio.run(
        _complete_with_retry(
            backend,
            {"model": "m"},
            max_retries=2,
            initial_backoff=0.001,
            jitter=0.0,
            on_retry=broken_callback,
        )
    )
    assert out is not None
    assert backend.call_count == 2


def test_suggest_fixes_threads_retries_through() -> None:
    backend = _FlakyBackend(fail_count=2)
    result = suggest_fixes(
        _report_with_recs(),
        _synthetic_records(3),
        _synthetic_records(3),
        backend,
        max_retries=4,
    )
    assert len(result.suggestions) == 1
    assert backend.call_count == 3


# ---- scale — 10k pair trace -----------------------------------------------


def test_evidence_truncation_bounds_prompt_at_10k_pairs() -> None:
    """10k-pair trace with huge text — evidence window must stay bounded."""
    big_records = _synthetic_records(10_000, text_size=5000)

    class _ProbeBackend:
        id = "probe"
        last_prompt_chars = 0

        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            # Grab the user message size — the only thing that should grow.
            user_msg = next(m for m in payload["messages"] if m["role"] == "user")
            type(self).last_prompt_chars = len(user_msg["content"])
            return {
                "model": "m",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "suggestions": [
                                    {
                                        "anchor": "rec-0",
                                        "proposal": "x",
                                        "snippet": None,
                                        "confidence": 0.5,
                                        "rationale": "",
                                    }
                                ]
                            }
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
            }

    backend = _ProbeBackend()
    start = time.perf_counter()
    result = suggest_fixes(_report_with_recs(n_recs=3), big_records, big_records, backend)
    elapsed = time.perf_counter() - start

    # Per-record truncation cap is 1800 chars. Anchor limit is 6 → up to 6 turns
    # x 4 records/turn (b-req, b-resp, c-req, c-resp) x 1800 chars each, plus
    # some JSON overhead and the schema. Total should stay well under 100k chars.
    assert (
        _ProbeBackend.last_prompt_chars < 100_000
    ), f"prompt grew to {_ProbeBackend.last_prompt_chars} chars — truncation broken"
    # And the overall call should finish quickly (no O(N²) traversal).
    assert elapsed < 3.0, f"suggest_fixes on 10k pairs took {elapsed:.2f}s"
    assert result.suggestions  # at least one suggestion made it through


def test_policy_diff_at_10k_pairs_under_1s() -> None:
    """policy_diff should scale near-linearly — 10k pairs in under a second."""
    from shadow.hierarchical import load_policy, policy_diff

    baseline = _synthetic_records(10_000, text_size=20)
    candidate = _synthetic_records(10_000, text_size=20)
    rules = load_policy(
        [
            {"id": "max-turns", "kind": "max_turns", "params": {"limit": 50_000}},
            {"id": "no-rm", "kind": "no_call", "params": {"tool": "rm_rf"}},
            {"id": "no-ssn", "kind": "forbidden_text", "params": {"text": "SSN:"}},
        ]
    )
    start = time.perf_counter()
    out = policy_diff(baseline, candidate, rules)
    elapsed = time.perf_counter() - start
    assert out.regressions == []
    assert elapsed < 1.0, f"policy_diff on 10k pairs took {elapsed:.2f}s"


# ---- concurrency ----------------------------------------------------------


def _run_suggest_fixes_task(backend: Any, records: list[dict]) -> int:
    """Thread target — synchronous wrapper around suggest_fixes."""
    out = suggest_fixes(_report_with_recs(), records, records, backend)
    return len(out.suggestions)


def test_100_concurrent_suggest_fixes_share_backend() -> None:
    """100 concurrent suggest_fixes across threads all succeed without interleaving bugs."""
    import concurrent.futures

    class _ThreadSafeBackend:
        """Each thread calls backend.complete via its own asyncio.run — this
        exercises the real contention pattern a server would see.
        """

        id = "thread-safe"

        def __init__(self) -> None:
            import threading

            self._lock = threading.Lock()
            self.total_calls = 0

        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            with self._lock:
                self.total_calls += 1
                n = self.total_calls
            # Simulate real API latency.
            await asyncio.sleep(0.01)
            return {
                "model": "m",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "suggestions": [
                                    {
                                        "anchor": "rec-0",
                                        "proposal": f"fix-{n}",
                                        "snippet": None,
                                        "confidence": 0.8,
                                        "rationale": "",
                                    }
                                ]
                            }
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }

    backend = _ThreadSafeBackend()
    records = _synthetic_records(3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(_run_suggest_fixes_task, backend, records) for _ in range(100)]
        results = [f.result(timeout=30) for f in futures]

    assert all(r == 1 for r in results), f"some workers returned 0 suggestions: {results}"
    assert backend.total_calls == 100
