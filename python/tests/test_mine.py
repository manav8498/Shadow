"""Tests for production trace mining."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.mine import mine


def _pair(
    text: str = "ok",
    stop: str = "end_turn",
    latency: int = 50,
    in_tok: int = 10,
    out_tok: int = 10,
    think: int = 0,
    tool: str | None = None,
    model: str = "m",
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if tool:
        content.append({"type": "tool_use", "id": f"c_{tool}", "name": tool, "input": {}})
    else:
        content.append({"type": "text", "text": text})
    req = {
        "version": "0.1",
        "id": _core.content_id({"model": model, "t": text}),
        "kind": "chat_request",
        "ts": "2026-04-24T00:00:00Z",
        "parent": "meta",
        "payload": {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "params": {},
        },
    }
    resp = {
        "version": "0.1",
        "id": _core.content_id({"model": model, "r": text, "s": stop}),
        "kind": "chat_response",
        "ts": "2026-04-24T00:00:01Z",
        "parent": req["id"],
        "payload": {
            "model": model,
            "content": content,
            "stop_reason": stop,
            "latency_ms": latency,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok, "thinking_tokens": think},
        },
    }
    return req, resp


def _trace(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = [
        {
            "version": "0.1",
            "id": "meta",
            "kind": "metadata",
            "ts": "2026-04-24T00:00:00Z",
            "parent": None,
            "payload": {"sdk": {"name": "test", "version": "0"}},
        }
    ]
    for req, resp in pairs:
        recs.append(req)
        recs.append(resp)
    return recs


# ---- clustering + picking -------------------------------------------------


def test_empty_input_is_safe() -> None:
    result = mine([])
    assert result.total_input_pairs == 0
    assert result.clusters_found == 0
    assert result.cases == []


def test_clusters_by_tool_and_stop_reason() -> None:
    # three end_turn clusters of identical pairs + one error stop
    trace = _trace(
        [
            _pair(text="a", stop="end_turn"),
            _pair(text="b", stop="end_turn"),  # same cluster (same tools, length bucket)
            _pair(text="c", stop="end_turn"),
            _pair(text="oops", stop="error"),  # different cluster
        ]
    )
    result = mine([trace], max_cases=10, per_cluster=1)
    clusters = {c.cluster for c in result.cases}
    assert len(clusters) == 2
    assert result.total_input_pairs == 4


def test_error_stop_scores_highest() -> None:
    trace = _trace(
        [
            _pair(stop="end_turn"),
            _pair(stop="end_turn"),
            _pair(stop="error"),
        ]
    )
    result = mine([trace], max_cases=10, per_cluster=5)
    error_cases = [
        c for c in result.cases if c.response_record["payload"]["stop_reason"] == "error"
    ]
    # Error case exists and has score higher than any non-error case.
    assert error_cases
    assert error_cases[0].score >= max(
        c.score for c in result.cases if c.response_record["payload"]["stop_reason"] != "error"
    )


def test_reasons_are_meaningful() -> None:
    trace = _trace(
        [
            _pair(stop="error"),
            _pair(stop="content_filter"),
            _pair(stop="max_tokens"),
            _pair(stop="end_turn", think=1500),
            _pair(stop="end_turn", latency=8000),
            _pair(stop="end_turn", out_tok=3000),
        ]
    )
    result = mine([trace], max_cases=20, per_cluster=5)
    reasons = {c.reason for c in result.cases}
    assert reasons & {
        "error_stop_reason",
        "refusal",
        "output_truncated",
        "heavy_reasoning",
        "high_latency",
        "long_response",
    }


def test_per_cluster_limits() -> None:
    # Five identical pairs land in the same cluster; per_cluster=2 keeps 2.
    trace = _trace([_pair(text="same") for _ in range(5)])
    result = mine([trace], max_cases=20, per_cluster=2)
    assert len(result.cases) == 2


def test_max_cases_cap() -> None:
    # Ten distinct clusters; max_cases=3 keeps the 3 highest-scoring.
    pairs = [_pair(text=f"t{i}", stop="error" if i < 2 else "end_turn") for i in range(10)]
    trace = _trace(pairs)
    result = mine([trace], max_cases=3, per_cluster=5)
    assert len(result.cases) == 3


def test_pricing_aware_scoring() -> None:
    cheap_trace = _trace([_pair(text="cheap", model="haiku", in_tok=10, out_tok=10)])
    expensive_trace = _trace([_pair(text="expensive", model="opus", in_tok=10000, out_tok=10000)])
    pricing = {
        "haiku": {"input": 1e-6, "output": 3e-6},
        "opus": {"input": 15e-6, "output": 75e-6},
    }
    result = mine(
        [cheap_trace, expensive_trace],
        max_cases=10,
        per_cluster=5,
        pricing=pricing,
    )
    # Expensive pair should score higher
    expensive = [c for c in result.cases if c.response_record["payload"]["model"] == "opus"]
    cheap = [c for c in result.cases if c.response_record["payload"]["model"] == "haiku"]
    assert expensive[0].score > cheap[0].score


def test_to_agentlog_produces_valid_trace() -> None:
    trace = _trace([_pair(stop="error"), _pair(stop="end_turn")])
    result = mine([trace], max_cases=5, per_cluster=1)
    recs = result.to_agentlog()
    # metadata + pairs of (chat_request, chat_response)
    assert recs[0]["kind"] == "metadata"
    # round-trips through the parser
    blob = _core.write_agentlog(recs)
    parsed = _core.parse_agentlog(blob)
    assert len(parsed) == len(recs)
    # metadata has mining stats
    assert "mining" in recs[0]["payload"]


def test_multi_trace_input() -> None:
    t1 = _trace([_pair(text="one", stop="end_turn")])
    t2 = _trace([_pair(text="two", stop="error")])
    result = mine([t1, t2], max_cases=10, per_cluster=5)
    assert result.total_input_pairs == 2
    assert len(result.cases) == 2


# ---- CLI ------------------------------------------------------------------


def test_cli_mine_end_to_end(tmp_path: Path) -> None:
    trace = _trace([_pair(stop="error"), _pair(stop="end_turn"), _pair(stop="end_turn")])
    src = tmp_path / "prod.agentlog"
    src.write_bytes(_core.write_agentlog(trace))
    dst = tmp_path / "mined.agentlog"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "mine",
            str(src),
            "--output",
            str(dst),
            "--max-cases",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    assert dst.exists()
    mined = _core.parse_agentlog(dst.read_bytes())
    # At least metadata + one pair = 3 records minimum
    assert len(mined) >= 3
