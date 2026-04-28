"""Tests for examples/harmful-content-judge — domain-aware harm classifier.

External eval flagged that the built-in safety axis is narrow (only
catches refusal/content_filter), so harmful semantic content
delivered without refusal slips through. This example demonstrates
the documented mitigation: a Judge axis with a domain rubric.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "harmful-content-judge"

if not _EXAMPLE_DIR.exists():
    pytestmark = pytest.mark.skip(reason="harmful-content-judge example missing")
else:
    sys.path.insert(0, str(_EXAMPLE_DIR))


def test_demo_exits_one_when_judge_flags_unsafe() -> None:
    """The fixtures + canned judge backend together must produce
    verdict='unsafe' for the candidate, causing the demo CLI to
    exit 1 (intentional — signals a regression-blocking event)."""
    proc = subprocess.run(
        [sys.executable, str(_EXAMPLE_DIR / "run_demo.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert proc.returncode == 1, (
        f"Demo should exit 1 on unsafe verdict; got {proc.returncode}\n"
        f"stdout: {proc.stdout[-500:]}\n"
        f"stderr: {proc.stderr[-200:]}"
    )
    assert "UNSAFE" in proc.stdout
    assert "CORRECTLY caught" in proc.stdout


def test_judge_score_map_maps_unsafe_to_zero() -> None:
    from harmful_content_judge import build_harm_judge

    class _Backend:
        @property
        def id(self) -> str:
            return "stub"

        async def complete(self, request):  # type: ignore[no-untyped-def]
            import json

            v = {"verdict": "unsafe", "confidence": 0.9, "reason": "x"}
            return {
                "model": "stub",
                "content": [{"type": "text", "text": json.dumps(v)}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            }

    judge = build_harm_judge(_Backend())
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "params": {"temperature": 0.0, "max_tokens": 64},
        "tools": [],
    }
    response = {
        "model": "test",
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "end_turn",
        "latency_ms": 1,
        "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
    }
    verdict = asyncio.run(judge.score_pair(response, response, request))
    assert verdict["verdict"] == "unsafe"
    assert verdict["score"] == 0.0


def test_judge_score_map_maps_safe_to_one() -> None:
    from harmful_content_judge import build_harm_judge

    class _Backend:
        @property
        def id(self) -> str:
            return "stub"

        async def complete(self, request):  # type: ignore[no-untyped-def]
            import json

            v = {"verdict": "safe", "confidence": 0.9, "reason": "x"}
            return {
                "model": "stub",
                "content": [{"type": "text", "text": json.dumps(v)}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            }

    judge = build_harm_judge(_Backend())
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "params": {"temperature": 0.0, "max_tokens": 64},
        "tools": [],
    }
    response = {
        "model": "test",
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "end_turn",
        "latency_ms": 1,
        "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
    }
    verdict = asyncio.run(judge.score_pair(response, response, request))
    assert verdict["verdict"] == "safe"
    assert verdict["score"] == 1.0
