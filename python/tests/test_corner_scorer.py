"""End-to-end tests for the LASSO-over-corners scorer.

Uses a synthetic FakeBackend that returns responses whose latency /
verbosity depend deterministically on which config-categories are
active. The LASSO should recover exactly the right attribution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from shadow.bisect import (
    active_categories,
    apply_config_to_request,
    build_intermediate_config,
    run_bisect,
    score_corners,
)
from shadow.errors import ShadowConfigError
from shadow.sdk import Session

# ---------------------------------------------------------------------------
# A deterministic FakeBackend — the config-category truth table.
# ---------------------------------------------------------------------------


class FakeBackend:
    """A backend that returns responses whose metrics depend on which
    config categories are active (i.e. swapped from A to B).

    - `prompt` active → verbosity doubles, latency unchanged.
    - `model` active → latency 3x, verbosity unchanged.
    - `params` active → small latency + verbosity bump.
    - `tools` active → stop_reason flips to 'tool_use'.
    """

    def __init__(
        self,
        config_a: dict[str, Any],
        backend_id: str = "fake",
    ) -> None:
        self._config_a = config_a
        self._id = backend_id
        # Fingerprint which categories are active in each received
        # request, by comparing to config_a.
        self._call_log: list[dict[str, bool]] = []

    @property
    def id(self) -> str:
        return self._id

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        mask = self._fingerprint(request)
        self._call_log.append(mask)
        return self._synthesize(mask)

    def _fingerprint(self, request: dict[str, Any]) -> dict[str, bool]:
        """Derive which categories are active from the request itself."""
        a = self._config_a
        mask: dict[str, bool] = {"model": False, "prompt": False, "params": False, "tools": False}
        if a.get("model") is not None and request.get("model") != a["model"]:
            mask["model"] = True
        if a.get("params") is not None and request.get("params") != a.get("params"):
            mask["params"] = True
        if a.get("tools") is not None and request.get("tools") != a.get("tools"):
            mask["tools"] = True
        system_text = (a.get("prompt") or {}).get("system")
        req_sys = next(
            (m.get("content") for m in request.get("messages", []) if m.get("role") == "system"),
            None,
        )
        if system_text is not None and req_sys != system_text:
            mask["prompt"] = True
        return mask

    @staticmethod
    def _synthesize(mask: dict[str, bool]) -> dict[str, Any]:
        base_latency = 100
        base_verbosity = 10
        latency = base_latency
        verbosity = base_verbosity
        stop = "end_turn"
        if mask["model"]:
            latency *= 3
        if mask["prompt"]:
            verbosity *= 2
        if mask["params"]:
            latency += 20
            verbosity += 2
        if mask["tools"]:
            stop = "tool_use"
        return {
            "model": "x",
            "content": [{"type": "text", "text": "a " * verbosity}],
            "stop_reason": stop,
            "latency_ms": latency,
            "usage": {
                "input_tokens": 10,
                "output_tokens": verbosity,
                "thinking_tokens": 0,
            },
        }


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _write_baseline(path: Path, system_text: str = "you are helpful") -> None:
    with Session(output_path=path) as s:
        s.record_chat(
            request={
                "model": "m-a",
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": "say hi"},
                ],
                "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
                "tools": [
                    {
                        "name": "search",
                        "description": "search",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    }
                ],
            },
            response={
                "model": "m-a",
                "content": [{"type": "text", "text": "a " * 10}],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "thinking_tokens": 0,
                },
            },
        )


def _config_a() -> dict[str, Any]:
    return {
        "model": "m-a",
        "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
        "prompt": {"system": "you are helpful"},
        "tools": [
            {
                "name": "search",
                "description": "search",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        ],
    }


def _config_b() -> dict[str, Any]:
    # Model swap + prompt rewrite. Params/tools unchanged.
    return {
        "model": "m-b",
        "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
        "prompt": {"system": "be thorough and expansive"},
        "tools": _config_a()["tools"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apply_config_swaps_model_and_system_message() -> None:
    request = {
        "model": "m-a",
        "messages": [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "hi"},
        ],
        "params": {"temperature": 0.1},
    }
    new = apply_config_to_request(
        {"model": "m-b", "prompt": {"system": "new"}, "params": {"temperature": 0.9}},
        request,
    )
    assert new["model"] == "m-b"
    assert new["messages"][0]["content"] == "new"
    assert new["messages"][1]["content"] == "hi"  # user msg preserved
    assert new["params"]["temperature"] == 0.9


def test_active_categories_returns_only_differing() -> None:
    a = _config_a()
    b = _config_b()
    assert set(active_categories(a, b)) == {"model", "prompt"}


def test_build_intermediate_config_is_per_category() -> None:
    a = _config_a()
    b = _config_b()
    # Mask: prompt=True, model=False. Expect prompt from B, model from A.
    intermediate = build_intermediate_config(a, b, {"prompt": True, "model": False})
    assert intermediate["model"] == a["model"]
    assert intermediate["prompt"]["system"] == b["prompt"]["system"]


def test_score_corners_recovers_true_drivers(tmp_path: Path) -> None:
    """Ground-truth test: FakeBackend makes latency depend on `model`
    and verbosity on `prompt`. LASSO should attribute each axis
    dominantly to the correct category.
    """
    baseline_path = tmp_path / "baseline.agentlog"
    _write_baseline(baseline_path)

    a = _config_a()
    b = _config_b()
    from shadow import _core

    baseline_records = _core.parse_agentlog(baseline_path.read_bytes())

    backend = FakeBackend(config_a=a)
    result = asyncio.run(score_corners(baseline_records, a, b, backend, seed=42))

    cats = result["categories"]
    assert set(cats) == {"model", "prompt"}

    # Latency should attribute dominantly to `model`.
    latency_attr = {row["category"]: row["weight"] for row in result["attributions"]["latency"]}
    assert latency_attr["model"] > 0.7, latency_attr
    # Verbosity should attribute dominantly to `prompt`.
    verbosity_attr = {row["category"]: row["weight"] for row in result["attributions"]["verbosity"]}
    assert verbosity_attr["prompt"] > 0.7, verbosity_attr


def test_run_bisect_uses_corner_scorer_when_backend_supplied(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.agentlog"
    _write_baseline(baseline_path)
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    # Minimal configs that differ on prompt only — tests k=1.
    cfg_a.write_text("prompt:\n  system: old\n")
    cfg_b.write_text("prompt:\n  system: new\n")

    backend = FakeBackend(config_a={"prompt": {"system": "old"}})

    result = run_bisect(cfg_a, cfg_b, baseline_path, backend=backend)
    assert result["mode"] == "lasso_over_corners"
    assert result["active_categories"] == ["prompt"]
    # With k=1, prompt gets 100% of whatever axes moved.
    verbosity_attr = {row["category"]: row["weight"] for row in result["attributions"]["verbosity"]}
    assert verbosity_attr.get("prompt", 0.0) == pytest.approx(1.0, abs=0.01)


def test_score_corners_errors_on_identical_configs(tmp_path: Path) -> None:
    baseline_path = tmp_path / "b.agentlog"
    _write_baseline(baseline_path)
    from shadow import _core

    baseline_records = _core.parse_agentlog(baseline_path.read_bytes())
    a = _config_a()
    backend = FakeBackend(config_a=a)
    with pytest.raises(ShadowConfigError, match="identical"):
        asyncio.run(score_corners(baseline_records, a, a, backend))
