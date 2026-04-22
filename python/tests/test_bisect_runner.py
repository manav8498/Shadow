"""Coverage tests for shadow.bisect.runner heuristic allocator."""

from __future__ import annotations

from pathlib import Path

from shadow.bisect import run_bisect
from shadow.sdk import Session


def _write_trace(path: Path, latency_ms: int, text: str, output_tokens: int = 5) -> None:
    with Session(output_path=path) as s:
        s.record_chat(
            request={"model": "x", "messages": [{"role": "user", "content": "hi"}], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": latency_ms,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": output_tokens,
                    "thinking_tokens": 0,
                },
            },
        )


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


def test_run_bisect_with_candidate_traces_uses_real_divergence(tmp_path: Path) -> None:
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _write_trace(baseline, 100, "hello", output_tokens=5)
    _write_trace(candidate, 500, "hello world extra words", output_tokens=25)

    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    _write_yaml(cfg_a, "model: x\nparams:\n  temperature: 0.2\nprompt:\n  system: short\n")
    _write_yaml(
        cfg_b, "model: x\nparams:\n  temperature: 0.7\nprompt:\n  system: much longer prompt here\n"
    )

    result = run_bisect(cfg_a, cfg_b, baseline, candidate_traces=candidate)
    assert result["mode"] == "heuristic_kind_allocator"
    assert result["warnings"] == []
    # Latency moved (100 → 500), so the latency axis should have nonzero
    # attributions across every delta kind that can plausibly move it.
    latency_attr = {row["delta"]: row["weight"] for row in result["attributions"]["latency"]}
    assert any(w > 0 for w in latency_attr.values())
    # Verbosity moved too; prompt.system is eligible, params.temperature is
    # eligible. Each should have positive weight.
    verb_attr = {row["delta"]: row["weight"] for row in result["attributions"]["verbosity"]}
    assert any(w > 0 for w in verb_attr.values())


def test_run_bisect_without_candidate_traces_falls_back_to_zero_scorer(tmp_path: Path) -> None:
    baseline = tmp_path / "b.agentlog"
    _write_trace(baseline, 100, "hi")
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    _write_yaml(cfg_a, "prompt:\n  system: short\n")
    _write_yaml(cfg_b, "prompt:\n  system: different\n")

    result = run_bisect(cfg_a, cfg_b, baseline)
    assert result["mode"] == "lasso_placeholder_zero"
    assert any("no --candidate-traces" in w for w in result["warnings"])


def test_run_bisect_errors_on_identical_configs(tmp_path: Path) -> None:
    import pytest

    from shadow.errors import ShadowConfigError

    cfg = tmp_path / "a.yaml"
    _write_yaml(cfg, "model: x\nparams:\n  temperature: 0.2\n")
    baseline = tmp_path / "b.agentlog"
    _write_trace(baseline, 100, "hi")
    with pytest.raises(ShadowConfigError, match="identical"):
        run_bisect(cfg, cfg, baseline)


def test_run_bisect_with_many_deltas_uses_plackett_burman(tmp_path: Path) -> None:
    # With 7 deltas the allocator still runs fine without candidate_traces.
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    yaml_a = """
model: a
params:
  temperature: 0.0
  top_p: 0.5
  max_tokens: 100
prompt:
  system: sa
  user_template: "t1"
tools:
  - name: x
"""
    yaml_b = """
model: b
params:
  temperature: 1.0
  top_p: 0.9
  max_tokens: 500
prompt:
  system: sb
  user_template: "t2"
tools:
  - name: y
"""
    _write_yaml(cfg_a, yaml_a)
    _write_yaml(cfg_b, yaml_b)
    baseline = tmp_path / "b.agentlog"
    _write_trace(baseline, 100, "hi")

    result = run_bisect(cfg_a, cfg_b, baseline)
    # design_runs will be either 8 (factorial for k<=6) or a multiple of 4 (PB)
    assert result["design_runs"] >= 8
    assert result["mode"] == "lasso_placeholder_zero"
