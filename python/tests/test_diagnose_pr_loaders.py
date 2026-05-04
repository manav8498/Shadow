"""Tests for `shadow.diagnose_pr.loaders`.

Two surfaces:
  * load_config(path) — YAML config (same schema as `shadow replay`)
  * load_traces(paths) — one or more .agentlog files / dirs

Both raise typed errors (ShadowConfigError / ShadowParseError) so
the CLI can surface a clean message instead of a stack trace."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_load_config_round_trips_a_real_demo_yaml(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_config

    p = tmp_path / "baseline.yaml"
    p.write_text(
        "model: claude-opus-4-7\n"
        "params:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 512\n"
        "prompt:\n"
        "  system: 'You are a refund agent.'\n"
    )
    cfg = load_config(p)
    assert cfg["model"] == "claude-opus-4-7"
    assert cfg["params"]["temperature"] == pytest.approx(0.2)
    assert "refund agent" in cfg["prompt"]["system"]


def test_load_config_missing_file_raises_shadow_config_error(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_config
    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_invalid_yaml_raises_shadow_config_error(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_config
    from shadow.errors import ShadowConfigError

    p = tmp_path / "bad.yaml"
    p.write_text("model: : :")
    with pytest.raises(ShadowConfigError, match="could not parse"):
        load_config(p)


def test_load_traces_from_file_returns_one_loaded_trace(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces
    from shadow.sdk import Session

    p = tmp_path / "t.agentlog"
    with Session(output_path=p, tags={"env": "test"}) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "hi"}],
                "params": {},
            },
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )

    loaded = load_traces([p])
    assert len(loaded) == 1
    t = loaded[0]
    assert t.path == p
    # trace_id is the envelope `meta.trace_id` (a UUID hex) when
    # present; falls back to the metadata record's content id
    # ("sha256:...") for older traces written before v3.0.5.
    assert isinstance(t.trace_id, str) and (
        t.trace_id.startswith("sha256:") or len(t.trace_id) == 32
    )
    assert len(t.records) >= 2  # metadata + at least one chat pair


def test_load_traces_from_directory_globs_agentlog_files(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces
    from shadow.sdk import Session

    for name in ("a.agentlog", "b.agentlog"):
        with Session(output_path=tmp_path / name, tags={}) as s:
            s.record_chat(
                request={
                    "model": "x",
                    "messages": [{"role": "user", "content": "h"}],
                    "params": {},
                },
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": "h"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 1,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )

    loaded = load_traces([tmp_path])
    assert len(loaded) == 2
    paths = sorted(t.path.name for t in loaded)
    assert paths == ["a.agentlog", "b.agentlog"]


def test_load_traces_skips_non_agentlog_files(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces

    (tmp_path / "readme.txt").write_text("hello")
    (tmp_path / "data.json").write_text("{}")
    loaded = load_traces([tmp_path])
    assert loaded == []


def test_load_traces_corrupt_file_raises_shadow_parse_error(tmp_path: Path) -> None:
    from shadow.diagnose_pr.loaders import load_traces
    from shadow.errors import ShadowParseError

    p = tmp_path / "broken.agentlog"
    p.write_bytes(b"not jsonl at all\n")
    with pytest.raises(ShadowParseError):
        load_traces([p])
