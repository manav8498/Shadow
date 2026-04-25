"""Tests for the Agent Behavior Certificate (ABOM) module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadow import _core
from shadow.certify import (
    CERT_VERSION,
    build_certificate,
    render_terminal,
    verify_certificate,
)
from shadow.cli.app import app
from shadow.errors import ShadowConfigError
from shadow.sdk import Session

runner = CliRunner()


def _make_trace(path: Path, *, model: str = "claude-opus-4-7") -> None:
    with Session(output_path=path, tags={"env": "test"}) as s:
        s.record_chat(
            request={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a refund agent."},
                    {"role": "user", "content": "refund please"},
                ],
                "params": {"temperature": 0.0},
                "tools": [
                    {
                        "name": "lookup_order",
                        "description": "look up an order",
                        "input_schema": {
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"],
                        },
                    }
                ],
            },
            response={
                "model": model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 5, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


def test_build_certificate_extracts_models_prompts_and_tools(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())

    cert = build_certificate(trace=records, agent_id="refund-agent@1.0")
    assert cert.cert_version == CERT_VERSION
    assert cert.agent_id == "refund-agent@1.0"
    assert "claude-opus-4-7" in cert.models
    assert len(cert.prompt_hashes) == 1
    assert cert.prompt_hashes[0].startswith("sha256:")
    assert len(cert.tool_schemas) == 1
    assert cert.tool_schemas[0]["name"] == "lookup_order"
    assert cert.tool_schemas[0]["hash"].startswith("sha256:")
    assert cert.cert_id.startswith("sha256:")


def test_certificate_is_self_verifying(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="agent-x")

    ok, detail = verify_certificate(cert.to_dict())
    assert ok, detail


def test_tampering_with_certificate_breaks_verification(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="agent-x")

    payload = cert.to_dict()
    # Tamper with the agent_id; cert_id must no longer match.
    payload["agent_id"] = "totally-different-agent"
    ok, detail = verify_certificate(payload)
    assert not ok
    assert "mismatch" in detail


def test_unsupported_cert_version_is_rejected() -> None:
    payload = {"cert_version": "0.99", "cert_id": "sha256:deadbeef"}
    ok, detail = verify_certificate(payload)
    assert not ok
    assert "unsupported cert_version" in detail


def test_policy_hash_is_recorded(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("rules:\n  - kind: max_turns\n    params: {n: 8}\n")
    cert = build_certificate(trace=records, agent_id="agent-x", policy_path=policy_path)
    assert cert.policy_hash is not None
    assert cert.policy_hash.startswith("sha256:")


def test_baseline_produces_regression_suite(tmp_path: Path) -> None:
    base = tmp_path / "base.agentlog"
    cand = tmp_path / "cand.agentlog"
    _make_trace(base)
    _make_trace(cand, model="gpt-4o-mini")  # different model
    base_records = _core.parse_agentlog(base.read_bytes())
    cand_records = _core.parse_agentlog(cand.read_bytes())

    cert = build_certificate(trace=cand_records, agent_id="agent-x", baseline_trace=base_records)
    assert cert.regression_suite is not None
    assert "axes" in cert.regression_suite
    assert len(cert.regression_suite["axes"]) == 9


def test_empty_trace_raises_config_error() -> None:
    with pytest.raises(ShadowConfigError, match="empty"):
        build_certificate(trace=[], agent_id="x")


def test_trace_without_metadata_raises_config_error() -> None:
    with pytest.raises(ShadowConfigError, match="metadata"):
        build_certificate(
            trace=[{"kind": "chat_request", "id": "sha256:x", "parent": None, "payload": {}}],
            agent_id="x",
        )


def test_render_terminal_shows_critical_fields(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="agent-x")
    out = render_terminal(cert)
    assert "Agent Behavior Certificate" in out
    assert "agent-x" in out
    assert cert.cert_id in out
    assert "lookup_order" in out


# ---- CLI integration ----------------------------------------------------


def test_cli_certify_writes_certificate(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    out_path = tmp_path / "release.cert.json"
    result = runner.invoke(
        app,
        [
            "certify",
            str(trace_path),
            "--agent-id",
            "refund-agent@1.0",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text())
    assert payload["agent_id"] == "refund-agent@1.0"
    assert payload["cert_id"].startswith("sha256:")


def test_cli_verify_cert_passes_for_valid_certificate(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    out_path = tmp_path / "release.cert.json"
    runner.invoke(
        app,
        ["certify", str(trace_path), "--agent-id", "x", "--output", str(out_path)],
    )
    result = runner.invoke(app, ["verify-cert", str(out_path)])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_cli_verify_cert_fails_for_tampered_certificate(tmp_path: Path) -> None:
    trace_path = tmp_path / "release.agentlog"
    _make_trace(trace_path)
    out_path = tmp_path / "release.cert.json"
    runner.invoke(
        app,
        ["certify", str(trace_path), "--agent-id", "x", "--output", str(out_path)],
    )
    payload = json.loads(out_path.read_text())
    payload["agent_id"] = "tampered"
    out_path.write_text(json.dumps(payload))

    result = runner.invoke(app, ["verify-cert", str(out_path)])
    assert result.exit_code == 1
    assert "fail" in result.output


def test_cli_certify_with_baseline_writes_regression_suite(tmp_path: Path) -> None:
    base = tmp_path / "base.agentlog"
    cand = tmp_path / "cand.agentlog"
    _make_trace(base)
    _make_trace(cand)
    out_path = tmp_path / "release.cert.json"
    result = runner.invoke(
        app,
        [
            "certify",
            str(cand),
            "--agent-id",
            "x",
            "--output",
            str(out_path),
            "--baseline",
            str(base),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text())
    assert payload["regression_suite"] is not None
    assert len(payload["regression_suite"]["axes"]) == 9
