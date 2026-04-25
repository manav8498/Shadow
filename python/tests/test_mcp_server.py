"""Tests for the shadow MCP server mode.

We exercise the handler functions directly rather than spinning up a
stdio server and a mock client, which would add a process boundary for
no extra coverage. The tool-registration shape is checked too, to
catch accidental schema drift.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

# The MCP server module pulls in the `mcp` SDK at import time for the
# protocol types. When the user hasn't installed `shadow-diff[mcp]`
# we skip the whole file — matches how the adapter test files handle
# their optional framework deps.
pytest.importorskip("mcp")

from shadow import _core  # noqa: E402
from shadow.mcp_server import (  # noqa: E402
    TOOL_HANDLERS,
    _build_tools,
    _call_tool_impl,
    _handle_certify,
    _handle_check_policy,
    _handle_diff,
    _handle_summarise,
    _handle_token_diff,
    _handle_verify_cert,
)


@pytest.fixture
def tiny_traces(tmp_path: Path) -> tuple[Path, Path]:
    """Two minimal .agentlog files that differ on one axis."""

    def _mk(output: Path, text: str) -> None:
        meta = {"sdk": {"name": "shadow", "version": "test"}}
        meta_id = _core.content_id(meta)
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "params": {}}
        req_id = _core.content_id(req)
        resp = {
            "model": "m",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": 10,
            "usage": {"input_tokens": 1, "output_tokens": len(text), "thinking_tokens": 0},
        }
        resp_id = _core.content_id(resp)
        recs = [
            {
                "version": "0.1",
                "id": meta_id,
                "kind": "metadata",
                "ts": "2026-04-24T00:00:00Z",
                "parent": None,
                "payload": meta,
            },
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": "2026-04-24T00:00:01Z",
                "parent": meta_id,
                "payload": req,
            },
            {
                "version": "0.1",
                "id": resp_id,
                "kind": "chat_response",
                "ts": "2026-04-24T00:00:02Z",
                "parent": req_id,
                "payload": resp,
            },
        ]
        output.write_bytes(_core.write_agentlog(recs))

    b = tmp_path / "b.agentlog"
    c = tmp_path / "c.agentlog"
    _mk(b, "hello")
    _mk(c, "goodbye, this is a longer response that should trigger verbosity")
    return b, c


# ---- tool list ------------------------------------------------------------


def test_build_tools_returns_all_seven() -> None:
    tools = _build_tools()
    names = [t.name for t in tools]
    assert names == [
        "shadow_diff",
        "shadow_check_policy",
        "shadow_token_diff",
        "shadow_schema_watch",
        "shadow_summarise",
        "shadow_certify",
        "shadow_verify_cert",
    ]
    # Every tool must have a non-empty description and an inputSchema
    for t in tools:
        assert t.description and len(t.description) > 20
        assert t.inputSchema and "properties" in t.inputSchema
        assert "required" in t.inputSchema


def test_tool_handlers_mapping_matches_list() -> None:
    tool_names = {t.name for t in _build_tools()}
    assert set(TOOL_HANDLERS) == tool_names


# ---- shadow_diff ----------------------------------------------------------


def test_diff_handler_returns_nine_rows(tiny_traces: tuple[Path, Path]) -> None:
    b, c = tiny_traces
    result = asyncio.run(_handle_diff({"baseline": str(b), "candidate": str(c)}))
    assert len(result["rows"]) == 9
    axes = {row["axis"] for row in result["rows"]}
    assert axes >= {"semantic", "trajectory", "verbosity", "latency"}


def test_diff_handler_includes_policy_when_path_given(
    tiny_traces: tuple[Path, Path], tmp_path: Path
) -> None:
    b, c = tiny_traces
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "rules:\n"
        "  - id: short-response\n"
        "    kind: required_stop_reason\n"
        "    params: {allowed: [end_turn]}\n"
        "    severity: error\n"
    )
    result = asyncio.run(
        _handle_diff({"baseline": str(b), "candidate": str(c), "policy_path": str(policy)})
    )
    assert "policy_diff" in result


def test_diff_handler_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(_handle_diff({"baseline": "/nope", "candidate": "/still-nope"}))


# ---- shadow_check_policy --------------------------------------------------


def test_check_policy_handler_reports_structure(
    tiny_traces: tuple[Path, Path], tmp_path: Path
) -> None:
    b, c = tiny_traces
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps([{"id": "r", "kind": "no_call", "params": {"tool": "x"}}]))
    result = asyncio.run(
        _handle_check_policy({"baseline": str(b), "candidate": str(c), "policy_path": str(policy)})
    )
    for key in ("baseline_violations", "candidate_violations", "regressions", "fixes"):
        assert key in result


# ---- shadow_token_diff ----------------------------------------------------


def test_token_diff_handler(tiny_traces: tuple[Path, Path]) -> None:
    b, c = tiny_traces
    result = asyncio.run(_handle_token_diff({"baseline": str(b), "candidate": str(c)}))
    assert "dimensions" in result
    assert "normalised_shift" in result


def test_token_diff_respects_top_k(tiny_traces: tuple[Path, Path]) -> None:
    b, c = tiny_traces
    result = asyncio.run(
        _handle_token_diff({"baseline": str(b), "candidate": str(c), "top_k_pairs": 3})
    )
    assert len(result["worst_pairs"]) <= 3


# ---- shadow_summarise -----------------------------------------------------


def test_summarise_handler_returns_text(tiny_traces: tuple[Path, Path], tmp_path: Path) -> None:
    b, c = tiny_traces
    diff = asyncio.run(_handle_diff({"baseline": str(b), "candidate": str(c)}))
    report_path = tmp_path / "r.json"
    report_path.write_text(json.dumps(diff))
    result = asyncio.run(_handle_summarise({"report_json": str(report_path)}))
    assert "summary" in result
    assert isinstance(result["summary"], str)


# ---- top-level call_tool dispatcher ---------------------------------------


def test_call_tool_impl_unknown_name_returns_error_text() -> None:
    out = asyncio.run(_call_tool_impl("shadow_nonsense", {}))
    assert len(out) == 1
    payload = json.loads(out[0].text)
    assert "error" in payload
    assert "available" in payload


def test_call_tool_impl_catches_handler_exception(tiny_traces: tuple[Path, Path]) -> None:
    out = asyncio.run(_call_tool_impl("shadow_diff", {}))  # missing required args
    assert len(out) == 1
    payload = json.loads(out[0].text)
    assert "error" in payload


def test_call_tool_impl_round_trip(tiny_traces: tuple[Path, Path]) -> None:
    b, c = tiny_traces
    out = asyncio.run(_call_tool_impl("shadow_diff", {"baseline": str(b), "candidate": str(c)}))
    assert len(out) == 1
    payload = json.loads(out[0].text)
    assert len(payload["rows"]) == 9


# ---- shadow_certify / shadow_verify_cert ---------------------------------


def test_certify_handler_writes_and_returns_cert(
    tiny_traces: tuple[Path, Path], tmp_path: Path
) -> None:
    _, c = tiny_traces
    out_path = tmp_path / "release.cert.json"
    result = asyncio.run(
        _handle_certify(
            {
                "trace": str(c),
                "agent_id": "test-agent@1.0",
                "output": str(out_path),
            }
        )
    )
    assert result["cert_id"].startswith("sha256:")
    assert result["output"] == str(out_path)
    assert "cert" in result
    # The file on disk must round-trip back to the same cert_id.
    on_disk = json.loads(out_path.read_text())
    assert on_disk["cert_id"] == result["cert_id"]


def test_certify_handler_with_baseline_includes_regression_suite(
    tiny_traces: tuple[Path, Path], tmp_path: Path
) -> None:
    b, c = tiny_traces
    out_path = tmp_path / "release.cert.json"
    result = asyncio.run(
        _handle_certify(
            {
                "trace": str(c),
                "agent_id": "x",
                "output": str(out_path),
                "baseline": str(b),
            }
        )
    )
    rs = result["cert"]["regression_suite"]
    assert rs is not None
    assert len(rs["axes"]) == 9


def test_verify_cert_handler_passes_for_valid_cert(
    tiny_traces: tuple[Path, Path], tmp_path: Path
) -> None:
    _, c = tiny_traces
    out_path = tmp_path / "release.cert.json"
    asyncio.run(_handle_certify({"trace": str(c), "agent_id": "x", "output": str(out_path)}))
    result = asyncio.run(_handle_verify_cert({"cert": str(out_path)}))
    assert result["ok"] is True
    assert result["cert_id"].startswith("sha256:")


def test_verify_cert_handler_fails_for_tampered_cert(
    tiny_traces: tuple[Path, Path], tmp_path: Path
) -> None:
    _, c = tiny_traces
    out_path = tmp_path / "release.cert.json"
    asyncio.run(_handle_certify({"trace": str(c), "agent_id": "x", "output": str(out_path)}))
    payload = json.loads(out_path.read_text())
    payload["agent_id"] = "tampered"
    out_path.write_text(json.dumps(payload))
    result = asyncio.run(_handle_verify_cert({"cert": str(out_path)}))
    assert result["ok"] is False
    assert "mismatch" in result["detail"]
