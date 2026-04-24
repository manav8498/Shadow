"""Tests for `shadow import --format mcp`.

MCP (Model Context Protocol) is Anthropic's JSON-RPC-2.0 standard
for agent ↔ tool communication (spec 2025-06-18, adopted across
Claude Desktop, Cursor, Windsurf, Zed, VS Code by early 2026).

The importer turns MCP session logs into Shadow `.agentlog`
records. These tests lock the round-trip shape end-to-end so Shadow's
existing Rust differ (trajectory / conformance / first-divergence)
works on imported MCP traces without any further wiring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.errors import ShadowConfigError
from shadow.importers.mcp import mcp_to_agentlog

# ---- canonical MCP fixtures (minimal, real-shape) ----------------------


def _initialize_exchange() -> list[dict]:
    """Every MCP session opens with initialize / initialized."""
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "acme-server", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]


def _tools_list_exchange() -> list[dict]:
    return [
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "search_orders",
                        "description": "Search orders by customer id.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"customer_id": {"type": "string"}},
                            "required": ["customer_id"],
                        },
                    },
                    {
                        "name": "refund_order",
                        "description": "Issue a refund.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "order_id": {"type": "string"},
                                "amount_usd": {"type": "number"},
                            },
                            "required": ["order_id", "amount_usd"],
                        },
                    },
                ]
            },
        },
    ]


def _tools_call_exchange(
    call_id: int, name: str, args: dict, result_text: str = "ok"
) -> list[dict]:
    return [
        {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        },
        {
            "jsonrpc": "2.0",
            "id": call_id,
            "result": {"content": [{"type": "text", "text": result_text}]},
        },
    ]


def _canonical_session() -> list[dict]:
    """A realistic ~5-message MCP session with tool listing + 2 calls."""
    return (
        _initialize_exchange()
        + _tools_list_exchange()
        + _tools_call_exchange(3, "search_orders", {"customer_id": "C42"}, "order ORD-123 found")
        + _tools_call_exchange(
            4, "refund_order", {"order_id": "ORD-123", "amount_usd": 30.0}, "refund issued"
        )
    )


# ---- shape round-trip ---------------------------------------------------


def test_list_input_produces_metadata_plus_request_response_pairs() -> None:
    records = mcp_to_agentlog(_canonical_session())
    kinds = [r["kind"] for r in records]
    # metadata + (chat_request + chat_response) * 2 = 5 records
    assert kinds == ["metadata", "chat_request", "chat_response", "chat_request", "chat_response"]


def test_metadata_captures_tools_list() -> None:
    records = mcp_to_agentlog(_canonical_session())
    meta = records[0]
    tools = meta["payload"]["tools"]
    assert len(tools) == 2
    assert {t["name"] for t in tools} == {"search_orders", "refund_order"}
    # inputSchema is renamed input_schema to match Shadow's convention
    # (Anthropic + OpenAI both use input_schema / parameters).
    assert "input_schema" in tools[0]
    assert tools[0]["input_schema"]["required"] == ["customer_id"]


def test_tool_call_content_block_is_anthropic_shape() -> None:
    records = mcp_to_agentlog(_canonical_session())
    responses = [r for r in records if r["kind"] == "chat_response"]
    assert len(responses) == 2
    first = responses[0]["payload"]["content"]
    # First content block must be a tool_use with name + input matching
    # the JSON-RPC call.
    assert first[0]["type"] == "tool_use"
    assert first[0]["name"] == "search_orders"
    assert first[0]["input"] == {"customer_id": "C42"}
    # Second block (when result present) must be a tool_result.
    assert first[1]["type"] == "tool_result"
    assert first[1]["tool_use_id"] == first[0]["id"]
    assert first[1]["content"] == [{"type": "text", "text": "order ORD-123 found"}]


def test_parent_chain_is_connected() -> None:
    """Each record's parent must point at its predecessor so the differ
    can walk the trace in order."""
    from itertools import pairwise

    records = mcp_to_agentlog(_canonical_session())
    assert records[0]["parent"] is None
    for prev, curr in pairwise(records):
        assert (
            curr["parent"] == prev["id"]
        ), f"{curr['kind']} parent {curr['parent']!r} != prev {prev['kind']} id {prev['id']!r}"


# ---- error handling -----------------------------------------------------


def test_empty_input_raises_config_error() -> None:
    with pytest.raises(ShadowConfigError, match="no JSON-RPC messages"):
        mcp_to_agentlog([])


def test_mcp_error_response_maps_to_is_error_tool_result() -> None:
    session = _tools_call_exchange(9, "refund_order", {"order_id": "X", "amount_usd": 10})
    # Replace the second message (response) with an error.
    session[1] = {
        "jsonrpc": "2.0",
        "id": 9,
        "error": {"code": -32602, "message": "Invalid params: order X not found"},
    }
    records = mcp_to_agentlog(session)
    # metadata + chat_request + chat_response
    assert len(records) == 3
    content = records[-1]["payload"]["content"]
    tool_result = next(b for b in content if b.get("type") == "tool_result")
    assert tool_result["is_error"] is True
    assert "Invalid params" in tool_result["content"][0]["text"]


def test_orphan_request_without_response_still_emits_a_record() -> None:
    """A tool call that never got a response (client disconnect, crash)
    should still produce a `chat_response` with just the tool_use block."""
    session = [
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "foo", "arguments": {}},
        }
    ]
    records = mcp_to_agentlog(session)
    assert len(records) == 3  # metadata + request + response
    content = records[-1]["payload"]["content"]
    assert content[0]["type"] == "tool_use"
    # No tool_result since there was no response.
    assert all(b.get("type") != "tool_result" for b in content)


# ---- input format tolerance ---------------------------------------------


def test_wrapped_object_format_extracts_messages_and_metadata() -> None:
    wrapped = {
        "messages": _canonical_session(),
        "session_id": "s-42",
        "server_name": "acme-server",
    }
    records = mcp_to_agentlog(wrapped)
    meta = records[0]
    mcp_meta = meta["payload"]["mcp_session_metadata"]
    assert mcp_meta["session_id"] == "s-42"
    assert mcp_meta["server_name"] == "acme-server"


def test_single_message_dict_accepted() -> None:
    """A bare single-message dict (one-shot probe) shouldn't crash."""
    msg = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    records = mcp_to_agentlog(msg)
    # metadata only — no tools/call in the input.
    assert len(records) == 1
    assert records[0]["kind"] == "metadata"
    assert "warnings" in records[0]["payload"]


def test_non_list_non_dict_input_raises() -> None:
    with pytest.raises(ShadowConfigError, match="must be a JSON-RPC message list"):
        mcp_to_agentlog(42)  # type: ignore[arg-type]


# ---- CLI integration ----------------------------------------------------


def _run_cli_import(src: Path, out: Path, fmt: str = "mcp") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "import",
            str(src),
            "--format",
            fmt,
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_import_mcp_jsonl_produces_agentlog(tmp_path: Path) -> None:
    """`shadow import --format mcp` on a real JSONL log round-trips."""
    src = tmp_path / "session.jsonl"
    src.write_text("\n".join(json.dumps(m) for m in _canonical_session()) + "\n")
    out = tmp_path / "imported.agentlog"
    r = _run_cli_import(src, out)
    assert r.returncode == 0, r.stderr
    assert out.is_file()

    # Parse the agentlog via the Rust core to prove it's a valid log.
    from shadow import _core

    parsed = _core.parse_agentlog(out.read_bytes())
    assert parsed[0]["kind"] == "metadata"
    # 2 tools/call exchanges → 2 chat_response records.
    responses = [r for r in parsed if r["kind"] == "chat_response"]
    assert len(responses) == 2


def test_cli_import_mcp_json_array_also_works(tmp_path: Path) -> None:
    """MCP Inspector exports are JSON arrays — must be accepted too."""
    src = tmp_path / "session.json"
    src.write_text(json.dumps(_canonical_session(), indent=2))
    out = tmp_path / "imported.agentlog"
    r = _run_cli_import(src, out)
    assert r.returncode == 0, r.stderr


def test_imported_mcp_trace_can_be_diffed(tmp_path: Path) -> None:
    """Two MCP sessions where the second renames a tool should show up
    on Shadow's trajectory axis — this is the end-to-end promise."""
    baseline_session = (
        _initialize_exchange()
        + _tools_list_exchange()
        + _tools_call_exchange(3, "search_orders", {"customer_id": "C42"}, "found")
    )
    candidate_session = (
        _initialize_exchange()
        + _tools_list_exchange()
        + _tools_call_exchange(
            3,
            "search_orders",
            {"cid": "C42"},  # arg renamed customer_id -> cid
            "found",
        )
    )

    base_log = tmp_path / "base.jsonl"
    cand_log = tmp_path / "cand.jsonl"
    base_log.write_text("\n".join(json.dumps(m) for m in baseline_session))
    cand_log.write_text("\n".join(json.dumps(m) for m in candidate_session))

    base_out = tmp_path / "base.agentlog"
    cand_out = tmp_path / "cand.agentlog"
    assert _run_cli_import(base_log, base_out).returncode == 0
    assert _run_cli_import(cand_log, cand_out).returncode == 0

    diff = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "diff",
            str(base_out),
            str(cand_out),
            "--output-json",
            str(tmp_path / "diff.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert diff.returncode == 0
    report = json.loads((tmp_path / "diff.json").read_text())
    trajectory = next(r for r in report["rows"] if r["axis"] == "trajectory")
    # Arg rename customer_id -> cid changes tool-shape → trajectory must move.
    assert (
        abs(trajectory["delta"]) > 0.0
    ), f"expected trajectory movement on arg rename; got {trajectory}"
