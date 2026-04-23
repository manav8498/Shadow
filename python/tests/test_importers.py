"""Tests for Langfuse and Braintrust → `.agentlog` importers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from shadow.cli.app import app
from shadow.importers import braintrust_to_agentlog, langfuse_to_agentlog

runner = CliRunner()


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


def _langfuse_export_with(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "traces": [
            {
                "id": "trace-1",
                "name": "agent",
                "observations": observations,
            }
        ]
    }


def test_langfuse_generation_becomes_request_response_pair() -> None:
    data = _langfuse_export_with(
        [
            {
                "id": "obs-1",
                "type": "generation",
                "name": "chat",
                "startTime": "2026-04-21T10:00:00.000Z",
                "endTime": "2026-04-21T10:00:00.150Z",
                "model": "gpt-4.1",
                "modelParameters": {"temperature": 0.2, "max_tokens": 256},
                "input": [{"role": "user", "content": "hi"}],
                "output": {"role": "assistant", "content": "hello"},
                "usage": {"input": 4, "output": 1, "total": 5},
                "level": "DEFAULT",
            }
        ]
    )
    records = langfuse_to_agentlog(data)
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4.1"
    assert req["messages"][0] == {"role": "user", "content": "hi"}
    assert req["params"]["temperature"] == 0.2
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "hello"
    assert resp["usage"]["input_tokens"] == 4
    assert resp["usage"]["output_tokens"] == 1
    # Latency derived from start/end timestamps (150ms).
    assert resp["latency_ms"] == 150


def test_langfuse_ignores_non_generation_observations() -> None:
    data = _langfuse_export_with(
        [
            {"id": "s1", "type": "span", "name": "retrieve"},
            {
                "id": "g1",
                "type": "generation",
                "model": "m",
                "input": [{"role": "user", "content": "q"}],
                "output": "a",
                "usage": {"input": 1, "output": 1},
            },
        ]
    )
    records = langfuse_to_agentlog(data)
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_langfuse_error_level_maps_to_content_filter() -> None:
    data = _langfuse_export_with(
        [
            {
                "id": "obs-1",
                "type": "generation",
                "model": "m",
                "input": [{"role": "user", "content": "q"}],
                "output": "refused",
                "level": "ERROR",
            }
        ]
    )
    records = langfuse_to_agentlog(data)
    resp = records[2]["payload"]
    assert resp["stop_reason"] == "content_filter"


def test_langfuse_missing_traces_raises_config_error() -> None:
    import pytest

    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError, match="traces"):
        langfuse_to_agentlog({})


# ---------------------------------------------------------------------------
# Braintrust
# ---------------------------------------------------------------------------


def test_braintrust_string_input_output_becomes_user_message_and_text() -> None:
    rows = [
        {
            "input": "what is the capital of France?",
            "output": "Paris",
            "metadata": {"model": "gpt-4.1", "temperature": 0.2},
            "metrics": {"latency": 0.412, "prompt_tokens": 9, "completion_tokens": 1},
        }
    ]
    records = braintrust_to_agentlog(rows)
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4.1"
    assert req["messages"][0] == {"role": "user", "content": "what is the capital of France?"}
    assert req["params"]["temperature"] == 0.2
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "Paris"
    assert resp["latency_ms"] == 412
    assert resp["usage"]["input_tokens"] == 9
    assert resp["usage"]["output_tokens"] == 1


def test_braintrust_dict_input_uses_messages_field() -> None:
    rows = [
        {
            "input": {
                "messages": [
                    {"role": "system", "content": "you are helpful"},
                    {"role": "user", "content": "hi"},
                ]
            },
            "output": {"content": "hello", "finish_reason": "stop"},
            "metadata": {"model": "gpt-4.1"},
            "metrics": {},
        }
    ]
    records = braintrust_to_agentlog(rows)
    req = records[1]["payload"]
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    resp = records[2]["payload"]
    assert resp["stop_reason"] == "end_turn"


def test_braintrust_finish_reason_mapping() -> None:
    rows = [
        {
            "input": "q",
            "output": {"content": "a", "finish_reason": "length"},
            "metadata": {"model": "m"},
            "metrics": {},
        }
    ]
    records = braintrust_to_agentlog(rows)
    assert records[2]["payload"]["stop_reason"] == "max_tokens"


def test_braintrust_non_list_input_raises_config_error() -> None:
    import pytest

    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError):
        braintrust_to_agentlog({"nope": 1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_import_langfuse_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "lf.json"
    src.write_text(
        json.dumps(
            _langfuse_export_with(
                [
                    {
                        "id": "g",
                        "type": "generation",
                        "model": "m",
                        "input": [{"role": "user", "content": "q"}],
                        "output": "a",
                        "usage": {"input": 1, "output": 1},
                    }
                ]
            )
        )
    )
    out = tmp_path / "out.agentlog"
    result = runner.invoke(app, ["import", str(src), "--format", "langfuse", "--output", str(out)])
    assert result.exit_code == 0, result.output
    from shadow import _core

    records = _core.parse_agentlog(out.read_bytes())
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_cli_import_braintrust_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "bt.jsonl"
    row = {"input": "hi", "output": "hey", "metadata": {"model": "m"}, "metrics": {}}
    src.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    out = tmp_path / "out.agentlog"
    result = runner.invoke(
        app, ["import", str(src), "--format", "braintrust", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    from shadow import _core

    records = _core.parse_agentlog(out.read_bytes())
    assert sum(1 for r in records if r["kind"] == "chat_response") == 2
