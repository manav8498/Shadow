"""Tests for ``shadow.adapters.ag2.ShadowAG2Adapter``.

AG2 hooks are pass-through callables on a ``ConversableAgent``. We
construct a real agent with ``llm_config=False`` (so no network) and
invoke the hooks directly — which is exactly what the framework does
around the LLM call. No real LLM traffic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from autogen.agentchat import ConversableAgent
except ImportError:
    pytest.skip("ag2 not installed", allow_module_level=True)

from shadow import _core
from shadow.adapters.ag2 import ShadowAG2Adapter
from shadow.sdk import Session


def _fire_llm_cycle(agent: ConversableAgent, messages: list[dict], response: object) -> None:
    """Drive the safeguard hooks the way AG2's internal send path does."""
    for hook in agent.hook_lists["safeguard_llm_inputs"]:
        hook(messages)
    for hook in agent.hook_lists["safeguard_llm_outputs"]:
        hook(response)


def test_hooks_record_chat_pair(tmp_path: Path) -> None:
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        adapter = ShadowAG2Adapter(s)
        agent = ConversableAgent(
            name="planner",
            llm_config={"config_list": [{"model": "gpt-4.1", "api_key": "fake"}]},
            human_input_mode="NEVER",
        )
        adapter.install(agent)
        _fire_llm_cycle(
            agent,
            [{"role": "user", "content": "plan the migration"}],
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Step 1: backup."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": "gpt-4.1",
            },
        )

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4.1"
    assert req["agent"]["name"] == "planner"
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "Step 1: backup."
    assert resp["usage"]["input_tokens"] == 10
    assert resp["usage"]["output_tokens"] == 5
    assert resp["stop_reason"] == "end_turn"


def test_hooks_capture_tool_calls_in_response(tmp_path: Path) -> None:
    """OpenAI-style tool_calls on the response become tool_use blocks."""
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        adapter = ShadowAG2Adapter(s)
        agent = ConversableAgent(
            name="tooluser",
            llm_config={"config_list": [{"model": "gpt-4.1", "api_key": "fake"}]},
            human_input_mode="NEVER",
        )
        adapter.install(agent)
        _fire_llm_cycle(
            agent,
            [{"role": "user", "content": "what's the weather in SF?"}],
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_w",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": {"city": "SF"},
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "model": "gpt-4.1",
            },
        )

    records = _core.parse_agentlog(out.read_bytes())
    resp = next(r for r in records if r["kind"] == "chat_response")
    tu = next(b for b in resp["payload"]["content"] if b.get("type") == "tool_use")
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"city": "SF"}
    assert resp["payload"]["stop_reason"] == "tool_use"


def test_install_all_wires_multiple_agents(tmp_path: Path) -> None:
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        adapter = ShadowAG2Adapter(s)
        a = ConversableAgent(
            name="a",
            llm_config={"config_list": [{"model": "m-a", "api_key": "fake"}]},
            human_input_mode="NEVER",
        )
        b = ConversableAgent(
            name="b",
            llm_config={"config_list": [{"model": "m-b", "api_key": "fake"}]},
            human_input_mode="NEVER",
        )
        adapter.install_all([a, b])
        _fire_llm_cycle(a, [{"role": "user", "content": "alpha-in"}], "alpha-out")
        _fire_llm_cycle(b, [{"role": "user", "content": "beta-in"}], "beta-out")

    records = _core.parse_agentlog(out.read_bytes())
    responses = [r for r in records if r["kind"] == "chat_response"]
    assert len(responses) == 2
    # Each request carries its agent's name — pairing is intact.
    requests = [r for r in records if r["kind"] == "chat_request"]
    agent_names = {r["payload"]["agent"]["name"] for r in requests}
    assert agent_names == {"a", "b"}


def test_adapter_does_not_swallow_unexpected_response(tmp_path: Path) -> None:
    """Non-OpenAI-shaped responses (raw string, raw dict) still produce a record."""
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        adapter = ShadowAG2Adapter(s)
        agent = ConversableAgent(
            name="plain",
            llm_config={"config_list": [{"model": "gpt-4.1", "api_key": "fake"}]},
            human_input_mode="NEVER",
        )
        adapter.install(agent)
        _fire_llm_cycle(agent, [{"role": "user", "content": "hi"}], "hi back")

    records = _core.parse_agentlog(out.read_bytes())
    resp = next(r for r in records if r["kind"] == "chat_response")
    assert resp["payload"]["content"][0]["text"] == "hi back"
