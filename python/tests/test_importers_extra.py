"""Tests for the LangSmith and OpenAI-Evals importers."""

from __future__ import annotations

import pytest

from shadow.importers import langsmith_to_agentlog, openai_evals_to_agentlog

# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------


def test_langsmith_llm_run_becomes_pair() -> None:
    runs = [
        {
            "id": "r1",
            "run_type": "llm",
            "name": "ChatOpenAI",
            "start_time": "2026-04-21T10:00:00.000000",
            "end_time": "2026-04-21T10:00:00.250000",
            "inputs": {"messages": [[{"role": "user", "content": "hi"}]]},
            "outputs": {
                "generations": [[{"text": "hello", "generation_info": {"finish_reason": "stop"}}]],
                "llm_output": {"token_usage": {"prompt_tokens": 4, "completion_tokens": 1}},
            },
            "extra": {"invocation_params": {"model": "gpt-4o", "temperature": 0.2}},
        }
    ]
    records = langsmith_to_agentlog(runs)
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4o"
    assert req["messages"][0] == {"role": "user", "content": "hi"}
    assert req["params"]["temperature"] == 0.2
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "hello"
    assert resp["latency_ms"] == 250
    assert resp["usage"]["input_tokens"] == 4
    assert resp["usage"]["output_tokens"] == 1


def test_langsmith_ignores_non_llm_runs() -> None:
    runs = [
        {"id": "c1", "run_type": "chain", "name": "main"},
        {
            "id": "r1",
            "run_type": "llm",
            "name": "ChatOpenAI",
            "start_time": "2026-04-21T10:00:00",
            "end_time": "2026-04-21T10:00:00",
            "inputs": {"messages": [[{"role": "user", "content": "x"}]]},
            "outputs": {"generations": [[{"text": "y"}]]},
            "extra": {"invocation_params": {"model": "gpt-4o"}},
        },
        {"id": "t1", "run_type": "tool", "name": "search"},
    ]
    records = langsmith_to_agentlog(runs)
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_langsmith_finish_reason_mapping() -> None:
    runs = [
        {
            "id": "r1",
            "run_type": "llm",
            "name": "x",
            "start_time": "2026-04-21T10:00:00",
            "end_time": "2026-04-21T10:00:00",
            "inputs": {"messages": [[{"role": "user", "content": "q"}]]},
            "outputs": {
                "generations": [[{"text": "a", "generation_info": {"finish_reason": "length"}}]]
            },
            "extra": {"invocation_params": {"model": "m"}},
        }
    ]
    records = langsmith_to_agentlog(runs)
    assert records[2]["payload"]["stop_reason"] == "max_tokens"


def test_langsmith_non_list_raises() -> None:
    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError):
        langsmith_to_agentlog({"nope": 1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OpenAI Evals
# ---------------------------------------------------------------------------


def test_openai_evals_sampling_event_becomes_pair() -> None:
    events = [
        {
            "run_id": "run_1",
            "event_id": "ev_1",
            "sample_id": "s_1",
            "type": "sampling",
            "data": {
                "prompt": [{"role": "user", "content": "hi"}],
                "sampled": ["hello"],
                "options": {"model": "gpt-4o", "temperature": 0.2},
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
            "created_at": "2026-04-21T10:00:00Z",
        }
    ]
    records = openai_evals_to_agentlog(events)
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4o"
    assert req["messages"][0] == {"role": "user", "content": "hi"}
    assert req["params"]["temperature"] == 0.2
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "hello"
    assert resp["usage"]["input_tokens"] == 4
    assert resp["usage"]["output_tokens"] == 1


def test_openai_evals_string_prompt_treated_as_user_message() -> None:
    events = [
        {
            "event_id": "ev",
            "sample_id": "s",
            "type": "sampling",
            "data": {
                "prompt": "what is 2+2?",
                "sampled": ["4"],
                "options": {"model": "m"},
            },
        }
    ]
    records = openai_evals_to_agentlog(events)
    req = records[1]["payload"]
    assert req["messages"][0] == {"role": "user", "content": "what is 2+2?"}


def test_openai_evals_drops_non_sampling_events() -> None:
    events = [
        {"type": "match", "data": {}},
        {"type": "metrics", "data": {}},
        {
            "event_id": "ev",
            "sample_id": "s",
            "type": "sampling",
            "data": {
                "prompt": "x",
                "sampled": ["y"],
                "options": {"model": "m"},
            },
        },
    ]
    records = openai_evals_to_agentlog(events)
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_openai_evals_non_list_raises() -> None:
    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError):
        openai_evals_to_agentlog({"nope": 1})  # type: ignore[arg-type]
