"""Tests for ``shadow.adapters.langgraph.ShadowLangChainHandler``.

Real langchain-core + langgraph are required (installed via the
``shadow-diff[langgraph]`` extra). Tests use ``FakeListChatModel`` and
``FakeMessagesListChatModel`` so no real LLM calls happen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pytest

try:
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode
except ImportError:
    pytest.skip("langchain-core / langgraph not installed", allow_module_level=True)

from shadow import _core
from shadow.adapters.langgraph import ShadowLangChainHandler
from shadow.sdk import Session


class State(TypedDict):
    """Shared graph state for all tests in this file.

    Must live at module scope so LangGraph's pydantic forward-reference
    resolution inside ``StateGraph.compile()`` can find it.
    """

    messages: list[Any]


# ---- minimal single-call graph -------------------------------------------


@pytest.mark.asyncio
async def test_handler_records_chat_pair_from_single_node_graph(tmp_path: Path) -> None:
    """A minimal graph with one chat-model node should produce one pair."""
    out = tmp_path / "trace.agentlog"
    fake = FakeListChatModel(responses=["hello there"])

    def call_model(state: State) -> dict:
        resp = fake.invoke(state["messages"])
        return {"messages": state["messages"] + [resp]}

    graph = StateGraph(State)
    graph.add_node("chat", call_model)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    compiled = graph.compile()

    with Session(output_path=out, auto_instrument=False) as s:
        handler = ShadowLangChainHandler(s)
        await compiled.ainvoke(
            {"messages": [HumanMessage("hi")]},
            config={"callbacks": [handler], "configurable": {"thread_id": "t-1"}},
        )

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["messages"][0]["content"] == "hi"
    assert req.get("thread_id") == "t-1"
    resp = records[2]["payload"]
    assert any(b.get("text") == "hello there" for b in resp["content"])
    assert resp["stop_reason"] == "end_turn"


# ---- tool-calling graph --------------------------------------------------


@pytest.mark.asyncio
async def test_handler_captures_tool_call_and_result(tmp_path: Path) -> None:
    """A prebuilt ToolNode flow should emit chat pair + tool_call + tool_result."""
    out = tmp_path / "trace.agentlog"

    @tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    ai_with_tool = AIMessage(
        content="",
        tool_calls=[{"id": "call_abc", "name": "get_weather", "args": {"city": "SF"}}],
    )
    ai_final = AIMessage(content="It is sunny in SF.")
    fake = FakeMessagesListChatModel(responses=[ai_with_tool, ai_final])

    def call_model(state: State) -> dict:
        resp = fake.invoke(state["messages"])
        return {"messages": state["messages"] + [resp]}

    def should_continue(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    tool_node = ToolNode([get_weather])

    graph = StateGraph(State)
    graph.add_node("chat", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "chat")
    compiled = graph.compile()

    with Session(output_path=out, auto_instrument=False) as s:
        handler = ShadowLangChainHandler(s)
        await compiled.ainvoke(
            {"messages": [HumanMessage("what's the weather in SF?")]},
            config={"callbacks": [handler], "configurable": {"thread_id": "t-weather"}},
        )

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds.count("chat_request") == 2
    assert kinds.count("chat_response") == 2
    assert kinds.count("tool_call") == 1
    assert kinds.count("tool_result") == 1

    tool_resp = next(
        r
        for r in records
        if r["kind"] == "chat_response"
        and any(b.get("type") == "tool_use" for b in r["payload"]["content"])
    )
    tool_use = next(b for b in tool_resp["payload"]["content"] if b.get("type") == "tool_use")
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == {"city": "SF"}
    assert tool_resp["payload"]["stop_reason"] == "tool_use"

    tool_result = next(r for r in records if r["kind"] == "tool_result")
    assert "sunny in SF" in str(tool_result["payload"]["output"])


# ---- error path ----------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_records_error_as_stop_reason(tmp_path: Path) -> None:
    """on_llm_error should produce a chat_response with stop_reason=error."""
    out = tmp_path / "trace.agentlog"

    class BoomModel(FakeListChatModel):
        def _call(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("upstream 503")

        async def _acall(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("upstream 503")

    fake = BoomModel(responses=["never reached"])

    async def call_model(state: State) -> dict:
        resp = await fake.ainvoke(state["messages"])
        return {"messages": state["messages"] + [resp]}

    graph = StateGraph(State)
    graph.add_node("chat", call_model)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    compiled = graph.compile()

    with Session(output_path=out, auto_instrument=False) as s:
        handler = ShadowLangChainHandler(s)
        with pytest.raises(RuntimeError):
            await compiled.ainvoke(
                {"messages": [HumanMessage("hi")]},
                config={"callbacks": [handler]},
            )

    records = _core.parse_agentlog(out.read_bytes())
    resp = next(r for r in records if r["kind"] == "chat_response")
    assert resp["payload"]["stop_reason"] == "error"
    assert "RuntimeError" in str(resp["payload"].get("error", {}).get("type", ""))


# ---- concurrent fan-out safety -------------------------------------------


@pytest.mark.asyncio
async def test_handler_is_safe_under_concurrent_branches(tmp_path: Path) -> None:
    """Two concurrent LLM calls with distinct run_ids must not cross-contaminate.

    LangGraph parallel sub-graphs routinely fan out; the handler's
    ``run_id`` -> pending-dict keying must survive that.
    """
    import asyncio

    out = tmp_path / "trace.agentlog"

    fake_a = FakeListChatModel(responses=["a-response"])
    fake_b = FakeListChatModel(responses=["b-response"])

    with Session(output_path=out, auto_instrument=False) as s:
        handler = ShadowLangChainHandler(s)
        await asyncio.gather(
            fake_a.ainvoke([HumanMessage("alpha")], config={"callbacks": [handler]}),
            fake_b.ainvoke([HumanMessage("beta")], config={"callbacks": [handler]}),
        )

    records = _core.parse_agentlog(out.read_bytes())
    responses = [r for r in records if r["kind"] == "chat_response"]
    assert len(responses) == 2
    response_texts = {
        next(b["text"] for b in r["payload"]["content"] if b.get("type") == "text")
        for r in responses
    }
    assert response_texts == {"a-response", "b-response"}


# ---- smoke import --------------------------------------------------------


def test_module_exports_shadow_handler_alias() -> None:
    from shadow.adapters.langgraph import ShadowHandler, ShadowLangChainHandler

    assert ShadowHandler is ShadowLangChainHandler
