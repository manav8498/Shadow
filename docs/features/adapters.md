# Framework adapters

Shadow's core instrumentation patches the Anthropic and OpenAI SDKs directly, which covers the majority of agents since those are the underlying LLM clients. Agents built on a framework layer above that client also get captured, but the framework's own structure (graph nodes, crew kickoffs, agent conversations) is invisible to direct SDK patching. The adapters in `shadow.adapters.*` hook the framework's native extension surface so that structure lands in the `.agentlog` too.

Three framework adapters ship in v1.4:

| Framework | Module | Install |
|---|---|---|
| LangGraph / LangChain | `shadow.adapters.langgraph` | `pip install 'shadow-diff[langgraph]'` |
| CrewAI | `shadow.adapters.crewai` | `pip install 'shadow-diff[crewai]'` |
| AG2 (formerly AutoGen) | `shadow.adapters.ag2` | `pip install 'shadow-diff[ag2]'` |

## LangGraph / LangChain

The adapter is an `AsyncCallbackHandler` subclass. Drop it into `RunnableConfig.callbacks`:

The `[langgraph]` extra pulls `langchain-core`, `langgraph`, and `langchain-openai` (the chat provider most LangGraph users pick). To run against Anthropic, Bedrock, or something else, add the matching LangChain integration alongside — `pip install 'shadow-diff[langgraph]' langchain-anthropic` and so on. The adapter itself is provider-neutral and doesn't care which chat model you instantiate.


```python
from shadow.sdk import Session
from shadow.adapters.langgraph import ShadowLangChainHandler

with Session(output_path="trace.agentlog") as s:
    handler = ShadowLangChainHandler(s)
    result = await graph.ainvoke(
        {"messages": [HumanMessage("...")]},
        config={
            "callbacks": [handler],
            "configurable": {"thread_id": "t-42"},
        },
    )
```

Hooks:

- `on_chat_model_start` / `on_llm_end` / `on_llm_error` produce the `chat_request` and `chat_response` pair
- `on_tool_start` / `on_tool_end` / `on_tool_error` produce `tool_call` / `tool_result` records

The handler pair-buffers by LangChain's `run_id` so concurrent graph branches (which LangGraph can spawn for fan-outs) never cross-contaminate. The `thread_id` from the config's `configurable` block carries through as the session boundary, so one graph invocation is one session even across tool loops.

Works under both sync `invoke` and async `ainvoke`. Subclassing `AsyncCallbackHandler` avoids the known sync-on-async race where LangChain dispatches sync callbacks through `loop.run_in_executor`.

## CrewAI

The adapter is a `BaseEventListener` subclass wired to CrewAI's `crewai_event_bus`. Instantiate it inside your Session:

```python
from shadow.sdk import Session
from shadow.adapters.crewai import ShadowCrewAIListener

with Session(output_path="trace.agentlog") as s:
    ShadowCrewAIListener(s)
    result = crew.kickoff(inputs={"topic": "..."})
```

Wired events:

- `LLMCallStartedEvent` / `LLMCallCompletedEvent` / `LLMCallFailedEvent` produce the chat pair
- `ToolUsageStartedEvent` / `ToolUsageFinishedEvent` / `ToolUsageErrorEvent` produce the tool pair
- `CrewKickoffStartedEvent` writes an authoritative metadata marker so Shadow's session detector treats one kickoff as one session, even though every `LLMCallCompleted` ends with `end_turn`

The `call_id` field on the LLM events is the pairing key for the chat pair, so concurrent crews never mix responses.

If you're driving the event bus with synthetic events in tests, pass `quiet_internal_listeners=True` to silence CrewAI's built-in telemetry handlers that expect real `Crew` objects:

```python
ShadowCrewAIListener(s, quiet_internal_listeners=True)
```

## AG2

AG2's `ConversableAgent.register_hook` is the canonical extension surface. The adapter wraps it and captures every LLM call routed through the registered agents:

```python
from autogen.agentchat import ConversableAgent
from shadow.sdk import Session
from shadow.adapters.ag2 import ShadowAG2Adapter

planner = ConversableAgent(name="planner", llm_config={...}, human_input_mode="NEVER")
executor = ConversableAgent(name="executor", llm_config={...}, human_input_mode="NEVER")

with Session(output_path="trace.agentlog") as s:
    adapter = ShadowAG2Adapter(s)
    adapter.install_all([planner, executor])
    planner.initiate_chat(executor, message="plan the migration")
```

Hooks:

- `safeguard_llm_inputs` captures the full messages list going into each LLM call
- `safeguard_llm_outputs` captures the response

AG2 also ships an OTel exporter as of Feb 2026 (`autogen.opentelemetry`), which emits GenAI-compliant spans but redacts message bodies by default. The adapter grabs the bodies the exporter won't give you. If you want both (timing metrics plus content), run the OTel exporter alongside Shadow and use `shadow import --format otel` on the exported file; the v1.40 importer reads whatever the exporter emits.

Per-agent registration is per-instance. A `GroupChat` of five agents needs five `install()` calls, or one `install_all([...])`.

## Session grouping and policy evaluation

All three adapters feed through the same `Session.record_chat` / `record_tool_call` / `record_tool_result` contract as direct SDK instrumentation, so every downstream feature works identically:

- `shadow diff` produces the nine-axis behavioral diff
- `shadow check-policy` with `scope: session` evaluates each framework's natural session boundary correctly
- `shadow mine` clusters turn-pairs by tool sequence and selects representative cases
- `shadow mcp-serve` exposes the captured traces to any MCP-aware client

If your trace contains multiple sessions (many kickoffs, many threads, many conversations) and you want a per-session breakdown, session-scoped policy rules work out of the box for LangGraph (via `thread_id`) and CrewAI (via kickoff markers). AG2 infers session boundaries from the prior response's `stop_reason`, so one `initiate_chat` cycle is one session.
