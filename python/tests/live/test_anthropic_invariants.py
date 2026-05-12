"""Live invariants against the real Anthropic Messages API.

Gated by ``SHADOW_RUN_NETWORK_TESTS=1`` + ``ANTHROPIC_API_KEY`` (see
``conftest.py``). Each test asserts a single invariant that breaks
when the provider changes shape — response shape, streaming-
aggregation correctness, tool-use round-tripping through Shadow.

Token budget: ~$0.005 across these tests on ``claude-haiku-4-5``
with short prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow import _core

# Cheapest Claude tier currently published; matches the model id
# already in use by `test_judge_live.py`.
MODEL = "claude-haiku-4-5-20251001"


def _records_from(path: Path) -> list[dict[str, Any]]:
    return _core.parse_agentlog(path.read_bytes())


def test_messages_response_shape(live_anthropic_client: Any) -> None:
    """Minimal Messages call: content blocks present + non-empty,
    usage block carries input/output token counts. Catches a SDK
    response-shape change before users hit it.
    """
    client = live_anthropic_client
    response = client.messages.create(
        model=MODEL,
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with 'OK'."}],
    )
    assert response.content, "expected at least one content block"
    text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
    assert text_blocks, "expected at least one text content block"
    assert text_blocks[0].text.strip(), "expected non-empty text"
    assert response.usage is not None
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0
    assert response.model.startswith(
        "claude-haiku-4-5"
    ), f"returned model '{response.model}' diverged from requested family"


def test_messages_streaming_aggregates_correctly(
    live_anthropic_client: Any, shadow_session: tuple[Any, Path]
) -> None:
    """Streaming Messages + auto-instrument emits exactly one
    ``chat_request`` + one ``chat_response`` record per call (not one
    per chunk), and the aggregated text is non-empty.
    """
    session, out = shadow_session
    prompt = "Reply with exactly the single word 'OK' and nothing else."

    with session:
        client = live_anthropic_client
        with client.messages.stream(
            model=MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for _ in stream.text_stream:
                pass

    records = _records_from(out)
    chat_requests = [r for r in records if r["kind"] == "chat_request"]
    chat_responses = [r for r in records if r["kind"] == "chat_response"]
    assert len(chat_requests) == 1, (
        f"expected exactly 1 chat_request, got {len(chat_requests)}: "
        "Shadow's stream aggregator must coalesce chunks into a single pair"
    )
    assert len(chat_responses) == 1, f"expected exactly 1 chat_response, got {len(chat_responses)}"

    resp_payload = chat_responses[0]["payload"]
    text_blocks = [b for b in resp_payload.get("content", []) if b.get("type") == "text"]
    assert text_blocks, "expected aggregated text content in recorded response"
    aggregated = "".join(b["text"] for b in text_blocks).strip().lower()
    assert "ok" in aggregated, f"aggregated text missing expected token: {aggregated!r}"


def test_tool_use_block_round_trip(
    live_anthropic_client: Any, shadow_session: tuple[Any, Path]
) -> None:
    """Messages call with a tool spec that the model is likely to
    invoke: ``tool_use`` blocks must survive the auto-instrument
    translator and land in the recorded ``chat_response.content``.
    """
    session, out = shadow_session
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name."}},
                "required": ["city"],
            },
        }
    ]

    with session:
        client = live_anthropic_client
        client.messages.create(
            model=MODEL,
            max_tokens=200,
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo? Use the tool.",
                }
            ],
        )

    records = _records_from(out)
    chat_responses = [r for r in records if r["kind"] == "chat_response"]
    assert chat_responses, "expected at least one chat_response record"
    content = chat_responses[-1]["payload"].get("content", [])
    tool_uses = [b for b in content if b.get("type") == "tool_use"]
    assert tool_uses, (
        "expected tool_use block in recorded chat_response.content; "
        "either the model declined to call the tool (unlikely with this "
        "prompt) or the auto-instrument translator dropped it"
    )
    # Each tool_use block carries the tool name and a parsed input.
    block = tool_uses[0]
    assert block.get("name") == "get_weather"
    assert "input" in block, "tool_use block missing 'input'"
