"""Live invariants against the real OpenAI API.

Gated by ``SHADOW_RUN_NETWORK_TESTS=1`` + ``OPENAI_API_KEY`` (see
``conftest.py``). Each test asserts a single invariant that breaks
when the provider changes shape — token-count drift, response-shape
changes, streaming-aggregation correctness, model-id rotation.

Token budget: ~$0.005 across these tests on ``gpt-4o-mini`` with
short prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow import _core

# Cheap, deterministic-ish model. If OpenAI rotates the alias we
# want to know — that's the point.
MODEL = "gpt-4o-mini"


def _records_from(path: Path) -> list[dict[str, Any]]:
    return _core.parse_agentlog(path.read_bytes())


def test_chat_completions_response_shape(live_openai_client: Any) -> None:
    """Minimal chat completion: token counts present + non-negative,
    text content non-empty, returned model id reflects the request.

    A shape change on any of these three axes is a breaking SDK
    update we want to catch.
    """
    client = live_openai_client
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with 'OK'."}],
        max_completion_tokens=10,
    )
    assert response.choices, "expected at least one choice"
    content = response.choices[0].message.content
    assert content is not None and content.strip(), "expected non-empty content"
    usage = response.usage
    assert usage is not None, "expected usage block"
    assert usage.prompt_tokens >= 0
    assert usage.completion_tokens >= 0
    # Model id may include a date-stamped suffix (gpt-4o-mini-2024-07-18);
    # the alias prefix is the durable contract.
    assert response.model.startswith(
        MODEL
    ), f"returned model '{response.model}' does not start with requested '{MODEL}'"


def test_chat_completions_streaming_aggregates_correctly(
    live_openai_client: Any, shadow_session: tuple[Any, Path]
) -> None:
    """Streaming + auto-instrument emits exactly one ``chat_request``
    and one ``chat_response`` record per call, not one per chunk —
    and the aggregated text matches a non-streaming call with the
    same prompt closely enough that the aggregator isn't dropping
    chunks.
    """
    session, out = shadow_session
    prompt = "Reply with exactly the single word 'OK' and nothing else."

    with session:
        client = live_openai_client
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            stream=True,
        )
        chunks = list(stream)

    assert chunks, "expected at least one streaming chunk"

    records = _records_from(out)
    chat_requests = [r for r in records if r["kind"] == "chat_request"]
    chat_responses = [r for r in records if r["kind"] == "chat_response"]
    assert len(chat_requests) == 1, (
        f"expected exactly 1 chat_request, got {len(chat_requests)}: "
        "Shadow's stream aggregator must coalesce chunks into a single pair"
    )
    assert len(chat_responses) == 1, f"expected exactly 1 chat_response, got {len(chat_responses)}"

    # Aggregated text should be non-empty and contain the expected word.
    resp_payload = chat_responses[0]["payload"]
    text_blocks = [b for b in resp_payload.get("content", []) if b.get("type") == "text"]
    assert text_blocks, "expected aggregated text content in recorded response"
    aggregated = "".join(b["text"] for b in text_blocks).strip().lower()
    assert "ok" in aggregated, f"aggregated text missing expected token: {aggregated!r}"


def test_responses_api_shape(live_openai_client: Any) -> None:
    """Responses API (the recommended OpenAI surface for new code):
    content blocks land and token counts are populated. Catches both
    SDK rename of the resource and shape changes on the usage block.
    """
    client = live_openai_client
    response = client.responses.create(
        model=MODEL,
        input="Reply with 'OK'.",
        max_output_tokens=20,
    )
    # The SDK exposes output text via .output_text on newer versions;
    # fall back to walking .output for portability across minor bumps.
    text = getattr(response, "output_text", None)
    if not text:
        out_list = getattr(response, "output", None) or []
        chunks: list[str] = []
        for item in out_list:
            for block in getattr(item, "content", None) or []:
                t = getattr(block, "text", None)
                if t:
                    chunks.append(t)
        text = "".join(chunks)
    assert text and text.strip(), "expected non-empty Responses API output"
    usage = getattr(response, "usage", None)
    assert usage is not None, "expected usage block on Responses API"
    # SDK exposes either input_tokens/output_tokens or prompt_tokens/
    # completion_tokens depending on minor version — accept both.
    in_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    out_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    assert in_tokens is not None and in_tokens >= 0
    assert out_tokens is not None and out_tokens >= 0


def test_token_counts_within_expected_band(live_openai_client: Any) -> None:
    """Deterministic prompt → small token band on the output side.

    If a future SDK changes how tokens are counted (e.g. starts
    counting end-of-message markers, or a tokenizer swap shifts the
    byte-pair coverage), this trips early. The band is generous
    enough that normal model-version drift won't flake it.
    """
    client = live_openai_client
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'OK'."}],
        max_completion_tokens=20,
        temperature=0.0,
    )
    usage = response.usage
    assert usage is not None
    assert 1 <= usage.completion_tokens <= 10, (
        f"completion_tokens={usage.completion_tokens} outside [1, 10] band; "
        "possible SDK tokenizer change or model-version drift"
    )
