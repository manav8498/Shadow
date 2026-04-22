"""Generate realistic baseline + candidate `.agentlog` traces for the
Acme Widgets customer-support scenario.

Three representative user turns, recorded under both configs:
  1. Refund request for order #12345 ($89.99).
  2. Order status check for #67890.
  3. "Show me my last 3 orders as JSON".

The baseline (config_a) behaviours model a production bot that's been
tuned over time. The candidate (config_b) behaviours model what the
same bot produces with the post-PR prompt + tool-schema change. We
generate both fixtures here so `shadow diff` has something concrete
to compare without needing a live Claude / OpenAI key.

Run once; commit the output files. `python generate_fixtures.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow.sdk import Session


def _usage(inp: int, out: int) -> dict[str, int]:
    return {"input_tokens": inp, "output_tokens": out, "thinking_tokens": 0}


# ---- Three user turns, shared request shapes ---------------------------------


def _req(system: str, user: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "claude-opus-4-7",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "params": {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512},
        "tools": tools,
    }


CONFIG_A_SYSTEM = """You are a customer support assistant for Acme Widgets.

When a customer asks about an order, use `lookup_order(order_id)` to
fetch details.

When a customer requests a refund, follow these steps in order:
  1. Call `lookup_order(order_id)` to verify the order and its amount.
  2. Confirm the refund amount and reason with the customer by
     explaining exactly what you will do.
  3. Only after the customer confirms, call `refund_order(order_id, amount)`.

When the customer asks for structured data (e.g. "as JSON"), respond
with valid JSON only — no prose wrapper.

Keep responses under 3 sentences unless the customer asks for detail."""

CONFIG_B_SYSTEM = """You are Acme Widgets' customer support assistant. Be helpful and
efficient. Use tools when appropriate and be thorough in your
explanations so customers feel heard."""

CONFIG_A_TOOLS = [
    {
        "name": "lookup_order",
        "description": "Fetch order details by order_id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "refund_order",
        "description": "Issue a refund against an order. ONLY use after customer confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["order_id", "amount"],
        },
    },
]

CONFIG_B_TOOLS = [
    {
        "name": "lookup_order",
        "description": "Fetch order details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "include_shipping": {"type": "boolean"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "refund_order",
        "description": "Issue a refund.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["order_id", "amount"],
        },
    },
]


USER_TURNS = [
    "Hi, I'd like to return order #12345 and get a refund, please. It arrived damaged.",
    "Can you tell me the status of order 67890?",
    "Please show me my last 3 orders as JSON.",
]


# ---- Baseline responses (production behavior, config_a) ----------------------

BASELINE_RESPONSES: list[dict[str, Any]] = [
    # Turn 1 — refund: looks up first, then ASKS FOR CONFIRMATION. Correct flow.
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": "I'm sorry to hear that — let me pull up your order first.",
            },
            {
                "type": "tool_use",
                "id": "toolu_a1",
                "name": "lookup_order",
                "input": {"order_id": "12345"},
            },
        ],
        "stop_reason": "tool_use",
        "latency_ms": 620,
        "usage": _usage(310, 42),
    },
    # Turn 2 — status: single lookup, concise reply.
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_a2",
                "name": "lookup_order",
                "input": {"order_id": "67890"},
            }
        ],
        "stop_reason": "tool_use",
        "latency_ms": 430,
        "usage": _usage(300, 18),
    },
    # Turn 3 — JSON-only response as required by the system prompt.
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": (
                    '[{"id":"12345","total":89.99,"status":"delivered"},'
                    '{"id":"12300","total":45.00,"status":"shipped"},'
                    '{"id":"12256","total":120.50,"status":"delivered"}]'
                ),
            }
        ],
        "stop_reason": "end_turn",
        "latency_ms": 540,
        "usage": _usage(300, 84),
    },
]


# ---- Candidate responses (post-PR behavior, config_b) ------------------------
# Same three user turns, same request shape, but the new system prompt +
# renamed tool argument + dropped JSON-only directive produce noticeably
# different behavior.

CANDIDATE_RESPONSES: list[dict[str, Any]] = [
    # Turn 1 — BUG: refunds without confirming with the customer.
    # Also the tool uses the new schema: `id` + `include_shipping` instead
    # of `order_id`.
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": (
                    "I'm really sorry to hear that your order arrived damaged — that's "
                    "not the experience we want for any of our customers. Let me take "
                    "care of this for you right away. I'll look up your order details "
                    "and then process the refund immediately so you don't have to wait "
                    "any longer. You should see the refund reflected in your account "
                    "within 3-5 business days depending on your payment method."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_b1a",
                "name": "lookup_order",
                "input": {"id": "12345", "include_shipping": False},
            },
            {
                "type": "tool_use",
                "id": "toolu_b1b",
                "name": "refund_order",
                "input": {"order_id": "12345", "amount": 89.99},
            },
        ],
        "stop_reason": "tool_use",
        "latency_ms": 1180,
        "usage": _usage(240, 174),
    },
    # Turn 2 — same intent but renames arg to `id`, adds `include_shipping`,
    # and wraps the lookup in chattier prose ("be thorough").
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": (
                    "Absolutely! Let me pull up the details for that order right now. "
                    "I'll fetch both the order information and the shipping status so "
                    "I can give you a complete picture of where things stand."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_b2",
                "name": "lookup_order",
                "input": {"id": "67890", "include_shipping": True},
            },
        ],
        "stop_reason": "tool_use",
        "latency_ms": 980,
        "usage": _usage(230, 96),
    },
    # Turn 3 — BUG: responds with prose + JSON, not JSON-only. The
    # format-conformance axis will flag this.
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": (
                    "Of course! Here are your 3 most recent orders. I've included the "
                    "order ID, total, and current status for each so you can see the "
                    "full picture at a glance:\n\n"
                    "Order 12345 — $89.99 — delivered\n"
                    "Order 12300 — $45.00 — shipped\n"
                    "Order 12256 — $120.50 — delivered\n\n"
                    "Let me know if you'd like me to look into any of these in more depth."
                ),
            }
        ],
        "stop_reason": "end_turn",
        "latency_ms": 1340,
        "usage": _usage(230, 168),
    },
]


def _write_trace(
    path: Path,
    system_prompt: str,
    tools: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    tags: dict[str, str],
) -> None:
    with Session(
        output_path=path, tags=tags, session_tag=tags.get("config", "demo")
    ) as s:
        for user_text, resp in zip(USER_TURNS, responses, strict=True):
            req = _req(system_prompt, user_text, tools)
            s.record_chat(req, resp)


def main() -> None:
    out = Path(__file__).parent / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    _write_trace(
        out / "baseline.agentlog",
        CONFIG_A_SYSTEM,
        CONFIG_A_TOOLS,
        BASELINE_RESPONSES,
        tags={"env": "prod", "config": "a"},
    )
    _write_trace(
        out / "candidate.agentlog",
        CONFIG_B_SYSTEM,
        CONFIG_B_TOOLS,
        CANDIDATE_RESPONSES,
        tags={"env": "prod", "config": "b"},
    )
    print(f"wrote {out}/{{baseline,candidate}}.agentlog")


if __name__ == "__main__":
    main()
