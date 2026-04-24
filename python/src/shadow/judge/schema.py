"""SchemaConformanceJudge — does the candidate output match an expected schema?

Complements `FormatJudge` (which does mechanical JSON-schema
validation without calling an LLM). This judge uses an LLM to check
**semantic** conformance to a target schema: is each required field
present AND does its value plausibly satisfy the field's intended
meaning?

Example: a customer-support agent is expected to produce

    {"order_id": string, "action": "refund"|"exchange", "amount_usd": number}

`FormatJudge` would accept `{"order_id": "X", "action": "refund",
"amount_usd": "many"}` (string-in-place-of-number is the only issue).
`SchemaConformanceJudge` catches "many" as a semantically invalid
value for `amount_usd` — the field exists, its JSON type is wrong,
and the LLM is good at catching that kind of mistake.

Use FormatJudge for tight structural checks, this for "does the
output mean what the schema says it should."

Usage:

    judge = SchemaConformanceJudge(
        backend=OpenAILLM(),
        expected_schema={
            "order_id": "string (Acme order identifier)",
            "action": "one of: refund, exchange, escalate",
            "amount_usd": "positive number in USD",
        },
    )
"""

from __future__ import annotations

from typing import Any

from shadow.judge.base import JudgeVerdict
from shadow.judge.llm import LlmJudge
from shadow.llm.base import LlmBackend

_RUBRIC_TEMPLATE = """You are reviewing whether the CANDIDATE response conforms to the
EXPECTED OUTPUT SCHEMA below. Check both shape AND meaning: a required
field containing a nonsensical value is a failure, not a pass.

EXPECTED SCHEMA:
__SCHEMA__

TASK:
{task}

CANDIDATE RESPONSE:
{candidate}

Reply with ONLY this JSON object (no preamble, no code fence):
  {{"verdict": "conforms" or "violates",
    "confidence": 0-1,
    "reason": "ONE short sentence naming the first violated field"}}
"""


class SchemaConformanceJudge:
    """Semantic schema conformance via LLM review."""

    def __init__(
        self,
        backend: LlmBackend,
        expected_schema: dict[str, str],
        model: str | None = None,
    ) -> None:
        if not expected_schema:
            raise ValueError("expected_schema must contain at least one field")
        schema_block = "\n".join(
            f"  - `{name}`: {description}" for name, description in expected_schema.items()
        )
        rubric = _RUBRIC_TEMPLATE.replace("__SCHEMA__", schema_block)
        self._judge = LlmJudge(
            backend=backend,
            rubric=rubric,
            score_map={"conforms": 1.0, "violates": 0.0},
            model=model,
        )

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        return await self._judge.score_pair(baseline_response, candidate_response, request_context)


__all__ = ["SchemaConformanceJudge"]
