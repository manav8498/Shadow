"""Importers for foreign trace-export formats → Shadow `.agentlog`.

Turns Shadow's open spec from "yet another format" into a bridge: if
you've already instrumented your agent with Langfuse or Braintrust,
you can run Shadow's differ on the data you already have without
re-recording.

Supported today:

- **Langfuse** — the `langfuse export` JSON shape: a top-level
  `traces` list where each trace has nested `observations` (spans),
  typed as `"generation"` for LLM calls and `"span"` for everything
  else. Fields we map: `input.messages`, `output.content`, `model`,
  `modelParameters`, `usage.{input,output,total}`, `startTime`,
  `endTime`, `level` (for errors).

- **Braintrust** — the `braintrust export experiment` row shape:
  one JSON object per line (or a JSON array) with `input`, `output`,
  `metadata`, `metrics`, and `tags`. We map `input` → chat request,
  `output` → chat response, and `metrics.latency` → `latency_ms`.

Both importers are best-effort: unknown fields are ignored, missing
required fields raise `ShadowConfigError` with a helpful hint.
"""

from shadow.importers.braintrust import BRAINTRUST_FORMAT, braintrust_to_agentlog
from shadow.importers.langfuse import LANGFUSE_FORMAT, langfuse_to_agentlog
from shadow.importers.langsmith import LANGSMITH_FORMAT, langsmith_to_agentlog
from shadow.importers.mcp import MCP_FORMAT, mcp_to_agentlog
from shadow.importers.openai_evals import OPENAI_EVALS_FORMAT, openai_evals_to_agentlog
from shadow.importers.otel import OTEL_FORMAT, otel_to_agentlog
from shadow.importers.pydantic_ai import PYDANTIC_AI_FORMAT, pydantic_ai_to_agentlog
from shadow.importers.vercel_ai import VERCEL_AI_FORMAT, vercel_ai_to_agentlog

__all__ = [
    "BRAINTRUST_FORMAT",
    "LANGFUSE_FORMAT",
    "LANGSMITH_FORMAT",
    "MCP_FORMAT",
    "OPENAI_EVALS_FORMAT",
    "OTEL_FORMAT",
    "PYDANTIC_AI_FORMAT",
    "VERCEL_AI_FORMAT",
    "braintrust_to_agentlog",
    "langfuse_to_agentlog",
    "langsmith_to_agentlog",
    "mcp_to_agentlog",
    "openai_evals_to_agentlog",
    "otel_to_agentlog",
    "pydantic_ai_to_agentlog",
    "vercel_ai_to_agentlog",
]
