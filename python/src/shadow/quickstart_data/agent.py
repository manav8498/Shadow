"""A toy agent that records its LLM interactions with `shadow.sdk.Session`.

Illustrative — run against a real provider to produce a baseline trace.
For the offline demo, `demo.sh` uses the pre-committed fixtures under
`fixtures/` instead of calling a live API; this file exists so README
readers can see the shape of an instrumented agent.

Run: python agent.py --output baseline.agentlog
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml

from shadow.llm import LlmBackend, PositionalMockLLM
from shadow.sdk import Session


async def ask(
    backend: LlmBackend,
    session: Session,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": "claude-opus-4-7",
        "messages": messages,
        "params": {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512},
    }
    if tools:
        request["tools"] = tools
    response = await backend.complete(request)
    session.record_chat(request, response)
    return response


async def main(config_path: Path, output: Path, reference: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    tools = config.get("tools", [])
    system = config["prompt"]["system"]

    # In a real deployment this is AnthropicLLM / OpenAILLM. The demo
    # uses a PositionalMockLLM backed by a recorded reference so the
    # example is fully offline.
    backend: LlmBackend = PositionalMockLLM.from_path(reference)

    with Session(output_path=output, tags={"env": "demo"}, session_tag="demo-agent") as sess:
        # Turn 1: tool-using question.
        await ask(
            backend,
            sess,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Find all Rust files in this repo."},
            ],
            tools=tools,
        )
        # Turn 2: follow-up summarisation.
        await ask(
            backend,
            sess,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Summarise each file in one sentence."},
            ],
        )
        # Turn 3: structured output.
        await ask(
            backend,
            sess,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Return the findings as JSON."},
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("baseline.agentlog"))
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_a.yaml")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "baseline.agentlog",
    )
    args = parser.parse_args()
    asyncio.run(main(args.config, args.output, args.reference))
