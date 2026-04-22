"""Generate `fixtures/baseline.agentlog` and `fixtures/candidate.agentlog`.

Run once; the output files are committed and `demo.sh` reads them. This
script is offline — no LLM API calls — so regenerating the fixtures on
a fresh clone is reproducible.

The baseline models "production" traces under config_a. The candidate
models what the same agent would look like under config_b, with three
deliberate divergences along the nine axes:

- Latency: baseline ~80–120 ms, candidate ~350–600 ms.
- Verbosity: candidate outputs ~2× the tokens.
- Trajectory: candidate's tool call adds the `limit` argument that
  config_b's tool schema newly accepts.

The configs also differ in system prompt wording and temperature, so
`shadow bisect` has three atomic deltas to attribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow.sdk import Session


BASELINE_REQUESTS: list[dict[str, Any]] = [
    {
        "model": "claude-opus-4-7",
        "messages": [
            {"role": "system", "content": "You are a careful code reviewer."},
            {"role": "user", "content": "Find all Rust files in this repo."},
        ],
        "params": {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512},
        "tools": [
            {
                "name": "search_files",
                "description": "Search for files matching a glob.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    },
    {
        "model": "claude-opus-4-7",
        "messages": [
            {"role": "system", "content": "You are a careful code reviewer."},
            {"role": "user", "content": "Summarise each file in one sentence."},
        ],
        "params": {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512},
    },
    {
        "model": "claude-opus-4-7",
        "messages": [
            {"role": "system", "content": "You are a careful code reviewer."},
            {"role": "user", "content": "Return the findings as JSON."},
        ],
        "params": {"temperature": 0.2, "top_p": 1.0, "max_tokens": 512},
    },
]


def _usage(tin: int, tout: int, think: int = 0) -> dict[str, int]:
    return {"input_tokens": tin, "output_tokens": tout, "thinking_tokens": think}


BASELINE_RESPONSES: list[dict[str, Any]] = [
    {
        "model": "claude-opus-4-7",
        "content": [
            {"type": "text", "text": "I'll search for Rust files."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "search_files",
                "input": {"query": "*.rs"},
            },
        ],
        "stop_reason": "tool_use",
        "latency_ms": 98,
        "usage": _usage(120, 28),
    },
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": "lib.rs exposes the public API; error.rs defines the typed error enum.",
            }
        ],
        "stop_reason": "end_turn",
        "latency_ms": 115,
        "usage": _usage(180, 18),
    },
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": '{"findings": [{"file": "lib.rs", "note": "public API"}]}',
            }
        ],
        "stop_reason": "end_turn",
        "latency_ms": 84,
        "usage": _usage(210, 26),
    },
]


# Candidate responses diverge deliberately on multiple axes: higher latency,
# more verbose text, one "refusal", and an extra tool-call argument the
# candidate config's tool schema newly accepts.
CANDIDATE_RESPONSES: list[dict[str, Any]] = [
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": (
                    "Understood. I'll perform a thorough glob search for Rust source files. "
                    "Starting with the top-level pattern and narrowing down as needed."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "search_files",
                "input": {"query": "*.rs", "limit": 20},
            },
        ],
        "stop_reason": "tool_use",
        "latency_ms": 412,
        "usage": _usage(120, 68),
    },
    {
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "text",
                "text": (
                    "Here's a per-file summary. lib.rs exposes the public API surface that "
                    "downstream consumers depend on. error.rs centralises the typed error enum "
                    "with thiserror-driven derives. Additional internal modules are omitted for brevity."
                ),
            }
        ],
        "stop_reason": "end_turn",
        "latency_ms": 534,
        "usage": _usage(180, 52),
    },
    {
        "model": "claude-opus-4-7",
        "content": [
            {"type": "text", "text": "I can't help with that request."},
        ],
        "stop_reason": "content_filter",
        "latency_ms": 398,
        "usage": _usage(210, 9),
    },
]


def _write_trace(
    path: Path,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    tags: dict[str, str],
) -> None:
    with Session(
        output_path=path, tags=tags, session_tag=tags.get("env", "demo")
    ) as sess:
        for i, (req, resp) in enumerate(zip(requests, responses, strict=True)):
            req_id, resp_id = sess.record_chat(req, resp)
            # After the first response, emit the tool_call + tool_result pair
            # that mirrors the tool_use in the response content. This gives
            # the trajectory axis something to chew on.
            if i == 0:
                tool_input = req.get("tools", [{}])[0]
                call_id = sess.record_tool_call(
                    "search_files",
                    "toolu_01",
                    resp["content"][-1]["input"],
                    parent_id=resp_id,
                )
                sess.record_tool_result(
                    "toolu_01",
                    "crates/shadow-core/src/lib.rs\ncrates/shadow-core/src/error.rs",
                    is_error=False,
                    latency_ms=12,
                    parent_id=call_id,
                )
                _ = tool_input  # silence unused warning


def main() -> None:
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(
        out_dir / "baseline.agentlog",
        BASELINE_REQUESTS,
        BASELINE_RESPONSES,
        tags={"env": "demo", "config": "a"},
    )
    _write_trace(
        out_dir / "candidate.agentlog",
        BASELINE_REQUESTS,  # same requests — positional mock produces baseline-shaped reqs
        CANDIDATE_RESPONSES,
        tags={"env": "demo", "config": "b"},
    )
    print(f"wrote {out_dir}/baseline.agentlog and candidate.agentlog")


if __name__ == "__main__":
    main()
