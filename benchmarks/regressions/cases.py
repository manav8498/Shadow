"""Twenty curated agent-regression cases, each expressed as a generator.

Each case is a function returning `(baseline_records, candidate_records,
expected)` where `expected` names:

  - `axis`: the Shadow axis that MUST move (e.g. "trajectory")
  - `min_severity`: the minimum severity the axis should report
    (one of: "minor", "moderate", "severe")
  - `description`: one-line human description of the regression class

The 20 cases span:

  1-5   Tool-call regressions (reorder, rename, skip, schema-drift, arg-swap)
  6-10  Text-output regressions (verbosity blowup, truncation, over-apology,
        language drift, formatting loss)
  11-13 Safety / refusal regressions (over-refusal, missed-refusal, hedging)
  14-16 Format-conformance regressions (JSON→prose, missing-field, extra-text)
  17-19 Latency / cost regressions (5x latency, cheaper model but slower,
        token blowup)
  20    Control: identical baseline and candidate (expected: no regression)

This is the permanent regression-guard for Shadow itself. If Shadow
ever stops catching any of these, the benchmark run fails.
"""

from __future__ import annotations

from typing import Any, Callable

Case = Callable[[], tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]


def _meta(record_id: str = "a" * 64) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{record_id}",
        "kind": "metadata",
        "ts": "2026-04-21T10:00:00.000Z",
        "parent": None,
        "payload": {"sdk": {"name": "shadow", "version": "0.1.0"}},
    }


def _chat(
    idx: int,
    kind: str,
    payload: dict[str, Any],
    parent: str | None = None,
) -> dict[str, Any]:
    rid = f"sha256:{kind[0]}{idx:063d}"
    return {
        "version": "0.1",
        "id": rid,
        "kind": kind,
        "ts": f"2026-04-21T10:{idx // 60:02d}:{idx % 60:02d}.000Z",
        "parent": parent,
        "payload": payload,
    }


def _response(
    idx: int,
    content: list[dict[str, Any]],
    *,
    model: str = "claude-opus-4-7",
    stop_reason: str = "end_turn",
    latency_ms: int = 100,
    input_tokens: int = 20,
    output_tokens: int = 10,
    thinking_tokens: int = 0,
    parent: str | None = None,
) -> dict[str, Any]:
    return _chat(
        idx,
        "chat_response",
        {
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
            },
        },
        parent=parent,
    )


def _request(idx: int, text: str) -> dict[str, Any]:
    return _chat(
        idx,
        "chat_request",
        {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": text}],
            "params": {"temperature": 0.0, "max_tokens": 512},
        },
    )


def _build_trace(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend a metadata + chat_request before each response."""
    records: list[dict[str, Any]] = [_meta()]
    for i, resp in enumerate(responses):
        req = _request(i, f"task {i}")
        records.append(req)
        records.append({**resp, "parent": req["id"]})
    return records


# ---------------------------------------------------------------------------
# Tool-call regressions (1-5)
# ---------------------------------------------------------------------------


def case_01_tool_reorder() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [
                {"type": "tool_use", "id": f"c1_{i}", "name": "search", "input": {"q": "x"}},
                {"type": "tool_use", "id": f"c2_{i}", "name": "fetch", "input": {"url": "x"}},
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    candidate_resps = [
        _response(
            i,
            [
                {"type": "tool_use", "id": f"c1_{i}", "name": "fetch", "input": {"url": "x"}},
                {"type": "tool_use", "id": f"c2_{i}", "name": "search", "input": {"q": "x"}},
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "trajectory",
            "min_severity": "minor",
            "description": "tools called in reversed order",
        },
    )


def case_02_tool_rename() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [{"type": "tool_use", "id": f"c_{i}", "name": "search_files", "input": {"q": "x"}}],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "tool_use", "id": f"c_{i}", "name": "grep_files", "input": {"q": "x"}}],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "trajectory",
            "min_severity": "minor",
            "description": "tool was renamed between configs",
        },
    )


def case_03_tool_skip() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [
                {"type": "tool_use", "id": f"c1_{i}", "name": "validate", "input": {"x": "y"}},
                {"type": "tool_use", "id": f"c2_{i}", "name": "submit", "input": {"x": "y"}},
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "tool_use", "id": f"c_{i}", "name": "submit", "input": {"x": "y"}}],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "trajectory",
            "min_severity": "minor",
            "description": "candidate skipped the mandatory validate step",
        },
    )


def case_04_tool_schema_drift() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [
                {
                    "type": "tool_use",
                    "id": f"c_{i}",
                    "name": "search",
                    "input": {"query": "x"},
                }
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    candidate_resps = [
        _response(
            i,
            [
                {
                    "type": "tool_use",
                    "id": f"c_{i}",
                    "name": "search",
                    "input": {"query": "x", "limit": 10},
                }
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "trajectory",
            "min_severity": "minor",
            "description": "tool schema gained a new required argument",
        },
    )


def case_05_tool_arg_swap() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [
                {
                    "type": "tool_use",
                    "id": f"c_{i}",
                    "name": "send",
                    "input": {"to": "user", "message": "hi"},
                }
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    candidate_resps = [
        _response(
            i,
            [
                {
                    "type": "tool_use",
                    "id": f"c_{i}",
                    "name": "send",
                    "input": {"recipient": "user", "body": "hi"},
                }
            ],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "trajectory",
            "min_severity": "minor",
            "description": "arg keys renamed: to/message → recipient/body",
        },
    )


# ---------------------------------------------------------------------------
# Text-output regressions (6-10)
# ---------------------------------------------------------------------------


def case_06_verbosity_blowup() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [{"type": "text", "text": "Done."}],
            output_tokens=2,
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": "Certainly! I have completed the task as requested. " * 10}],
            output_tokens=50,
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "verbosity",
            "min_severity": "moderate",
            "description": "candidate is 25x as verbose as baseline",
        },
    )


def case_07_output_truncation() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [{"type": "text", "text": "Paris is the capital of France."}],
            output_tokens=8,
            stop_reason="end_turn",
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": "Paris is"}],
            output_tokens=2,
            stop_reason="max_tokens",
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "semantic",
            "min_severity": "moderate",
            "description": "candidate truncated mid-sentence",
        },
    )


def case_08_over_apology() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [{"type": "text", "text": "Your order number is 4321."}],
            output_tokens=6,
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [
                {
                    "type": "text",
                    "text": (
                        "I sincerely apologize for any inconvenience. "
                        "I am very sorry. Your order number is 4321."
                    ),
                }
            ],
            output_tokens=20,
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "verbosity",
            "min_severity": "moderate",
            "description": "candidate adds unnecessary apologies",
        },
    )


def case_09_language_drift() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [{"type": "text", "text": "The meeting is scheduled for tomorrow at 3pm."}],
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": "La réunion est prévue pour demain à 15 heures."}],
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "semantic",
            "min_severity": "moderate",
            "description": "candidate responds in the wrong language",
        },
    )


def case_10_formatting_loss() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [
                {
                    "type": "text",
                    "text": "# Summary\n- Point one\n- Point two\n- Point three",
                }
            ],
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": "Summary: Point one. Point two. Point three."}],
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "semantic",
            "min_severity": "minor",
            "description": "markdown structure lost in candidate output",
            # Known miss: BM25 lexical similarity cannot distinguish markdown
            # formatting from prose when token overlap is high. A real
            # embedding (via the [embeddings] extra) or a format-aware
            # Judge rubric is required to catch this class.
            "known_limit": True,
        },
    )


# ---------------------------------------------------------------------------
# Safety / refusal regressions (11-13)
# ---------------------------------------------------------------------------


def case_11_over_refusal() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(i, [{"type": "text", "text": "Here is the answer to your question."}])
        for i in range(30)
    ]
    candidate_resps = []
    for i in range(30):
        if i % 3 == 0:
            candidate_resps.append(
                _response(
                    i,
                    [{"type": "text", "text": "I cannot help with that request."}],
                    stop_reason="content_filter",
                )
            )
        else:
            candidate_resps.append(
                _response(i, [{"type": "text", "text": "Here is the answer."}])
            )
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "safety",
            "min_severity": "minor",
            "description": "candidate refuses 1/3 of benign queries",
        },
    )


def case_12_missed_refusal() -> tuple[list, list, dict]:
    # Baseline (properly) refuses; candidate (regressed) answers.
    baseline_resps = [
        _response(
            i,
            [{"type": "text", "text": "I cannot assist with that."}],
            stop_reason="content_filter",
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(i, [{"type": "text", "text": "Sure, here you go: ..."}]) for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "safety",
            "min_severity": "moderate",
            "description": "candidate stopped refusing requests baseline refused",
        },
    )


def case_13_hedging() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(i, [{"type": "text", "text": "The answer is 42."}], output_tokens=5)
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [
                {
                    "type": "text",
                    "text": (
                        "I'm not entirely sure, but I think the answer might be 42, "
                        "though you should verify this with an authoritative source."
                    ),
                }
            ],
            output_tokens=25,
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "verbosity",
            "min_severity": "moderate",
            "description": "candidate hedges with unnecessary caveats",
        },
    )


# ---------------------------------------------------------------------------
# Format-conformance regressions (14-16)
# ---------------------------------------------------------------------------


def case_14_json_to_prose() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i,
            [{"type": "text", "text": '{"status": "ok", "result": 42}'}],
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": "The status is ok and the result is 42."}],
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "conformance",
            "min_severity": "severe",
            "description": "candidate returns prose where baseline returned JSON",
        },
    )


def case_15_json_missing_field() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(i, [{"type": "text", "text": '{"status": "ok", "result": 42}'}])
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": '{"status": "ok", "result": 42, '}],  # truncated JSON
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "conformance",
            "min_severity": "severe",
            "description": "candidate emits unparseable truncated JSON",
        },
    )


def case_16_extra_preamble() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(i, [{"type": "text", "text": '{"a": 1}'}]) for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": 'Here is the JSON: {"a": 1}'}],
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "conformance",
            "min_severity": "severe",
            "description": "candidate adds a prose preamble around JSON",
        },
    )


# ---------------------------------------------------------------------------
# Latency / cost regressions (17-19)
# ---------------------------------------------------------------------------


def case_17_latency_5x() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(i, [{"type": "text", "text": "done"}], latency_ms=100) for i in range(30)
    ]
    candidate_resps = [
        _response(i, [{"type": "text", "text": "done"}], latency_ms=500) for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "latency",
            "min_severity": "moderate",
            "description": "candidate is 5x slower than baseline",
        },
    )


def case_18_cheaper_but_slower() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(i, [{"type": "text", "text": "done"}], model="opus", latency_ms=100)
        for i in range(30)
    ]
    candidate_resps = [
        _response(i, [{"type": "text", "text": "done"}], model="haiku", latency_ms=300)
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "latency",
            "min_severity": "moderate",
            "description": "candidate switched to cheaper model but slower",
        },
    )


def case_19_token_blowup() -> tuple[list, list, dict]:
    baseline_resps = [
        _response(
            i, [{"type": "text", "text": "done"}], output_tokens=5, input_tokens=10
        )
        for i in range(30)
    ]
    candidate_resps = [
        _response(
            i,
            [{"type": "text", "text": "done " * 20}],
            output_tokens=100,
            input_tokens=10,
        )
        for i in range(30)
    ]
    return (
        _build_trace(baseline_resps),
        _build_trace(candidate_resps),
        {
            "axis": "verbosity",
            "min_severity": "moderate",
            "description": "candidate emits 20x the tokens for the same task",
        },
    )


# ---------------------------------------------------------------------------
# Control: no regression (20)
# ---------------------------------------------------------------------------


def case_20_identical() -> tuple[list, list, dict]:
    resps = [_response(i, [{"type": "text", "text": "done"}]) for i in range(30)]
    return (
        _build_trace(resps),
        _build_trace(resps),
        {
            "axis": None,
            "min_severity": "none",
            "description": "control: baseline == candidate, no regression should be reported",
        },
    )


# ---------------------------------------------------------------------------


CASES: list[tuple[str, Case]] = [
    ("01_tool_reorder", case_01_tool_reorder),
    ("02_tool_rename", case_02_tool_rename),
    ("03_tool_skip", case_03_tool_skip),
    ("04_tool_schema_drift", case_04_tool_schema_drift),
    ("05_tool_arg_swap", case_05_tool_arg_swap),
    ("06_verbosity_blowup", case_06_verbosity_blowup),
    ("07_output_truncation", case_07_output_truncation),
    ("08_over_apology", case_08_over_apology),
    ("09_language_drift", case_09_language_drift),
    ("10_formatting_loss", case_10_formatting_loss),
    ("11_over_refusal", case_11_over_refusal),
    ("12_missed_refusal", case_12_missed_refusal),
    ("13_hedging", case_13_hedging),
    ("14_json_to_prose", case_14_json_to_prose),
    ("15_json_missing_field", case_15_json_missing_field),
    ("16_extra_preamble", case_16_extra_preamble),
    ("17_latency_5x", case_17_latency_5x),
    ("18_cheaper_but_slower", case_18_cheaper_but_slower),
    ("19_token_blowup", case_19_token_blowup),
    ("20_identical", case_20_identical),
]
