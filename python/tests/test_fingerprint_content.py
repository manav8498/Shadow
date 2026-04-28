"""Headline tests for the v2.7 content-aware fingerprint dimensions.

The pre-v2.7 D=8 fingerprint was blind to "agent gives wrong answer
with the same tool-call pattern" — Hotelling T² on D=8 returned no
shift on this regression class. The v2.7 expansion adds four
content-aware dimensions (text_chars_log, arg_keys_total_log,
error_token_flag, numeric_token_density) so the joint test
statistic moves on the same fixture.

Each test below builds two traces that share tool-call patterns but
differ in content, runs Hotelling T² (with permutation p-value to
sidestep the small-n F-approximation degeneracy), and asserts the
test rejects the null at the 0.05 significance level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shadow.statistical.fingerprint import (
    DEFAULT_CONFIG,
    DIM,
    BehavioralVector,
    FingerprintConfig,
    fingerprint_trace,
)
from shadow.statistical.hotelling import hotelling_t2


def _record(idx: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{idx:064x}",
        "kind": "chat_response",
        "ts": "2026-04-28T00:00:00.000Z",
        "parent": "sha256:" + "0" * 64,
        "meta": {},
        "payload": payload,
    }


def _response(
    idx: int,
    *,
    text: str,
    tools: list[str] | None = None,
    stop: str = "end_turn",
    output_tokens: int = 100,
    latency_ms: float = 600.0,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for j, name in enumerate(tools or []):
        content.append({"type": "tool_use", "id": f"t{idx}_{j}", "name": name, "input": {}})
    return _record(
        idx,
        {
            "model": "x",
            "content": content,
            "stop_reason": stop,
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": 200,
                "output_tokens": output_tokens,
                "thinking_tokens": 0,
            },
        },
    )


class TestDimensionExpansion:
    def test_dim_is_twelve(self) -> None:
        assert DIM == 12

    def test_behavioural_vector_carries_all_twelve_features(self) -> None:
        rec = _response(1, text="hello world", tools=["lookup"])
        mat = fingerprint_trace([rec])
        assert mat.shape == (1, DIM)
        # The new dimensions are positions 8..11.
        # Features 8 (text_chars_log) and 11 (numeric_token_density) are
        # bounded in [0, 1]; we sanity-check the range.
        assert 0.0 <= float(mat[0, 8]) <= 1.0  # text_chars_log
        assert 0.0 <= float(mat[0, 9]) <= 1.0  # arg_keys_total_log
        assert mat[0, 10] in (0.0, 1.0)  # error_token_flag is binary
        assert 0.0 <= float(mat[0, 11]) <= 1.0  # numeric_token_density


class TestErrorTokenFlag:
    def test_error_text_sets_flag(self) -> None:
        rec = _response(1, text="The lookup returned an error: not found.")
        mat = fingerprint_trace([rec])
        assert mat[0, 10] == 1.0

    def test_clean_text_clears_flag(self) -> None:
        rec = _response(1, text="The lookup returned three matching results.")
        mat = fingerprint_trace([rec])
        assert mat[0, 10] == 0.0

    def test_case_insensitive(self) -> None:
        rec = _response(1, text="UNABLE TO PROCESS REQUEST")
        mat = fingerprint_trace([rec])
        assert mat[0, 10] == 1.0


class TestNumericTokenDensity:
    def test_numeric_heavy_text(self) -> None:
        rec = _response(1, text="The values are 12 47 88 99 102 with mean 69.6")
        mat = fingerprint_trace([rec])
        assert mat[0, 11] > 0.4

    def test_purely_textual_response(self) -> None:
        rec = _response(1, text="please consult a clinician for personalised advice")
        mat = fingerprint_trace([rec])
        assert mat[0, 11] == 0.0

    def test_currency_symbols_stripped(self) -> None:
        rec = _response(1, text="Total: $123.45 from 2 charges")
        mat = fingerprint_trace([rec])
        # 4 tokens total ("Total:", "$123.45", "from", "2", "charges") — wait that's 5.
        # tokens: ["Total:", "$123.45", "from", "2", "charges"] → 5 tokens
        # numeric after strip: "Total" (no), "123.45" (yes), "from" (no), "2" (yes), "charges" (no)
        assert mat[0, 11] == 0.4


class TestTextCharsAndArgComplexity:
    def test_long_text_pushes_text_chars(self) -> None:
        long_text = "word " * 5000  # 25000 chars
        rec = _response(1, text=long_text)
        mat = fingerprint_trace([rec], DEFAULT_CONFIG)
        assert mat[0, 8] > 0.9

    def test_short_text_keeps_text_chars_low(self) -> None:
        rec = _response(1, text="ok")
        mat = fingerprint_trace([rec])
        assert mat[0, 8] < 0.2

    def test_arg_keys_count_across_tools(self) -> None:
        # Custom tool block with rich input.
        rec = _record(
            1,
            {
                "model": "x",
                "content": [
                    {
                        "type": "text",
                        "text": "executing",
                    },
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "execute",
                        "input": {"q": "x", "limit": 10, "offset": 0, "sort": "asc"},
                    },
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "log",
                        "input": {"level": "info", "msg": "ok"},
                    },
                ],
                "stop_reason": "tool_use",
                "latency_ms": 100,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
        mat = fingerprint_trace([rec])
        # 4 + 2 = 6 keys total
        # arg_keys_scale=32 default; log(7)/log(33) ≈ 0.557
        assert mat[0, 9] > 0.4
        assert mat[0, 9] < 0.7


class TestContentDriftDetection:
    """Headline integration: same tool patterns / latency / output_tokens
    but content shift → Hotelling T² rejects the null.

    Pre-v2.7 (D=8) the fingerprints would have been near-identical and
    the test would not have rejected. Post-v2.7 (D=12) the new content
    dimensions move and the test correctly detects the drift.
    """

    def test_error_messages_emerge_with_same_tool_pattern(self) -> None:
        # Baseline: clean responses, no error tokens.
        # Candidate: same tool pattern, but every response now contains an
        # error message. D=8 misses this; D=12 catches it via
        # error_token_flag + lower numeric density + shorter text.
        baseline = [
            _response(
                idx=i,
                text=f"Result {i + 1}: 42 successful lookups returned across 3 partitions.",
                tools=["search"],
            )
            for i in range(20)
        ]
        candidate = [
            _response(
                idx=100 + i,
                text="Sorry, I cannot complete that request.",
                tools=["search"],
            )
            for i in range(20)
        ]
        x1 = fingerprint_trace(baseline)
        x2 = fingerprint_trace(candidate)
        result = hotelling_t2(x1, x2, alpha=0.05, permutations=500, rng=np.random.default_rng(7))
        assert result.reject_null, (
            f"Hotelling T² should reject H0 on content-drifted fixtures with "
            f"same tool patterns; got p={result.p_value:.4f}"
        )

    def test_numeric_response_to_textual_response_shift(self) -> None:
        # Baseline: agent returns numeric tabular output.
        # Candidate: agent returns prose. Tools and latency identical.
        baseline = [
            _response(
                idx=i,
                text="| col_a | col_b | col_c |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |",
                tools=["query"],
            )
            for i in range(20)
        ]
        candidate = [
            _response(
                idx=200 + i,
                text="The query returned several rows of data; please review.",
                tools=["query"],
            )
            for i in range(20)
        ]
        x1 = fingerprint_trace(baseline)
        x2 = fingerprint_trace(candidate)
        result = hotelling_t2(x1, x2, alpha=0.05, permutations=500, rng=np.random.default_rng(11))
        assert result.reject_null, (
            f"Hotelling T² should reject H0 when numeric-density shifts; "
            f"got p={result.p_value:.4f}"
        )

    def test_no_content_drift_does_not_reject(self) -> None:
        # Baseline-vs-baseline (same content distribution) — Hotelling
        # should NOT reject. Verifies the new dimensions don't induce
        # spurious rejections.
        baseline_a = [
            _response(idx=i, text=f"Found {i} matches.", tools=["search"]) for i in range(20)
        ]
        baseline_b = [
            _response(idx=300 + i, text=f"Found {i + 1} matches.", tools=["search"])
            for i in range(20)
        ]
        x1 = fingerprint_trace(baseline_a)
        x2 = fingerprint_trace(baseline_b)
        result = hotelling_t2(x1, x2, alpha=0.05, permutations=500, rng=np.random.default_rng(13))
        assert not result.reject_null, (
            f"Hotelling T² should NOT reject H0 on baseline-vs-baseline; "
            f"got p={result.p_value:.4f}"
        )


class TestConfigCustomisation:
    def test_custom_char_scale(self) -> None:
        # Under default char_scale=16384, a 1000-char response yields
        # log(1001)/log(16385) ≈ 0.71 — not saturated. Under
        # char_scale=500 the same response saturates to 1.0.
        rec = _response(1, text="x" * 1000)
        default_mat = fingerprint_trace([rec], DEFAULT_CONFIG)
        tight_cfg = FingerprintConfig(char_scale=500)
        tight_mat = fingerprint_trace([rec], tight_cfg)
        assert default_mat[0, 8] < 0.85  # not saturated
        assert tight_mat[0, 8] == 1.0  # saturated under tighter scale
        assert tight_mat[0, 8] > default_mat[0, 8]  # tighter scale → larger value


class TestVectorRoundTrip:
    def test_vector_dataclass_has_all_fields(self) -> None:
        v = BehavioralVector(
            tool_call_rate=0.1,
            distinct_tool_frac=0.5,
            stop_end_turn=1.0,
            stop_tool_use=0.0,
            stop_other=0.0,
            output_len_log=0.3,
            latency_log=0.2,
            refusal_flag=0.0,
            text_chars_log=0.4,
            arg_keys_total_log=0.6,
            error_token_flag=1.0,
            numeric_token_density=0.25,
        )
        arr = v.to_array()
        assert arr.shape == (DIM,)
        # Last four positions are content-aware dims.
        assert arr[8] == 0.4
        assert arr[9] == 0.6
        assert arr[10] == 1.0
        assert arr[11] == 0.25
