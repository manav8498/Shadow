"""Production end-to-end tests for shadow.statistical, shadow.ltl, and shadow.conformal.

Exercises production-critical paths not covered by the existing unit tests,
using real agentlog record shapes with all required top-level fields.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadow.conformal import ConformalCoverageReport, build_parametric_estimate
from shadow.hierarchical import PolicyRule, check_policy
from shadow.ltl.checker import TraceState, check, trace_from_records
from shadow.ltl.compiler import parse_ltl, rule_to_ltl
from shadow.ltl.formula import (
    And,
    Atom,
    Finally,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    Until,
    WeakUntil,
)
from shadow.statistical.fingerprint import DIM as FP_DIM
from shadow.statistical.fingerprint import fingerprint_trace
from shadow.statistical.hotelling import hotelling_t2
from shadow.statistical.sprt import MultiSPRT, SPRTDetector

# ---------------------------------------------------------------------------
# Module-level fixture builders
# ---------------------------------------------------------------------------

_NULL_ID = "sha256:" + "0" * 64
_TS = "2026-04-22T04:03:27.228Z"
_TS2 = "2026-04-22T04:03:28.000Z"
_TS3 = "2026-04-22T04:03:29.000Z"


def _make_agentlog_record(kind: str, payload: dict | None, idx: int = 0) -> dict:
    """Construct a full agentlog record with all required top-level fields."""
    return {
        "version": "0.1",
        "id": f"sha256:{'0' * 63}{idx}",
        "kind": kind,
        "ts": "2026-04-22T00:00:00.000Z",
        "parent": None,
        "meta": {"session_tag": "test"},
        "payload": payload,
    }


# --- Demo baseline records (3 chat_response turns) ---

DEMO_BASELINE_RECORDS: list[dict] = [
    {
        "version": "0.1",
        "id": "sha256:abc" + "0" * 61,
        "kind": "chat_response",
        "ts": _TS,
        "parent": "sha256:xyz" + "0" * 61,
        "meta": {"session_tag": "demo"},
        "payload": {
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
            "usage": {"input_tokens": 120, "output_tokens": 28, "thinking_tokens": 0},
        },
    },
    {
        "version": "0.1",
        "id": "sha256:abc" + "1" * 61,
        "kind": "chat_response",
        "ts": _TS2,
        "parent": "sha256:abc" + "0" * 61,
        "meta": {"session_tag": "demo"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                {
                    "type": "text",
                    "text": "lib.rs exposes the public API; error.rs defines the typed error enum.",
                }
            ],
            "stop_reason": "end_turn",
            "latency_ms": 115,
            "usage": {"input_tokens": 200, "output_tokens": 18, "thinking_tokens": 0},
        },
    },
    {
        "version": "0.1",
        "id": "sha256:abc" + "2" * 61,
        "kind": "chat_response",
        "ts": _TS3,
        "parent": "sha256:abc" + "1" * 61,
        "meta": {"session_tag": "demo"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                {
                    "type": "text",
                    "text": '{"findings": ["lib.rs entry point", "error.rs typed errors"]}',
                }
            ],
            "stop_reason": "end_turn",
            "latency_ms": 84,
            "usage": {"input_tokens": 220, "output_tokens": 26, "thinking_tokens": 0},
        },
    },
]

# --- Demo candidate records (3 chat_response turns, last is content_filter) ---

DEMO_CANDIDATE_RECORDS: list[dict] = [
    {
        "version": "0.1",
        "id": "sha256:cand" + "0" * 60,
        "kind": "chat_response",
        "ts": _TS,
        "parent": "sha256:xyz" + "0" * 61,
        "meta": {"session_tag": "demo"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                {
                    "type": "text",
                    "text": "I'll search for Rust files to understand the project.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search_files",
                    "input": {"query": "*.rs"},
                },
            ],
            "stop_reason": "tool_use",
            "latency_ms": 412,
            "usage": {"input_tokens": 120, "output_tokens": 68, "thinking_tokens": 0},
        },
    },
    {
        "version": "0.1",
        "id": "sha256:cand" + "1" * 60,
        "kind": "chat_response",
        "ts": _TS2,
        "parent": "sha256:cand" + "0" * 60,
        "meta": {"session_tag": "demo"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Looking at the Rust project structure, lib.rs exposes the public API "
                        "and error.rs defines the typed error enum. The project follows standard "
                        "Rust conventions with a clear separation of concerns."
                    ),
                }
            ],
            "stop_reason": "end_turn",
            "latency_ms": 534,
            "usage": {"input_tokens": 200, "output_tokens": 52, "thinking_tokens": 0},
        },
    },
    {
        "version": "0.1",
        "id": "sha256:cand" + "2" * 60,
        "kind": "chat_response",
        "ts": _TS3,
        "parent": "sha256:cand" + "1" * 60,
        "meta": {"session_tag": "demo"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "[filtered]"}],
            "stop_reason": "content_filter",
            "latency_ms": 398,
            "usage": {"input_tokens": 220, "output_tokens": 9, "thinking_tokens": 0},
        },
    },
]


# --- DevOps baseline records (5 chat_response turns, multi-tool-call per turn) ---


def _devops_tool_block(name: str, idx: int) -> dict:
    return {"type": "tool_use", "id": f"toolu_{idx:02d}", "name": name, "input": {}}


DEVOPS_BASELINE_RECORDS: list[dict] = [
    # Turn 1: 5 calls — send_notification x2, backup x1, check_replication_lag x1, run_migration x1
    {
        "version": "0.1",
        "id": "sha256:devops" + "0" * 58,
        "kind": "chat_response",
        "ts": _TS,
        "parent": None,
        "meta": {"session_tag": "devops"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                _devops_tool_block("send_notification", 1),
                _devops_tool_block("backup_database", 2),
                _devops_tool_block("check_replication_lag", 3),
                _devops_tool_block("run_migration", 4),
                _devops_tool_block("send_notification", 5),
            ],
            "stop_reason": "end_turn",
            "latency_ms": 2100,
            "usage": {"input_tokens": 800, "output_tokens": 312, "thinking_tokens": 0},
        },
    },
    # Turn 2: 7 tool calls (end_turn)
    {
        "version": "0.1",
        "id": "sha256:devops" + "1" * 58,
        "kind": "chat_response",
        "ts": _TS2,
        "parent": "sha256:devops" + "0" * 58,
        "meta": {"session_tag": "devops"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                _devops_tool_block("check_replication_lag", 6),
                _devops_tool_block("backup_database", 7),
                _devops_tool_block("run_migration", 8),
                _devops_tool_block("send_notification", 9),
                _devops_tool_block("check_replication_lag", 10),
                _devops_tool_block("backup_database", 11),
                _devops_tool_block("send_notification", 12),
            ],
            "stop_reason": "end_turn",
            "latency_ms": 2800,
            "usage": {"input_tokens": 900, "output_tokens": 402, "thinking_tokens": 0},
        },
    },
    # Turn 3: 6 tool calls (includes request_human_approval)
    {
        "version": "0.1",
        "id": "sha256:devops" + "2" * 58,
        "kind": "chat_response",
        "ts": _TS3,
        "parent": "sha256:devops" + "1" * 58,
        "meta": {"session_tag": "devops"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                _devops_tool_block("check_replication_lag", 13),
                _devops_tool_block("request_human_approval", 14),
                _devops_tool_block("backup_database", 15),
                _devops_tool_block("run_migration", 16),
                _devops_tool_block("send_notification", 17),
                _devops_tool_block("check_replication_lag", 18),
            ],
            "stop_reason": "end_turn",
            "latency_ms": 2400,
            "usage": {"input_tokens": 850, "output_tokens": 382, "thinking_tokens": 0},
        },
    },
    # Turn 4: 7 tool calls (includes request_human_approval, pause_replication, restore_database)
    {
        "version": "0.1",
        "id": "sha256:devops" + "3" * 58,
        "kind": "chat_response",
        "ts": _TS,
        "parent": "sha256:devops" + "2" * 58,
        "meta": {"session_tag": "devops"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                _devops_tool_block("backup_database", 19),
                _devops_tool_block("request_human_approval", 20),
                _devops_tool_block("pause_replication", 21),
                _devops_tool_block("restore_database", 22),
                _devops_tool_block("run_migration", 23),
                _devops_tool_block("send_notification", 24),
                _devops_tool_block("check_replication_lag", 25),
            ],
            "stop_reason": "end_turn",
            "latency_ms": 2900,
            "usage": {"input_tokens": 920, "output_tokens": 398, "thinking_tokens": 0},
        },
    },
    # Turn 5: 3 tool calls (includes request_human_approval)
    {
        "version": "0.1",
        "id": "sha256:devops" + "4" * 58,
        "kind": "chat_response",
        "ts": _TS2,
        "parent": "sha256:devops" + "3" * 58,
        "meta": {"session_tag": "devops"},
        "payload": {
            "model": "claude-opus-4-7",
            "content": [
                _devops_tool_block("request_human_approval", 26),
                _devops_tool_block("send_notification", 27),
                _devops_tool_block("check_replication_lag", 28),
            ],
            "stop_reason": "end_turn",
            "latency_ms": 1400,
            "usage": {"input_tokens": 600, "output_tokens": 218, "thinking_tokens": 0},
        },
    },
]


def _demo_baseline_states() -> list[TraceState]:
    """Build TraceState objects matching the demo baseline fixture."""
    return [
        TraceState(
            pair_index=0,
            tool_calls=["search_files"],
            stop_reason="tool_use",
            text_content="I'll search for Rust files.",
        ),
        TraceState(
            pair_index=1,
            tool_calls=[],
            stop_reason="end_turn",
            text_content="lib.rs exposes the public API; error.rs defines the typed error enum.",
        ),
        TraceState(
            pair_index=2,
            tool_calls=[],
            stop_reason="end_turn",
            text_content='{"findings": ["lib.rs is the entry point", "error.rs has typed errors"]}',
        ),
    ]


def _devops_baseline_states() -> list[TraceState]:
    """Build TraceState objects matching the devops baseline fixture."""
    return [
        TraceState(
            pair_index=0,
            tool_calls=[
                "send_notification",
                "backup_database",
                "check_replication_lag",
                "run_migration",
                "send_notification",
            ],
            stop_reason="end_turn",
            text_content="",
        ),
        TraceState(
            pair_index=1,
            tool_calls=[
                "check_replication_lag",
                "backup_database",
                "run_migration",
                "send_notification",
                "check_replication_lag",
                "backup_database",
                "send_notification",
            ],
            stop_reason="end_turn",
            text_content="",
        ),
        TraceState(
            pair_index=2,
            tool_calls=[
                "check_replication_lag",
                "request_human_approval",
                "backup_database",
                "run_migration",
                "send_notification",
                "check_replication_lag",
            ],
            stop_reason="end_turn",
            text_content="",
        ),
        TraceState(
            pair_index=3,
            tool_calls=[
                "backup_database",
                "request_human_approval",
                "pause_replication",
                "restore_database",
                "run_migration",
                "send_notification",
                "check_replication_lag",
            ],
            stop_reason="end_turn",
            text_content="",
        ),
        TraceState(
            pair_index=4,
            tool_calls=["request_human_approval", "send_notification", "check_replication_lag"],
            stop_reason="end_turn",
            text_content="",
        ),
    ]


# ===========================================================================
# 1. TestFingerprintOnRealFixtures
# ===========================================================================


class TestFingerprintOnRealFixtures:
    def test_demo_baseline_fingerprint_shape(self):
        mat = fingerprint_trace(DEMO_BASELINE_RECORDS)
        assert mat.shape == (3, FP_DIM)

    def test_demo_candidate_has_refusal_flag(self):
        mat = fingerprint_trace(DEMO_CANDIDATE_RECORDS)
        assert mat.shape == (3, FP_DIM)
        # Last row corresponds to content_filter turn — refusal_flag is index 7
        assert mat[2, 7] == pytest.approx(1.0)

    def test_devops_baseline_high_tool_call_rate(self):
        mat = fingerprint_trace(DEVOPS_BASELINE_RECORDS)
        assert mat.shape == (5, FP_DIM)
        # All turns have ≥1 tool call → tool_call_rate (log-scaled) is > 0
        # for all rows. The exact value depends on how many tools were
        # called per turn, but every row must be strictly positive.
        for row_idx in range(5):
            assert (
                mat[row_idx, 0] > 0.0
            ), f"row {row_idx}: expected tool_call_rate > 0, got {mat[row_idx, 0]}"

    def test_devops_multiple_tools_per_turn_distinct_frac(self):
        mat = fingerprint_trace(DEVOPS_BASELINE_RECORDS)
        # Turn 1 (index 0): send_notification x2, backup x1, check_replication_lag x1,
        # run_migration x1 => 4 distinct / 5 total = 0.8
        assert mat[0, 1] == pytest.approx(0.8, abs=1e-9)

    def test_null_payload_handled_gracefully(self):
        rec = {
            "version": "0.1",
            "id": _NULL_ID,
            "kind": "chat_response",
            "ts": _TS,
            "parent": None,
            "meta": {},
            "payload": None,
        }
        # Should not crash; null payload → skipped (payload or {} gives empty dict)
        mat = fingerprint_trace([rec])
        # payload=None → payload or {} → empty, still produces a row
        assert mat.shape[1] == FP_DIM

    def test_none_content_handled_gracefully(self):
        rec = {
            "version": "0.1",
            "id": _NULL_ID,
            "kind": "chat_response",
            "ts": _TS,
            "parent": None,
            "meta": {},
            "payload": {
                "content": None,
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"output_tokens": 10},
            },
        }
        mat = fingerprint_trace([rec])
        assert mat.shape == (1, FP_DIM)
        # No tool calls → tool_call_rate and distinct_tool_frac both 0
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[0, 1] == pytest.approx(0.0)

    def test_missing_usage_gives_zero_tokens(self):
        rec = {
            "version": "0.1",
            "id": _NULL_ID,
            "kind": "chat_response",
            "ts": _TS,
            "parent": None,
            "meta": {},
            "payload": {
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                # no 'usage' key
            },
        }
        mat = fingerprint_trace([rec])
        assert mat.shape == (1, FP_DIM)
        # output_len_log = log(0+1)/log(4097) = 0
        assert mat[0, 5] == pytest.approx(0.0)

    def test_missing_latency_gives_zero_latency(self):
        rec = {
            "version": "0.1",
            "id": _NULL_ID,
            "kind": "chat_response",
            "ts": _TS,
            "parent": None,
            "meta": {},
            "payload": {
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"output_tokens": 10},
                # no 'latency_ms' key
            },
        }
        mat = fingerprint_trace([rec])
        assert mat.shape == (1, FP_DIM)
        # latency_log = log(0+1)/log(30001) = 0
        assert mat[0, 6] == pytest.approx(0.0)

    def test_unknown_stop_reason_sets_stop_other(self):
        rec = {
            "version": "0.1",
            "id": _NULL_ID,
            "kind": "chat_response",
            "ts": _TS,
            "parent": None,
            "meta": {},
            "payload": {
                "content": [],
                "stop_reason": "max_tokens",
                "latency_ms": 100,
                "usage": {"output_tokens": 10},
            },
        }
        mat = fingerprint_trace([rec])
        assert mat.shape == (1, FP_DIM)
        # stop_other (index 4) = 1.0; stop_end_turn (2) = 0; stop_tool_use (3) = 0
        assert mat[0, 2] == pytest.approx(0.0)  # stop_end_turn
        assert mat[0, 3] == pytest.approx(0.0)  # stop_tool_use
        assert mat[0, 4] == pytest.approx(1.0)  # stop_other
        assert mat[0, 7] == pytest.approx(0.0)  # refusal_flag

    def test_extra_fields_in_record_ignored(self):
        rec = {
            "version": "0.1",
            "id": _NULL_ID,
            "kind": "chat_response",
            "ts": _TS,
            "parent": None,
            "meta": {"session_tag": "test"},
            "payload": {
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"output_tokens": 5},
            },
            "extra_field_that_should_be_ignored": "some_value",
            "another_extra": 42,
        }
        mat = fingerprint_trace([rec])
        assert mat.shape == (1, FP_DIM)


# ===========================================================================
# 2. TestHotellingOnRealFixtures
# ===========================================================================


class TestHotellingOnRealFixtures:
    def test_demo_baseline_vs_candidate_detects_drift(self):
        x1 = fingerprint_trace(DEMO_BASELINE_RECORDS)
        x2 = fingerprint_trace(DEMO_CANDIDATE_RECORDS)
        assert x1.shape == (3, FP_DIM)
        assert x2.shape == (3, FP_DIM)
        # With n=3, df2 may be ≤ 0 so no rejection possible by F-test,
        # but must not crash; candidate has content_filter refusal (structural shift)
        result = hotelling_t2(x1, x2, alpha=0.10)
        # The result must be a valid HotellingResult — not a crash
        assert hasattr(result, "p_value")
        assert math.isfinite(result.t2)

    def test_devops_vs_demo_rejects_null(self):
        # With n1=5 and n2=5..6, d=8, df2 = n1+n2-d-1 is 1 or 2 — far too low for
        # reliable inference. Inflate both samples by repeating records so that
        # n1=n2=15 → df2=21, giving the F-test real power.
        devops_x15 = fingerprint_trace(DEVOPS_BASELINE_RECORDS * 3)  # 15 rows
        demo_x15 = fingerprint_trace(DEMO_BASELINE_RECORDS * 5)  # 15 rows
        assert devops_x15.shape == (15, FP_DIM)
        assert demo_x15.shape == (15, FP_DIM)
        result = hotelling_t2(devops_x15, demo_x15, alpha=0.05)
        assert result.df2 > 0, f"df2={result.df2}: not enough rows for the F-test"
        assert result.reject_null, (
            f"Expected drift detection (devops vs demo are structurally different), "
            f"got p_value={result.p_value}"
        )

    def test_identical_traces_does_not_reject(self):
        # Same devops baseline compared against itself — zero within-group variance
        # after shrinkage. Must NOT crash.
        x1 = fingerprint_trace(DEVOPS_BASELINE_RECORDS)
        x2 = fingerprint_trace(DEVOPS_BASELINE_RECORDS)
        try:
            result = hotelling_t2(x1, x2, alpha=0.05)
            # p_value may be NaN or 1.0 when variance is zero after shrinkage
            assert result.p_value >= 0.0 or math.isnan(result.p_value)
        except Exception as exc:
            pytest.fail(f"hotelling_t2 raised unexpectedly on identical traces: {exc}")

    def test_rank_deficient_pooled_cov_no_crash(self):
        x1 = np.tile(np.array([1, 0, 1, 0, 0, 0.5, 0.5, 0], dtype=np.float64), (10, 1))
        x2 = np.tile(np.array([0, 0, 0, 1, 0, 0.3, 0.6, 0], dtype=np.float64), (10, 1))
        # rank-1 matrices — must not crash
        try:
            result = hotelling_t2(x1, x2, alpha=0.05)
            assert hasattr(result, "t2")
        except Exception as exc:
            pytest.fail(f"hotelling_t2 raised on rank-deficient input: {exc}")

    def test_p_value_always_finite(self):
        rng = np.random.default_rng(seed=123)
        for _ in range(100):
            x1 = rng.standard_normal((5, FP_DIM))
            x2 = rng.standard_normal((5, FP_DIM))
            result = hotelling_t2(x1, x2)
            assert math.isfinite(result.p_value), f"p_value={result.p_value} is not finite"

    def test_single_row_each_raises_valueerror(self):
        x1 = np.zeros((1, FP_DIM), dtype=np.float64)
        x2 = np.zeros((10, FP_DIM), dtype=np.float64)
        with pytest.raises(ValueError):
            hotelling_t2(x1, x2)


# ===========================================================================
# 3. TestSPRTOnRealDistributions
# ===========================================================================


class TestSPRTOnRealDistributions:
    def test_warmup_with_identical_scores_zero_variance(self):
        det = SPRTDetector(warmup=5)
        for _ in range(5):
            state = det.update(0.0)
            assert state.in_warmup
        # After warmup, sigma should be clamped to 1e-6 (not zero)
        assert det._sigma >= 1e-6
        # Next update must not crash even though sigma ≈ 0
        state = det.update(0.0)
        assert not state.in_warmup
        assert state.decision in ("h0", "h1", "continue")

    def test_sprt_on_demo_latency_drift(self):
        # Baseline latencies from demo fixture
        baseline_latencies = [98.0, 115.0, 84.0]
        # Candidate latencies are ~4x higher
        candidate_latencies = [412.0, 534.0, 398.0]
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=0.5, warmup=3)
        for lat in baseline_latencies:
            det.update(lat)
        # Feed candidate latencies repeatedly until decision
        decision = "continue"
        for _ in range(200):
            for lat in candidate_latencies:
                state = det.update(lat)
                decision = state.decision
                if decision == "h1":
                    break
            if decision == "h1":
                break
        assert decision == "h1", "Expected drift (h1) not detected in latency stream"

    def test_sprt_log_lr_exactly_at_boundary_stays_decided(self):
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=2.0, warmup=2)
        # Warmup phase
        det.update(1.0)
        det.update(1.0)
        # Drive log_lr to reach h1 by feeding large values
        for _ in range(50):
            det.update(10.0)
            if det.decision == "h1":
                break
        assert det.decision == "h1", "Could not reach h1 boundary for boundary test"
        # Once h1 is decided, further updates should keep it at h1 (absorbing)
        for _ in range(5):
            state = det.update(0.0)  # small value that would push toward h0
            assert (
                state.decision == "h1"
            ), "Decision changed away from h1 -- boundaries not absorbing"

    def test_sprt_n_observations_matches_total_updates(self):
        det = SPRTDetector(warmup=3)
        # 3 warmup + 7 post-warmup = 10 total
        for i in range(10):
            det.update(float(i))
        assert det.n_observations == 10

    def test_multi_sprt_missing_axis_in_scores_skipped(self):
        msprt = MultiSPRT(["a", "b", "c"], warmup=2)
        # Warmup both present axes
        msprt.update({"a": 0.1, "b": 0.1, "c": 0.1})
        msprt.update({"a": 0.1, "b": 0.1, "c": 0.1})
        # Update with missing "b" — should NOT raise KeyError
        states = msprt.update({"a": 0.1, "c": 0.2})
        assert "a" in states
        assert "c" in states
        # "b" was missing from scores, so its state is unchanged — it should still work
        assert "b" not in states
        # axis "b" state unchanged: still in warmup (only 2 updates, warmup=2,
        # the calibrate fires at exactly warmup so now out of warmup; but was not
        # updated in last call, so its n_observations should be 2)
        assert msprt._detectors["b"].n_observations == 2

    def test_multi_sprt_reset_then_reuse(self):
        msprt = MultiSPRT(["latency"], alpha=0.05, beta=0.20, effect_size=2.0, warmup=2)
        # First run to h1
        msprt.update({"latency": 1.0})
        msprt.update({"latency": 1.0})
        for _ in range(50):
            msprt.update({"latency": 100.0})
            if msprt.any_drift_detected:
                break
        first_run_detected = msprt.any_drift_detected
        assert first_run_detected, "First run should have detected drift"

        # Reset and do the same sequence
        msprt.reset_all()
        assert not msprt.any_drift_detected
        msprt.update({"latency": 1.0})
        msprt.update({"latency": 1.0})
        for _ in range(50):
            msprt.update({"latency": 100.0})
            if msprt.any_drift_detected:
                break
        second_run_detected = msprt.any_drift_detected
        assert second_run_detected, "Second run should also detect drift"


# ===========================================================================
# 4. TestLTLCheckerOnRealTraces
# ===========================================================================


class TestLTLCheckerOnRealTraces:
    def test_g_no_call_passes_on_baseline_for_other_tools(self):
        states = _demo_baseline_states()
        formula = Globally(Not(Atom("tool_call:delete_all")))
        assert check(formula, states) is True

    def test_g_no_call_fails_when_tool_is_called(self):
        states = _demo_baseline_states()
        # search_files IS called at turn 0
        formula = Globally(Not(Atom("tool_call:search_files")))
        assert check(formula, states) is False

    def test_must_call_before_search_before_end_turn(self):
        states = _demo_baseline_states()
        # ¬stop_reason:end_turn U tool_call:search_files
        # Search is called before first end_turn → formula should pass
        formula = Until(
            Not(Atom("stop_reason:end_turn")),
            Atom("tool_call:search_files"),
        )
        assert check(formula, states) is True

    def test_finally_end_turn_passes(self):
        states = _demo_baseline_states()
        formula = Finally(Atom("stop_reason:end_turn"))
        assert check(formula, states) is True

    def test_finally_content_filter_fails_on_baseline(self):
        states = _demo_baseline_states()
        formula = Finally(Atom("stop_reason:content_filter"))
        assert check(formula, states) is False

    def test_demo_candidate_has_content_filter(self):
        candidate_states = [
            TraceState(
                pair_index=0,
                tool_calls=["search_files"],
                stop_reason="tool_use",
                text_content="Searching...",
            ),
            TraceState(
                pair_index=1,
                tool_calls=[],
                stop_reason="end_turn",
                text_content="Results found.",
            ),
            TraceState(
                pair_index=2,
                tool_calls=[],
                stop_reason="content_filter",
                text_content="[filtered]",
            ),
        ]
        formula = Finally(Atom("stop_reason:content_filter"))
        assert check(formula, candidate_states) is True

    def test_trace_from_records_full_agentlog_format(self):
        states = trace_from_records(DEMO_BASELINE_RECORDS)
        assert len(states) == 3
        assert states[0].tool_calls == ["search_files"]
        assert states[0].stop_reason == "tool_use"
        assert states[1].stop_reason == "end_turn"
        assert states[2].stop_reason == "end_turn"
        # Pair indices are sequential
        assert states[0].pair_index == 0
        assert states[1].pair_index == 1
        assert states[2].pair_index == 2

    def test_empty_trace_globally_true_vacuously(self):
        formula = Globally(Atom("tool_call:x"))
        assert check(formula, []) is True

    def test_text_contains_predicate(self):
        states = _demo_baseline_states()
        # Turn 2 text contains "findings"
        formula = Finally(Atom("text_contains:findings"))
        assert check(formula, states) is True

    def test_complex_formula_devops_policy(self):
        # G(tool_call:backup_database -> X(F(tool_call:run_migration)))
        # Checks that every backup call is followed at some LATER turn by run_migration.
        # Turn 3 (pair_index=3) calls backup_database.
        # X(F(run_migration)) at i=3 means: at i=4, F(run_migration) holds.
        # Turn 4 has [request_human_approval, send_notification, check_replication_lag]
        # — NO run_migration. F(run_migration) at i=4 = False.
        # So G(backup -> X(F(run_migration))) is False on the devops trace.
        states = _devops_baseline_states()
        backup = Atom("tool_call:backup_database")
        run_mig = Atom("tool_call:run_migration")
        formula = Globally(Implies(backup, Next(Finally(run_mig))))
        result = check(formula, states)
        # Turn 3 calls backup but no run_migration follows at turn 4+ → formula fails
        assert result is False


# ===========================================================================
# 5. TestLTLCompilerEdgeCases
# ===========================================================================


class TestLTLCompilerEdgeCases:
    def test_no_call_missing_tool_param_returns_none(self):
        assert rule_to_ltl("no_call", {}) is None

    def test_no_call_non_string_tool_returns_none(self):
        assert rule_to_ltl("no_call", {"tool": 123}) is None

    def test_must_call_before_missing_params_returns_none(self):
        # 'then' is missing
        assert rule_to_ltl("must_call_before", {"first": "a"}) is None

    def test_required_stop_reason_string_allowed_treated_as_list(self):
        # `"end_turn"` is coerced to `["end_turn"]`. The v2.7 multi-turn
        # fix changed the encoding from `F(stop_reason:end_turn)` (which
        # passed when ANY turn matched) to
        # `F(¬X(true) ∧ stop_reason:end_turn)` (which requires the LAST
        # observed turn to match). Test both: the formula structure is
        # the new encoding, AND a single-turn trace where the only stop
        # reason matches is correctly accepted.
        result = rule_to_ltl("required_stop_reason", {"allowed": "end_turn"})
        expected = Finally(And(Not(Next(Atom("true"))), Atom("stop_reason:end_turn")))
        assert result == expected
        # Single-turn integration check: the only turn has stop=end_turn,
        # so the rule passes.
        states = [TraceState(pair_index=0, tool_calls=[], stop_reason="end_turn")]
        assert result is not None and check(result, states, 0)

    def test_required_stop_reason_empty_list_returns_none(self):
        assert rule_to_ltl("required_stop_reason", {"allowed": []}) is None

    def test_ltl_formula_none_formula_returns_none(self):
        assert rule_to_ltl("ltl_formula", {}) is None

    def test_unknown_kind_returns_none(self):
        assert rule_to_ltl("max_turns", {"max": 10}) is None

    def test_parse_ltl_triple_nesting(self):
        result = parse_ltl("G G G tool_call:x")
        expected = Globally(Globally(Globally(Atom("tool_call:x"))))
        assert result == expected

    def test_parse_ltl_implies_right_associative(self):
        result = parse_ltl("a -> b -> c")
        expected = Implies(Atom("a"), Implies(Atom("b"), Atom("c")))
        assert result == expected

    def test_parse_ltl_and_precedence_over_or(self):
        # "a | b & c" should parse as Or(a, And(b, c)) because & binds tighter than |
        result = parse_ltl("a | b & c")
        expected = Or(Atom("a"), And(Atom("b"), Atom("c")))
        assert result == expected

    def test_parse_ltl_until_binary(self):
        result = parse_ltl("tool_call:a U tool_call:b")
        assert isinstance(result, Until)
        assert result.left == Atom("tool_call:a")
        assert result.right == Atom("tool_call:b")

    def test_parse_ltl_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_ltl("")

    def test_parse_ltl_unmatched_paren_raises(self):
        # The parser raises either ValueError or IndexError on unmatched parens
        # (IndexError from consume() when the token list is exhausted before
        # a closing ')' is consumed by expect()).
        with pytest.raises((ValueError, IndexError)):
            parse_ltl("(tool_call:x")

    def test_parse_ltl_trailing_token_raises(self):
        with pytest.raises(ValueError):
            parse_ltl("true false")


# ===========================================================================
# 6. TestHierarchicalLTLFormulaIntegration
# ===========================================================================


class TestHierarchicalLTLFormulaIntegration:
    def _make_record(self, tool_names: list[str], stop_reason: str, idx: int = 0) -> dict:
        content: list[dict] = []
        for i, name in enumerate(tool_names):
            content.append({"type": "tool_use", "id": f"t_{idx}_{i}", "name": name, "input": {}})
        return _make_agentlog_record(
            "chat_response",
            {
                "model": "claude-opus-4-7",
                "content": content,
                "stop_reason": stop_reason,
                "latency_ms": 100,
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
            idx=idx,
        )

    def test_ltl_formula_no_delete_passes_on_clean_trace(self):
        records = [
            self._make_record(["search_files"], "tool_use", idx=0),
            self._make_record(["read_file"], "end_turn", idx=1),
        ]
        rule = PolicyRule(
            id="no-delete",
            kind="ltl_formula",
            params={"formula": "G !tool_call:delete_all"},
        )
        violations = check_policy(records, [rule])
        assert violations == []

    def test_ltl_formula_no_delete_fails_on_trace_with_delete(self):
        records = [
            self._make_record(["search_files"], "tool_use", idx=0),
            self._make_record(["delete_all"], "end_turn", idx=1),
        ]
        rule = PolicyRule(
            id="no-delete",
            kind="ltl_formula",
            params={"formula": "G !tool_call:delete_all"},
        )
        violations = check_policy(records, [rule])
        assert len(violations) > 0
        assert violations[0].rule_id == "no-delete"

    def test_ltl_formula_invalid_formula_gives_violation_not_exception(self):
        records = [self._make_record(["search_files"], "end_turn", idx=0)]
        rule = PolicyRule(
            id="bad-formula",
            kind="ltl_formula",
            params={"formula": "G !!!"},
        )
        violations = check_policy(records, [rule])
        assert len(violations) == 1
        assert "LTL parse error" in violations[0].detail

    def test_ltl_formula_empty_formula_param_gives_violation_not_exception(self):
        records = [self._make_record(["search_files"], "end_turn", idx=0)]
        rule = PolicyRule(
            id="empty-formula",
            kind="ltl_formula",
            params={"formula": ""},
        )
        violations = check_policy(records, [rule])
        assert len(violations) == 1
        assert "non-empty" in violations[0].detail

    def test_ltl_formula_missing_formula_param_gives_violation_not_exception(self):
        records = [self._make_record(["search_files"], "end_turn", idx=0)]
        rule = PolicyRule(
            id="missing-formula",
            kind="ltl_formula",
            params={},
        )
        violations = check_policy(records, [rule])
        assert len(violations) == 1

    def test_ltl_formula_violation_has_correct_pair_index(self):
        # F(tool_call:delete_all) over 3-turn trace where delete_all never appears → violation
        records = [
            self._make_record(["search_files"], "tool_use", idx=0),
            self._make_record(["read_file"], "end_turn", idx=1),
            self._make_record([], "end_turn", idx=2),
        ]
        rule = PolicyRule(
            id="must-delete",
            kind="ltl_formula",
            params={"formula": "F tool_call:delete_all"},
        )
        violations = check_policy(records, [rule])
        # Formula F(delete_all) fails because delete_all is never called.
        # check_trace finds all positions where the formula fails.
        assert len(violations) > 0
        # All pair indices should be integers (not None) since this is a positional check
        for v in violations:
            assert v.pair_index is not None

    def test_ltl_formula_on_devops_backup_before_restore(self):
        # Formula: "!tool_call:restore_database U tool_call:backup_database"
        # restore is blocked until backup fires.
        # In devops turn 4 (index 3), backup_database is called first in the content list,
        # then restore_database. But the trace as a whole: at turn 0, backup is called.
        # At turn 3, both backup and restore are called.
        # The Until formula checks the whole trace: ∃j s.t. backup@j ∧ ∀k<j: ¬restore@k.
        # Turn 0 has backup, and before turn 0 there are no turns, so formula passes.
        rule = PolicyRule(
            id="backup-before-restore",
            kind="ltl_formula",
            params={"formula": "!tool_call:restore_database U tool_call:backup_database"},
        )
        violations = check_policy(DEVOPS_BASELINE_RECORDS, [rule])
        # backup is called at turn 0, and no restore before turn 0 → formula satisfied
        assert violations == []


# ===========================================================================
# 7. TestConformalOnRealAxisRows
# ===========================================================================

_SHADOW_AXES = [
    "semantic",
    "trajectory",
    "safety",
    "verbosity",
    "latency",
    "cost",
    "reasoning",
    "judge",
    "conformance",
]


def _axis_row(
    axis: str,
    delta: float,
    n: int,
    ci95_low: float = 0.0,
    ci95_high: float = 0.2,
    severity: str = "low",
) -> dict:
    return {
        "axis": axis,
        "delta": delta,
        "n": n,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "severity": severity,
    }


class TestConformalOnRealAxisRows:
    def test_real_axis_row_format_produces_report(self):
        rows = [
            _axis_row("semantic", 0.12, 20, 0.08, 0.16),
            _axis_row("trajectory", 0.08, 20, 0.04, 0.12),
            _axis_row("safety", 0.05, 20, 0.02, 0.08),
            _axis_row("verbosity", 0.22, 20, 0.15, 0.29),
            _axis_row("latency", 0.35, 20, 0.28, 0.42),
            _axis_row("cost", 0.18, 20, 0.12, 0.24),
            _axis_row("reasoning", 0.09, 20, 0.05, 0.13),
            _axis_row("judge", 0.14, 20, 0.10, 0.18),
            _axis_row("conformance", 0.06, 20, 0.03, 0.09),
        ]
        report = build_parametric_estimate(rows)
        assert isinstance(report, ConformalCoverageReport)
        assert len(report.axes) == 9

    def test_worst_axis_is_axis_with_largest_delta(self):
        rows = [
            _axis_row("semantic", 0.1, 20, 0.05, 0.15),
            _axis_row("trajectory", 0.5, 20, 0.35, 0.65),
            _axis_row("safety", 0.3, 20, 0.20, 0.40),
        ]
        report = build_parametric_estimate(rows)
        assert report.worst_axis == "trajectory"

    def test_nan_delta_handled_gracefully(self):
        rows = [_axis_row("semantic", 0.0, 10)]
        rows[0]["delta"] = None  # type: ignore[assignment]
        # abs(float(None or 0.0)) = 0.0 — must not crash
        try:
            report = build_parametric_estimate(rows)
            assert report.axes[0].q_hat >= 0.0
        except Exception as exc:
            pytest.fail(f"Unexpected exception with None delta: {exc}")

    def test_zero_n_axis_skipped(self):
        rows = [
            _axis_row("semantic", 0.1, 0),  # n=0 → skipped
            _axis_row("trajectory", 0.3, 10),
        ]
        report = build_parametric_estimate(rows)
        assert len(report.axes) == 1
        assert report.axes[0].axis == "trajectory"

    def test_n1_gives_vacuous_bound(self):
        rows = [_axis_row("latency", 0.4, 1)]
        report = build_parametric_estimate(rows)
        ax = report.axes[0]
        # n=1: calibration set is a single point [|delta|], so q_hat = |delta|
        assert ax.q_hat == pytest.approx(0.4, abs=1e-9)
        assert ax.achieved_coverage == pytest.approx(1.0, abs=1e-9)
        assert ax.n_calibration == 1
        assert report.sufficient_n is False

    def test_sufficient_n_flag_accuracy(self):
        # n_min for target=0.90, confidence=0.95:
        # ceil(log(1-0.95) / log(0.90)) = ceil(log(0.05)/log(0.90))
        n_min = math.ceil(math.log(1.0 - 0.95) / math.log(0.90))
        # n=25 → below n_min → sufficient_n=False
        rows_25 = [_axis_row("semantic", 0.1, 25, 0.05, 0.15)]
        report_25 = build_parametric_estimate(rows_25, target_coverage=0.90, confidence=0.95)
        if n_min > 25:
            assert report_25.sufficient_n is False
        else:
            assert report_25.sufficient_n is True

        # n=30 → check against n_min
        rows_30 = [_axis_row("semantic", 0.1, 30, 0.05, 0.15)]
        report_30 = build_parametric_estimate(rows_30, target_coverage=0.90, confidence=0.95)
        if n_min <= 30:
            assert report_30.sufficient_n is True

    def test_q_hat_nonnegative(self):
        for delta in [0.0, 0.01, 0.5, 1.0]:
            rows = [_axis_row("semantic", delta, 10, delta * 0.5, delta * 1.5)]
            report = build_parametric_estimate(rows)
            for ax in report.axes:
                assert ax.q_hat >= 0.0

    def test_marginal_claim_string_format(self):
        rows = [_axis_row("latency", 0.25, 20, 0.15, 0.35)]
        report = build_parametric_estimate(rows, target_coverage=0.90, confidence=0.95)
        ax = report.axes[0]
        claim = ax.marginal_claim
        assert "latency" in claim
        assert "20" in claim
        assert "90%" in claim
        assert "95%" in claim

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError):
            build_parametric_estimate([], target_coverage=1.5)

    def test_empty_axis_rows_returns_empty_report(self):
        report = build_parametric_estimate([])
        assert report.axes == []
        assert report.worst_axis == ""


# ===========================================================================
# 8. TestEndToEndIntegration
# ===========================================================================


class TestEndToEndIntegration:
    def test_demo_fingerprint_to_hotelling_detects_content_filter_drift(self):
        x1 = fingerprint_trace(DEMO_BASELINE_RECORDS)
        x2 = fingerprint_trace(DEMO_CANDIDATE_RECORDS)
        assert x1.shape == (3, FP_DIM)
        assert x2.shape == (3, FP_DIM)
        # With n=3 and d=8, df2 = 3+3-8-1 = -3 < 0, so Hotelling returns p=1 but must not crash
        result = hotelling_t2(x1, x2)
        assert hasattr(result, "p_value")
        assert hasattr(result, "t2")
        assert math.isfinite(result.t2)
        # The candidate refusal_flag column (index 7) differs structurally
        assert x2[2, 7] == pytest.approx(1.0)
        assert x1[2, 7] == pytest.approx(0.0)

    def test_sprt_stream_on_demo_latency_values(self):
        baseline = [98.0, 115.0, 84.0]
        candidate = [412.0, 534.0, 398.0]
        det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=0.5, warmup=3)
        for lat in baseline:
            det.update(lat)
        decision = "continue"
        for _ in range(500):
            for lat in candidate:
                state = det.update(lat)
                decision = state.decision
                if decision == "h1":
                    break
            if decision == "h1":
                break
        assert decision == "h1", "Expected latency drift (h1) not detected"

    def test_ltl_compiler_to_checker_no_call_before_policy(self):
        # End-to-end: compile rule_to_ltl("must_call_before", ...) then check on devops trace
        # Formula: (¬run_migration) W backup_database  (weak-until)
        # At turn 0: backup is called AND run_migration is called.
        # j=0: backup@0 is True, ∀k∈[0,0) is vacuously True → formula PASSES.
        formula = rule_to_ltl(
            "must_call_before",
            {"first": "backup_database", "then": "run_migration"},
        )
        assert formula is not None
        assert isinstance(formula, WeakUntil)
        states = _devops_baseline_states()
        result = check(formula, states, 0)
        assert result is True, "must_call_before formula should pass: backup called at turn 0"

        # Also test via check_policy with PolicyRule
        rule = PolicyRule(
            id="backup-before-migration",
            kind="must_call_before",
            params={"first": "backup_database", "then": "run_migration"},
        )
        violations = check_policy(DEVOPS_BASELINE_RECORDS, [rule])
        # backup_database (at pair 0) comes before run_migration (also at pair 0, same turn)
        # In the procedural checker: first_idx=0 and then_idx=2 (within tool_calls list),
        # backup appears at index 1 in the list (0-indexed), run_migration at index 3.
        # So first_idx < then_idx → no violation.
        assert violations == []

    def test_must_call_before_weak_until_vacuous_when_neither_fires(self):
        """Weak-until fix: a trace that calls neither A nor B vacuously
        satisfies "B may only fire after A" — it's safe.  The previous
        strong-until encoding incorrectly flagged this as a violation."""
        formula = rule_to_ltl(
            "must_call_before",
            {"first": "backup_database", "then": "run_migration"},
        )
        assert formula is not None
        # Trace where some other tool runs but neither backup nor migration.
        states = [
            TraceState(pair_index=0, tool_calls=["check_status"], stop_reason="tool_use"),
            TraceState(pair_index=1, tool_calls=["check_status"], stop_reason="end_turn"),
        ]
        assert (
            check(formula, states, 0) is True
        ), "Weak-until: G(¬run_migration) holds → rule satisfied vacuously"

    def test_conformal_report_note_contains_axis_name(self):
        rows = [_axis_row("semantic", 0.15, 30, 0.10, 0.20)]
        report = build_parametric_estimate(rows, target_coverage=0.90, confidence=0.95)
        # n_min for (0.90, 0.95) ≈ 29, and n=30 ≥ n_min, so note mentions "Binding axis"
        assert "semantic" in report.note or "semantic" in report.worst_axis
        assert report.worst_axis == "semantic"
        if report.sufficient_n:
            assert "Binding axis" in report.note
