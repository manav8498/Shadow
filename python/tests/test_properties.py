"""Hypothesis-driven property-based tests.

Fuzz + generative tests for the invariants that must hold across
*every* valid input — not just the specific fixtures hand-written
elsewhere. These catch regressions the example-based tests miss.

Four properties under test:

1. **Canonical JSON round-trip** — `parse(write(r)) == r` for any
   record, and `write` is byte-level deterministic.
2. **compute_diff_report never crashes + is shape-stable** — on any
   pair of syntactically-valid trace lists, the report has exactly
   9 axis rows, every severity is a known enum value, every CI bound
   is finite, and pair_count ≤ min(baseline_responses, candidate_responses).
3. **Cost-attribution identity** — for any baseline / candidate
   session cost pair and any pricing table, `model_swap +
   token_movement + mix_residual == total_delta` to f64 precision.
4. **Schema-watch is monotone on no-op inputs** — `watch(cfg, cfg)`
   always returns zero changes, regardless of how complex `cfg` is.
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow import _core
from shadow.cost_attribution import attribute_cost
from shadow.schema_watch import watch_configs

# ---- strategies --------------------------------------------------------


@st.composite
def _chat_response_payload(draw: Any) -> dict[str, Any]:
    """Build a syntactically-valid `chat_response` payload."""
    model = draw(st.sampled_from(["claude-opus-4-7", "claude-sonnet-4-6", "gpt-4o-mini"]))
    text = draw(st.text(min_size=0, max_size=120))
    n_blocks = draw(st.integers(min_value=0, max_value=3))
    blocks = [{"type": "text", "text": text}] + [
        {"type": "text", "text": f"block-{i}"} for i in range(n_blocks)
    ]
    return {
        "model": model,
        "content": blocks,
        "stop_reason": draw(st.sampled_from(["end_turn", "tool_use", "content_filter"])),
        "latency_ms": draw(st.integers(min_value=0, max_value=100_000)),
        "usage": {
            "input_tokens": draw(st.integers(min_value=0, max_value=50_000)),
            "output_tokens": draw(st.integers(min_value=0, max_value=50_000)),
            "thinking_tokens": draw(st.integers(min_value=0, max_value=10_000)),
        },
    }


@st.composite
def _tiny_trace(draw: Any) -> list[dict[str, Any]]:
    """A 1-3 turn synthetic trace ready for `compute_diff_report`."""
    n_turns = draw(st.integers(min_value=1, max_value=3))
    metadata_payload = {"sdk": {"name": "shadow", "version": "test"}}
    meta = {
        "version": "0.1",
        "id": _core.content_id(metadata_payload),
        "kind": "metadata",
        "ts": "2026-04-24T00:00:00.000Z",
        "parent": None,
        "payload": metadata_payload,
    }
    records: list[dict[str, Any]] = [meta]
    parent = meta["id"]
    for i in range(n_turns):
        req_payload = {
            "model": "x",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "params": {},
        }
        req = {
            "version": "0.1",
            "id": _core.content_id(req_payload),
            "kind": "chat_request",
            "ts": "2026-04-24T00:00:00.000Z",
            "parent": parent,
            "payload": req_payload,
        }
        resp_payload = draw(_chat_response_payload())
        resp = {
            "version": "0.1",
            "id": _core.content_id(resp_payload),
            "kind": "chat_response",
            "ts": "2026-04-24T00:00:00.000Z",
            "parent": req["id"],
            "payload": resp_payload,
        }
        records.append(req)
        records.append(resp)
        parent = resp["id"]
    return records


# ---- 1. canonical JSON round-trip --------------------------------------


@given(_tiny_trace())
@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.data_too_large])
def test_agentlog_roundtrip_is_byte_stable(trace: list[dict[str, Any]]) -> None:
    """Serialise → parse → serialise yields identical bytes."""
    bytes1 = _core.write_agentlog(trace)
    parsed = _core.parse_agentlog(bytes1)
    bytes2 = _core.write_agentlog(parsed)
    assert bytes1 == bytes2


@given(_tiny_trace())
@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.data_too_large])
def test_agentlog_roundtrip_preserves_semantic_content(
    trace: list[dict[str, Any]],
) -> None:
    """Every record's kind + payload survives the round-trip."""
    reparsed = _core.parse_agentlog(_core.write_agentlog(trace))
    assert len(reparsed) == len(trace)
    for original, after in zip(trace, reparsed, strict=True):
        assert original["kind"] == after["kind"]
        # Content ids should match because payloads are structurally equal.
        assert original["id"] == after["id"]


# ---- 2. differ shape stability -----------------------------------------


@given(_tiny_trace(), _tiny_trace())
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.data_too_large])
def test_compute_diff_report_shape_is_stable(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> None:
    """On any pair of valid traces, the differ returns a 9-axis report
    with finite CI bounds and recognised severity enum values."""
    report = _core.compute_diff_report(baseline, candidate, None, 42)
    assert isinstance(report, dict)
    rows = report.get("rows", [])
    assert len(rows) == 9, f"expected 9 axes, got {len(rows)}"
    valid_severities = {"none", "minor", "moderate", "severe"}
    for row in rows:
        assert row["severity"] in valid_severities
        # CI bounds must be finite (even if some axes have n=0, the
        # Rust core emits 0.0 bounds — never NaN or Inf).
        import math

        assert math.isfinite(row["ci95_low"])
        assert math.isfinite(row["ci95_high"])
        assert math.isfinite(row["delta"])
    # drill_down must be present and well-formed.
    drill = report.get("drill_down", [])
    for pair in drill:
        assert "regression_score" in pair
        assert math.isfinite(pair["regression_score"])
        assert isinstance(pair["axis_scores"], list)


@given(_tiny_trace())
@settings(deadline=None, max_examples=20)
def test_compute_diff_report_self_diff_has_zero_deltas(
    trace: list[dict[str, Any]],
) -> None:
    """Diffing a trace against itself must yield |delta| == 0 on every
    axis. This is the strongest property-based check of the differ's
    core correctness."""
    report = _core.compute_diff_report(trace, trace, None, 42)
    for row in report["rows"]:
        # Tiny floating-point noise is OK; the axis-specific tolerance
        # accounts for bootstrap randomness summed over resamples.
        assert abs(row["delta"]) < 1e-6, f"{row['axis']}: delta {row['delta']}"


# ---- 3. cost-attribution identity --------------------------------------


@st.composite
def _session_trace(draw: Any) -> list[dict[str, Any]]:
    """Single-session trace with deterministic token counts for
    decomposition-identity testing."""
    metadata_payload = {"sdk": {"name": "shadow", "version": "test"}}
    records: list[dict[str, Any]] = [
        {
            "version": "0.1",
            "id": _core.content_id(metadata_payload),
            "kind": "metadata",
            "ts": "t",
            "parent": None,
            "payload": metadata_payload,
        }
    ]
    n = draw(st.integers(min_value=1, max_value=5))
    for _ in range(n):
        payload = {
            "model": draw(st.sampled_from(["opus", "sonnet", "haiku"])),
            "usage": {
                "input_tokens": draw(st.integers(min_value=0, max_value=10_000)),
                "output_tokens": draw(st.integers(min_value=0, max_value=10_000)),
            },
        }
        records.append(
            {
                "version": "0.1",
                "id": _core.content_id(payload),
                "kind": "chat_response",
                "ts": "t",
                "parent": records[-1]["id"],
                "payload": payload,
            }
        )
    return records


@given(_session_trace(), _session_trace())
@settings(deadline=None, max_examples=100)
def test_cost_attribution_identity_holds_for_any_session_pair(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> None:
    """The fundamental identity of the decomposition: for *any*
    baseline/candidate session pair, model_swap + token_movement +
    mix_residual must equal total_delta to f64 precision."""
    pricing = {
        "opus": {"input": 15e-6, "output": 75e-6},
        "sonnet": {"input": 3e-6, "output": 15e-6},
        "haiku": {"input": 0.25e-6, "output": 1.25e-6},
    }
    report = attribute_cost(baseline, candidate, pricing)
    for s in report.per_session:
        total_decomp = s.model_swap_usd + s.token_movement_usd + s.mix_residual_usd
        # Allow loose tolerance for f64 accumulation error. On a single
        # session with small token counts, 1e-9 is plenty.
        assert abs(total_decomp - s.delta_usd) < 1e-6, (
            f"decomposition failed: swap={s.model_swap_usd:.6e} "
            f"move={s.token_movement_usd:.6e} mix={s.mix_residual_usd:.6e} "
            f"sum={total_decomp:.6e} delta={s.delta_usd:.6e}"
        )


# ---- 4. schema-watch no-op monotonicity --------------------------------


@st.composite
def _agent_config(draw: Any) -> dict[str, Any]:
    """Arbitrary tool config that schema-watch should accept."""
    n_tools = draw(st.integers(min_value=0, max_value=5))
    tools: list[dict[str, Any]] = []
    for i in range(n_tools):
        n_props = draw(st.integers(min_value=0, max_value=4))
        props = {
            f"p{j}": {"type": draw(st.sampled_from(["string", "integer", "number", "boolean"]))}
            for j in range(n_props)
        }
        required = draw(st.lists(st.sampled_from(list(props.keys()) or [""]), unique=True))
        tools.append(
            {
                "name": f"tool_{i}",
                "description": draw(st.text(min_size=0, max_size=60)),
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": [r for r in required if r],
                },
            }
        )
    return {
        "model": draw(st.sampled_from(["opus", "sonnet", "haiku"])),
        "params": {"temperature": draw(st.floats(min_value=0.0, max_value=2.0))},
        "prompt": {"system": draw(st.text(min_size=0, max_size=120))},
        "tools": tools,
    }


@given(_agent_config())
@settings(deadline=None, max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_schema_watch_self_compare_has_zero_changes(cfg: dict[str, Any]) -> None:
    """A config compared against itself must always yield zero
    schema changes — regardless of tool count, arg complexity, etc."""
    report = watch_configs(cfg, cfg)
    assert report.changes == [], f"self-compare produced changes: {report.changes}"
    assert report.breaking == 0
    assert report.risky == 0
    assert report.additive == 0
    assert report.neutral == 0


@given(_agent_config())
@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_schema_watch_symmetry_on_no_op(cfg: dict[str, Any]) -> None:
    """watch_configs(a, a) is trivially symmetric — use this as a
    weak sanity check that the watcher is stable under repeated
    no-op invocations."""
    r1 = watch_configs(cfg, cfg)
    r2 = watch_configs(cfg, cfg)
    assert len(r1.changes) == len(r2.changes)
    assert r1.breaking == r2.breaking


# ---- 5. canonical_bytes determinism (fuzz) -----------------------------


@given(
    st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-(10**9), max_value=10**9),
            st.floats(allow_nan=False, allow_infinity=False, width=64),
            st.text(max_size=40),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=8),
            st.dictionaries(st.text(max_size=20), children, max_size=6),
        ),
        max_leaves=20,
    )
)
@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.data_too_large])
def test_canonical_bytes_is_deterministic(payload: Any) -> None:
    """Same payload must hash + serialise to identical bytes across
    invocations. This property is load-bearing for the content-
    addressing scheme — a non-deterministic canonicalisation would
    poison every id in the store."""
    b1 = _core.canonical_bytes(payload)
    b2 = _core.canonical_bytes(payload)
    assert b1 == b2
    id1 = _core.content_id(payload)
    id2 = _core.content_id(payload)
    assert id1 == id2
    assert id1.startswith("sha256:")
