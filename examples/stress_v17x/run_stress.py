"""Adverse stress test for v1.7.0: must_match_json_schema + ABOM.

This is not a unit test. It probes both new features under extreme
conditions — massive payloads, hostile inputs, tampering, concurrency,
adversarial schemas, malformed JSON, performance scaling — so the
release is verified end-to-end before publishing.

Run:

    .venv/bin/python examples/stress_v17x/run_stress.py

No API key needed. Pure deterministic stress (the features under test
don't call out).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.certify import (
    build_certificate,
    render_terminal,
    verify_certificate,
)
from shadow.hierarchical import (
    PolicyRule,
    _check_must_match_json_schema,
)
from shadow.sdk import Session

# ---- pretty printing ----------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
DIM = "\033[2m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}")
    if detail:
        print(f"        {DIM}{detail}{RESET}")
    results.append((name, ok, detail))


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ---- helpers ------------------------------------------------------------


def _resp(text: str) -> dict[str, Any]:
    return {
        "kind": "chat_response",
        "id": "sha256:r",
        "ts": "t",
        "parent": "sha256:m",
        "payload": {
            "model": "m",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": 0,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        },
    }


REFUND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["decision", "amount"],
    "properties": {
        "decision": {"type": "string", "enum": ["refund", "decline", "escalate"]},
        "amount": {"type": "number", "minimum": 0},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


def _make_trace(
    path: Path, *, n_turns: int = 1, model: str = "claude-opus-4-7"
) -> None:
    with Session(output_path=path, tags={"env": "stress"}) as s:
        for _ in range(n_turns):
            s.record_chat(
                request={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a refund agent."},
                        {"role": "user", "content": "refund please"},
                    ],
                    "params": {"temperature": 0.0},
                    "tools": [
                        {
                            "name": "lookup_order",
                            "description": "look up an order",
                            "input_schema": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"],
                            },
                        }
                    ],
                },
                response={
                    "model": model,
                    "content": [
                        {"type": "text", "text": '{"decision":"refund","amount":42}'}
                    ],
                    "stop_reason": "end_turn",
                    "latency_ms": 10,
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 1,
                        "thinking_tokens": 0,
                    },
                },
            )


# ========================================================================
# JSON-Schema rule stress
# ========================================================================


def stress_schema_valid_payload_passes() -> None:
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": REFUND_SCHEMA},
        severity="error",
    )
    out = _check_must_match_json_schema(
        rule,
        [
            _resp(
                json.dumps(
                    {"decision": "refund", "amount": 49.99, "reason": "wrong colour"}
                )
            )
        ],
    )
    report("valid payload passes schema", out == [])


def stress_schema_malformed_json_varieties() -> None:
    """Truncated, unbalanced, surrogate pair, BOM, comment-style — every
    flavour of broken JSON must produce a violation, never crash."""
    cases = {
        "truncated": '{"decision": "refund", "amount":',
        "unbalanced_braces": '{"decision": "refund"',
        "trailing_comma": '{"decision": "refund", "amount": 1,}',
        "single_quotes": "{'decision': 'refund', 'amount': 1}",
        "leading_zero_int": '{"decision":"refund","amount":01}',
        "bom_prefix": '﻿{"decision":"refund","amount":1}',
        "literal_nan": '{"decision":"refund","amount":NaN}',
        "literal_infinity": '{"decision":"refund","amount":Infinity}',
        "javascript_comment": '{"decision":"refund"/*nope*/,"amount":1}',
        "completely_empty": "",
        "just_whitespace": "   \n\t   ",
        "html_tags": "<html>not json</html>",
    }
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": REFUND_SCHEMA},
        severity="error",
    )
    crashed: list[str] = []
    no_violation: list[str] = []
    for name, raw in cases.items():
        try:
            out = _check_must_match_json_schema(rule, [_resp(raw)])
            if not out:
                no_violation.append(name)
        except Exception as e:  # noqa: BLE001
            crashed.append(f"{name}: {type(e).__name__}: {e}")
    report(
        f"all {len(cases)} malformed-JSON variants surface as violations (no crashes)",
        not crashed and not no_violation,
        f"crashed={crashed} silently_passed={no_violation}",
    )


def stress_schema_unicode_and_special_chars() -> None:
    """Emoji, RTL text, surrogate pairs, zero-width spaces must validate
    cleanly — none should false-positive or crash the JSON parser."""
    payloads = [
        {"decision": "refund", "amount": 42, "reason": "🦀 wrong colour"},
        {"decision": "refund", "amount": 42, "reason": "اللون الخطأ"},  # arabic
        {"decision": "refund", "amount": 42, "reason": "色がおかしい"},  # japanese
        {"decision": "refund", "amount": 42, "reason": "zero​width"},  # zero-width
        {"decision": "refund", "amount": 42, "reason": "\\backslashes\\too"},
    ]
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": REFUND_SCHEMA},
        severity="error",
    )
    crashed = 0
    for p in payloads:
        try:
            out = _check_must_match_json_schema(
                rule, [_resp(json.dumps(p, ensure_ascii=False))]
            )
            if out:
                crashed += 1
        except Exception:  # noqa: BLE001
            crashed += 1
    report(
        f"{len(payloads)} unicode/RTL/surrogate payloads validate cleanly", crashed == 0
    )


def stress_schema_huge_payload_scales() -> None:
    """100 KB JSON document with thousands of items must validate in
    well under one second."""
    big_schema: dict[str, Any] = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "integer"},
                        "tag": {"type": "string"},
                    },
                },
            }
        },
    }
    big_doc = {"items": [{"id": i, "tag": f"row-{i}"} for i in range(2000)]}
    raw = json.dumps(big_doc)
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": big_schema},
        severity="error",
    )
    t0 = time.perf_counter()
    out = _check_must_match_json_schema(rule, [_resp(raw)])
    elapsed = time.perf_counter() - t0
    report(
        f"~{len(raw) // 1024} KB / 2000-item payload validates in {elapsed * 1000:.0f}ms",
        out == [] and elapsed < 1.0,
        f"elapsed={elapsed * 1000:.1f}ms",
    )


def stress_schema_complex_refs_and_oneof() -> None:
    """Schemas using $ref, oneOf, allOf must work as documented."""
    schema = {
        "$defs": {
            "address": {
                "type": "object",
                "required": ["street"],
                "properties": {"street": {"type": "string"}, "zip": {"type": "string"}},
            }
        },
        "oneOf": [
            {
                "type": "object",
                "required": ["kind", "address"],
                "properties": {
                    "kind": {"const": "billing"},
                    "address": {"$ref": "#/$defs/address"},
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["kind", "email"],
                "properties": {
                    "kind": {"const": "contact"},
                    "email": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ],
    }
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": schema},
        severity="error",
    )
    valid_a = json.dumps({"kind": "billing", "address": {"street": "1 Main"}})
    valid_b = json.dumps({"kind": "contact", "email": "a@b.com"})
    invalid = json.dumps({"kind": "contact"})  # missing email
    a_out = _check_must_match_json_schema(rule, [_resp(valid_a)])
    b_out = _check_must_match_json_schema(rule, [_resp(valid_b)])
    bad_out = _check_must_match_json_schema(rule, [_resp(invalid)])
    report(
        "oneOf/$ref schemas validate correctly",
        a_out == [] and b_out == [] and len(bad_out) == 1,
        f"a={len(a_out)} b={len(b_out)} bad={len(bad_out)}",
    )


def stress_schema_invalid_meta_short_circuits() -> None:
    """A schema that's itself invalid must short-circuit to one violation
    instead of producing N (per response). Important for PR comments
    that would otherwise overflow."""
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": {"type": "totally-fake-type"}},
        severity="error",
    )
    out = _check_must_match_json_schema(rule, [_resp("{}") for _ in range(50)])
    report(
        "invalid schema short-circuits to one violation (not 50)",
        len(out) == 1 and "schema itself is invalid" in out[0].detail,
    )


def stress_schema_external_path_pathological(tmp_path: Path) -> None:
    """schema_path with: not found / invalid JSON / unreadable / empty file."""
    cases: list[tuple[str, Path | None, str]] = []
    cases.append(("missing", tmp_path / "missing.schema.json", "schema_path not found"))
    bad_json = tmp_path / "bad.schema.json"
    bad_json.write_text("{not json")
    cases.append(("bad-json", bad_json, "not valid JSON"))
    empty = tmp_path / "empty.schema.json"
    empty.write_text("")
    cases.append(("empty", empty, "not valid JSON"))

    failures: list[str] = []
    for name, path, expected_substr in cases:
        rule = PolicyRule(
            id="r",
            kind="must_match_json_schema",
            params={"schema_path": str(path)},
            severity="error",
        )
        out = _check_must_match_json_schema(rule, [_resp(json.dumps({"x": 1}))])
        if len(out) != 1 or expected_substr not in out[0].detail:
            failures.append(f"{name}: got {out}")
    report(
        f"{len(cases)} pathological schema_path cases handled cleanly",
        not failures,
        f"{failures}",
    )


def stress_schema_both_or_neither_param() -> None:
    """Exactly one of schema|schema_path must be supplied — both or
    neither produces a single clear violation."""
    both = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": REFUND_SCHEMA, "schema_path": "/x"},
        severity="error",
    )
    neither = PolicyRule(
        id="r", kind="must_match_json_schema", params={}, severity="error"
    )
    out_both = _check_must_match_json_schema(both, [_resp("{}")])
    out_neither = _check_must_match_json_schema(neither, [_resp("{}")])
    report(
        "schema + schema_path together → 1 violation",
        len(out_both) == 1 and "exactly one" in out_both[0].detail,
    )
    report(
        "schema + schema_path missing → 1 violation",
        len(out_neither) == 1 and "exactly one" in out_neither[0].detail,
    )


def stress_schema_concurrent_validation() -> None:
    """Run policy_diff with the schema rule across 50 concurrent threads
    on the same shared rule object. Must not crash, must produce
    deterministic violation counts."""
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": REFUND_SCHEMA},
        severity="error",
    )
    valid = json.dumps({"decision": "refund", "amount": 1})
    invalid = "not json at all"

    def _one(_seed: int) -> int:
        responses = [_resp(valid if i % 2 == 0 else invalid) for i in range(20)]
        out = _check_must_match_json_schema(rule, responses)
        return len(out)

    with ThreadPoolExecutor(max_workers=16) as ex:
        counts = list(ex.map(_one, range(50)))
    expected = 10  # half the responses are invalid
    report(
        f"50 concurrent threads × 20 responses each return deterministic count={expected}",
        all(c == expected for c in counts),
        f"counts unique: {sorted(set(counts))}",
    )


def stress_schema_per_response_index_reported() -> None:
    """When the 7th response in a list violates the schema, the
    violation must carry pair_index=6 (zero-based)."""
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={"schema": REFUND_SCHEMA},
        severity="error",
    )
    valid = json.dumps({"decision": "refund", "amount": 1})
    responses = [_resp(valid)] * 6 + [_resp("not json")] + [_resp(valid)] * 3
    out = _check_must_match_json_schema(rule, responses)
    report(
        "violation carries the right pair_index (6) within the response list",
        len(out) == 1 and out[0].pair_index == 6,
        f"got pair_index={out[0].pair_index if out else None}",
    )


# ========================================================================
# ABOM (certify) stress
# ========================================================================


def stress_certify_deterministic_with_fixed_timestamp(tmp_path: Path) -> None:
    """Two builds with the same `released_at` and the same trace must
    produce identical `cert_id`. Anything less means the format isn't
    actually content-addressed."""
    import datetime

    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    fixed_ts = datetime.datetime(2026, 4, 25, 12, 0, 0, tzinfo=datetime.UTC)

    a = build_certificate(trace=records, agent_id="x", released_at=fixed_ts)
    b = build_certificate(trace=records, agent_id="x", released_at=fixed_ts)
    report(
        "identical inputs produce identical cert_id (deterministic)",
        a.cert_id == b.cert_id,
    )


def stress_certify_different_timestamps_differ(tmp_path: Path) -> None:
    """released_at is part of the hashed body. Two builds with different
    timestamps MUST produce different cert_id."""
    import datetime

    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())

    a = build_certificate(
        trace=records,
        agent_id="x",
        released_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
    )
    b = build_certificate(
        trace=records,
        agent_id="x",
        released_at=datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC),
    )
    report("different released_at produces different cert_id", a.cert_id != b.cert_id)


def stress_certify_per_field_tamper_detection(tmp_path: Path) -> None:
    """Tamper with each canonical field individually; verify_certificate
    must reject every single mutation."""
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path, n_turns=2)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="x")
    base = cert.to_dict()

    mutations: list[tuple[str, dict[str, Any]]] = [
        ("agent_id", {**base, "agent_id": "evil"}),
        ("released_at", {**base, "released_at": "2099-12-31T23:59:59Z"}),
        ("trace_id", {**base, "trace_id": "sha256:fakeoldtrace"}),
        ("models append", {**base, "models": [*base["models"], "gpt-4-evil"]}),
        ("models remove", {**base, "models": []}),
        (
            "models reorder",
            {**base, "models": list(reversed(base["models"]))}
            if len(base["models"]) >= 2
            else None,
        ),
        ("prompt_hashes mutate", {**base, "prompt_hashes": ["sha256:badprompt"]}),
        (
            "tool_schemas mutate",
            {**base, "tool_schemas": [{"name": "evil_tool", "hash": "sha256:bad"}]},
        ),
        ("policy_hash inject", {**base, "policy_hash": "sha256:fakepolicy"}),
        (
            "regression_suite inject",
            {**base, "regression_suite": {"baseline_trace_id": "x", "axes": []}},
        ),
    ]

    failures: list[str] = []
    for name, mut in mutations:
        if mut is None:
            continue
        ok, _detail = verify_certificate(mut)
        if ok:
            failures.append(name)
    report(
        f"every per-field tamper rejected ({sum(1 for _, m in mutations if m is not None)} mutations)",
        not failures,
        f"missed: {failures}",
    )


def stress_certify_cert_id_alone_tamper(tmp_path: Path) -> None:
    """Replacing only cert_id (with body unchanged) must fail."""
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="x")
    body = cert.to_dict()
    body["cert_id"] = "sha256:" + ("0" * 64)
    ok, detail = verify_certificate(body)
    report("replacing only cert_id (body unchanged) is rejected", not ok, detail)


def stress_certify_round_trip_through_disk(tmp_path: Path) -> None:
    """Round-trip JSON serialise → write → read → parse → verify must pass.
    Tests that the serialiser is canonical-preserving."""
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path, n_turns=3)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="round-trip")

    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(cert.to_dict(), indent=2) + "\n")
    payload = json.loads(cert_path.read_text())
    ok, detail = verify_certificate(payload)
    report("round-trip via disk preserves cert_id", ok, detail)


def stress_certify_concurrent_builds_match(tmp_path: Path) -> None:
    """20 threads building a cert from the same trace + fixed timestamp
    must all produce the same cert_id."""
    import datetime

    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path, n_turns=2)
    records = _core.parse_agentlog(trace_path.read_bytes())
    fixed_ts = datetime.datetime(2026, 4, 25, 12, 0, 0, tzinfo=datetime.UTC)

    def _build(_i: int) -> str:
        c = build_certificate(trace=records, agent_id="x", released_at=fixed_ts)
        return c.cert_id

    with ThreadPoolExecutor(max_workers=8) as ex:
        ids = list(ex.map(_build, range(20)))
    report(
        "20 concurrent build_certificate() calls produce one unique cert_id",
        len(set(ids)) == 1,
        f"distinct ids={len(set(ids))}",
    )


def stress_certify_unsupported_versions_rejected() -> None:
    bad_payloads = [
        {"cert_version": "0.0", "cert_id": "sha256:x"},
        {"cert_version": "0.99", "cert_id": "sha256:x"},
        {"cert_version": "1.0", "cert_id": "sha256:x"},
        {"cert_version": None, "cert_id": "sha256:x"},
        {"cert_id": "sha256:x"},  # missing version
    ]
    misses: list[Any] = []
    for p in bad_payloads:
        ok, _ = verify_certificate(p)
        if ok:
            misses.append(p.get("cert_version"))
    report(
        f"{len(bad_payloads)} bad cert_version values rejected",
        not misses,
        f"missed: {misses}",
    )


def stress_certify_malformed_cert_id_rejected() -> None:
    cases = [
        {"cert_version": "0.1", "cert_id": "not-a-hash"},
        {"cert_version": "0.1", "cert_id": "md5:abcd"},
        {"cert_version": "0.1", "cert_id": ""},
        {"cert_version": "0.1", "cert_id": None},
        {"cert_version": "0.1"},  # missing entirely
    ]
    misses: list[Any] = []
    for p in cases:
        ok, _ = verify_certificate(p)
        if ok:
            misses.append(p.get("cert_id"))
    report(f"{len(cases)} malformed cert_id values rejected", not misses)


def stress_certify_empty_trace_raises() -> None:
    from shadow.errors import ShadowConfigError

    raised = False
    try:
        build_certificate(trace=[], agent_id="x")
    except ShadowConfigError:
        raised = True
    report("empty trace raises ShadowConfigError", raised)


def stress_certify_huge_trace_scales(tmp_path: Path) -> None:
    """100-turn trace with many distinct prompts/tools/models must
    certify in well under 5 seconds."""
    trace_path = tmp_path / "huge.agentlog"
    with Session(output_path=trace_path, tags={"env": "stress"}) as s:
        for i in range(100):
            s.record_chat(
                request={
                    "model": f"model-{i % 5}",
                    "messages": [
                        {"role": "system", "content": f"system prompt v{i % 3}"},
                        {"role": "user", "content": f"q{i}"},
                    ],
                    "params": {"temperature": 0.0},
                    "tools": [
                        {
                            "name": f"tool_{j}",
                            "description": "t",
                            "input_schema": {
                                "type": "object",
                                "properties": {"x": {"type": "string"}},
                            },
                        }
                        for j in range(i % 4)
                    ],
                },
                response={
                    "model": f"model-{i % 5}",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 1,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "thinking_tokens": 0,
                    },
                },
            )
    records = _core.parse_agentlog(trace_path.read_bytes())
    t0 = time.perf_counter()
    cert = build_certificate(trace=records, agent_id="huge")
    elapsed = time.perf_counter() - t0
    ok, _ = verify_certificate(cert.to_dict())
    report(
        f"100-turn trace with 5 models / 3 prompts / 4 tools certifies in {elapsed * 1000:.0f}ms",
        ok and elapsed < 5.0 and len(cert.models) == 5 and len(cert.prompt_hashes) == 3,
        f"elapsed={elapsed * 1000:.1f}ms models={len(cert.models)} prompts={len(cert.prompt_hashes)} tools={len(cert.tool_schemas)}",
    )


def stress_certify_extra_fields_break_or_ignore(tmp_path: Path) -> None:
    """Adding an unknown field to the certificate body must NOT silently
    succeed — the cert_id is fixed, so any extra field is ignored
    when verifying. Document and verify behaviour: extra field is
    ignored, original cert_id still verifies."""
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="x")
    body = cert.to_dict()
    body["sneaky_field"] = "this should not break verification"
    ok, _ = verify_certificate(body)
    report(
        "extra unknown field does not break verification (forward-compat)",
        ok,
        "verify_certificate rebuilds canonical body from known fields only",
    )


def stress_certify_with_baseline_regression_suite(tmp_path: Path) -> None:
    """When --baseline is supplied, regression_suite.axes must contain
    all 9 expected axes."""
    base_path = tmp_path / "base.agentlog"
    cand_path = tmp_path / "cand.agentlog"
    _make_trace(base_path, n_turns=3)
    _make_trace(cand_path, n_turns=3, model="gpt-4o-mini")
    base = _core.parse_agentlog(base_path.read_bytes())
    cand = _core.parse_agentlog(cand_path.read_bytes())
    cert = build_certificate(trace=cand, agent_id="x", baseline_trace=base)
    expected_axes = {
        "semantic",
        "trajectory",
        "safety",
        "verbosity",
        "latency",
        "cost",
        "reasoning",
        "judge",
        "conformance",
    }
    rs = cert.regression_suite or {}
    actual_axes = {a["axis"] for a in rs.get("axes", [])}
    report(
        "regression_suite includes all 9 axes",
        actual_axes == expected_axes,
        f"missing={expected_axes - actual_axes} extra={actual_axes - expected_axes}",
    )


def stress_certify_terminal_render_safe(tmp_path: Path) -> None:
    """render_terminal must produce non-empty multi-line output and not
    raise on traces with extreme content."""
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path)
    records = _core.parse_agentlog(trace_path.read_bytes())
    cert = build_certificate(trace=records, agent_id="🦀-rust-agent")
    out = render_terminal(cert)
    report(
        "render_terminal handles unicode agent_id without crashing",
        "🦀-rust-agent" in out and out.count("\n") >= 5,
    )


# ========================================================================
# CLI integration
# ========================================================================


def stress_cli_certify_then_verify_pipeline(tmp_path: Path) -> None:
    """End-to-end: certify a trace via CLI, verify-cert exits 0; tamper,
    verify-cert exits 1."""
    from typer.testing import CliRunner

    from shadow.cli.app import app

    runner = CliRunner()
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path, n_turns=2)
    cert_path = tmp_path / "out.cert.json"

    r1 = runner.invoke(
        app,
        [
            "certify",
            str(trace_path),
            "--agent-id",
            "cli-agent",
            "--output",
            str(cert_path),
        ],
    )
    r2 = runner.invoke(app, ["verify-cert", str(cert_path)])

    payload = json.loads(cert_path.read_text())
    payload["agent_id"] = "tampered"
    cert_path.write_text(json.dumps(payload))
    r3 = runner.invoke(app, ["verify-cert", str(cert_path)])

    report(
        "certify → verify-cert (clean) → tamper → verify-cert (rejected)",
        r1.exit_code == 0 and r2.exit_code == 0 and r3.exit_code == 1,
        f"exit codes: certify={r1.exit_code} verify-clean={r2.exit_code} verify-tampered={r3.exit_code}",
    )


def stress_cli_diff_with_schema_rule_and_fail_on(tmp_path: Path) -> None:
    """End-to-end: shadow diff with --policy containing a
    must_match_json_schema rule + --fail-on severe must exit 1 when the
    candidate produces non-JSON output."""
    from typer.testing import CliRunner

    from shadow.cli.app import app

    runner = CliRunner()

    # Two traces — baseline outputs valid JSON, candidate outputs prose.
    base_path = tmp_path / "b.agentlog"
    cand_path = tmp_path / "c.agentlog"
    with Session(output_path=base_path, tags={"env": "x"}) as sb:
        sb.record_chat(
            request={"model": "m", "messages": [], "params": {}},
            response={
                "model": "m",
                "content": [
                    {"type": "text", "text": '{"decision":"refund","amount":1}'}
                ],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    with Session(output_path=cand_path, tags={"env": "x"}) as sc:
        sc.record_chat(
            request={"model": "m", "messages": [], "params": {}},
            response={
                "model": "m",
                "content": [{"type": "text", "text": "I cannot help with that."}],
                "stop_reason": "end_turn",
                "latency_ms": 10,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        f"""rules:
  - id: structured-output
    kind: must_match_json_schema
    params:
      schema: {json.dumps(REFUND_SCHEMA)}
    severity: error
"""
    )
    result = runner.invoke(
        app,
        [
            "diff",
            str(base_path),
            str(cand_path),
            "--policy",
            str(policy_path),
            "--fail-on",
            "severe",
        ],
    )
    # The candidate breaks the schema rule (severity=error → severe-equivalent).
    # The gate must trip. Exit code 1 is the contract.
    report(
        "diff --policy must_match_json_schema --fail-on severe trips on candidate-only violation",
        result.exit_code == 1,
        f"exit_code={result.exit_code}",
    )


# ========================================================================
# main
# ========================================================================


def main() -> int:
    print("\033[1m=== Shadow v1.7.0 adverse stress test ===\033[0m")

    tmp = Path(tempfile.mkdtemp(prefix="shadow-stress-v17-"))
    try:
        section("must_match_json_schema — happy paths")
        stress_schema_valid_payload_passes()
        stress_schema_unicode_and_special_chars()

        section("must_match_json_schema — adverse inputs")
        stress_schema_malformed_json_varieties()
        stress_schema_huge_payload_scales()
        stress_schema_complex_refs_and_oneof()
        stress_schema_invalid_meta_short_circuits()
        stress_schema_external_path_pathological(tmp)
        stress_schema_both_or_neither_param()
        stress_schema_per_response_index_reported()

        section("must_match_json_schema — concurrency")
        stress_schema_concurrent_validation()

        section("ABOM (certify) — determinism")
        stress_certify_deterministic_with_fixed_timestamp(tmp)
        stress_certify_different_timestamps_differ(tmp)
        stress_certify_concurrent_builds_match(tmp)

        section("ABOM (certify) — tamper detection")
        stress_certify_per_field_tamper_detection(tmp)
        stress_certify_cert_id_alone_tamper(tmp)
        stress_certify_round_trip_through_disk(tmp)
        stress_certify_extra_fields_break_or_ignore(tmp)

        section("ABOM (certify) — version + format")
        stress_certify_unsupported_versions_rejected()
        stress_certify_malformed_cert_id_rejected()
        stress_certify_empty_trace_raises()

        section("ABOM (certify) — scale + integration")
        stress_certify_huge_trace_scales(tmp)
        stress_certify_with_baseline_regression_suite(tmp)
        stress_certify_terminal_render_safe(tmp)

        section("CLI end-to-end")
        stress_cli_certify_then_verify_pipeline(tmp)
        stress_cli_diff_with_schema_rule_and_fail_on(tmp)
    finally:
        # Clean up the tempdir.
        for p in sorted(tmp.rglob("*"), reverse=True):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass

    n_total = len(results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print()
    print("\033[1m=== summary ===\033[0m")
    print(f"  passed: {n_pass}/{n_total}")
    if n_pass < n_total:
        print()
        print("\033[1m  failures:\033[0m")
        for name, ok, detail in results:
            if not ok:
                print(f"    - {name}")
                if detail:
                    print(f"      {detail}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
