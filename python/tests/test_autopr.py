"""Tests for `shadow.autopr` — synthesis + verification + CLI command.

Coverage targets:

1. Synthesis: known-regression fixtures produce the expected rule kinds.
2. Round-trip: every emitted rule is parseable by `load_policy()`.
3. Counterfactual gate: `verify_policy` enforces the baseline-clean /
   candidate-fires invariant.
4. Idempotency: re-running the synthesizer on the same input produces
   byte-identical YAML (so generated files don't churn in version
   control on every run).
5. CLI: `shadow autopr` exits 0 on a real regression, exits 2 when the
   counterfactual gate rejects, prints YAML to stdout when no --output.
6. Real-world: against the bundled quickstart fixtures (which contain a
   known regression), the synthesizer produces a verified policy.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import shadow.quickstart_data as _qs_data
from shadow import _core
from shadow.autopr import SynthesizedPolicy, synthesize_policy, verify_policy
from shadow.cli.app import app
from shadow.hierarchical import load_policy
from shadow.sdk import Session

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundled_fixtures() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the same baseline+candidate fixtures `shadow quickstart` ships."""
    root = resources.files(_qs_data) / "fixtures"
    b = _core.parse_agentlog(root.joinpath("baseline.agentlog").read_bytes())
    c = _core.parse_agentlog(root.joinpath("candidate.agentlog").read_bytes())
    return b, c


def _record_with_stop_reason(path: Path, *, stop_reason: str, text: str = "ok") -> None:
    """Write a one-pair `.agentlog` whose chat_response carries the
    given stop_reason. The synthesizer's stop_reason rule fires when
    baseline and candidate differ on this field."""
    with Session(output_path=path, tags={"env": "test"}) as s:
        s.record_chat(
            request={"model": "claude-opus-4-7", "messages": [], "params": {}},
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
                "latency_ms": 100,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_synthesize_against_bundled_fixtures_produces_a_rule() -> None:
    """The bundled fixtures carry a known regression (stop_reason flips
    from `end_turn` to `content_filter`). The synthesizer must surface
    at least one rule for it."""
    b, c = _bundled_fixtures()
    policy = synthesize_policy(b, c, name="bundled")
    assert policy.rules, "synthesizer produced no rules on a known-regression fixture"
    kinds = {r["kind"] for r in policy.rules}
    # The fixture's most pinnable signal is the stop_reason flip on the
    # safety axis. Other signals (latency, semantic) don't have policy
    # kinds — they must NOT show up here.
    assert "required_stop_reason" in kinds


def test_synthesize_idempotent(tmp_path: Path) -> None:
    """Re-running the synthesizer on the same input must produce
    byte-identical YAML so generated files don't churn in version
    control on every run."""
    b, c = _bundled_fixtures()
    p1 = synthesize_policy(b, c, name="fix")
    p2 = synthesize_policy(b, c, name="fix")
    assert p1.to_yaml() == p2.to_yaml()


def test_synthesize_clean_traces_produces_zero_rules(tmp_path: Path) -> None:
    """No regression -> no rules. The synthesizer must not invent
    constraints when the candidate matches the baseline."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _record_with_stop_reason(baseline, stop_reason="end_turn")
    _record_with_stop_reason(candidate, stop_reason="end_turn")
    b = _core.parse_agentlog(baseline.read_bytes())
    c = _core.parse_agentlog(candidate.read_bytes())
    policy = synthesize_policy(b, c, name="clean")
    assert policy.rules == []
    assert policy.diagnostics, "expected a 'no rules' diagnostic"


# ---------------------------------------------------------------------------
# Round-trip: every rule must be parseable
# ---------------------------------------------------------------------------


def test_every_emitted_rule_round_trips_through_load_policy() -> None:
    """A rule the synthesizer ships that fails `load_policy()` is a
    contract violation — it would crash any user who tries to apply
    the generated YAML."""
    b, c = _bundled_fixtures()
    policy = synthesize_policy(b, c, name="rt")
    if not policy.rules:
        pytest.skip("synthesizer produced no rules; nothing to round-trip")
    parsed = load_policy(policy.to_dict())
    assert len(parsed) == len(policy.rules)
    for raw, rule in zip(policy.rules, parsed, strict=True):
        assert raw["kind"] == rule.kind
        assert raw["id"] == rule.id


def test_emitted_yaml_parses_back_to_same_dict() -> None:
    """YAML serialisation must be lossless: load(dump(policy)) == policy."""
    b, c = _bundled_fixtures()
    policy = synthesize_policy(b, c, name="yaml")
    if not policy.rules:
        pytest.skip("synthesizer produced no rules; nothing to round-trip")
    rehydrated = yaml.safe_load(policy.to_yaml())
    assert rehydrated == policy.to_dict()


# ---------------------------------------------------------------------------
# Counterfactual gate
# ---------------------------------------------------------------------------


def test_verify_policy_passes_on_real_regression() -> None:
    """The whole point of the synthesizer: every shipped rule should
    fire on the candidate (catching the regression) and stay silent on
    the baseline (not over-fitting)."""
    b, c = _bundled_fixtures()
    policy = synthesize_policy(b, c, name="ver")
    if not policy.rules:
        pytest.skip("synthesizer produced no rules; nothing to verify")
    ok, reasons = verify_policy(policy, b, c)
    assert ok, "\n".join(reasons)


def test_verify_policy_rejects_empty_policy() -> None:
    """An empty rule list never 'catches' anything — verification must fail."""
    b, c = _bundled_fixtures()
    empty = SynthesizedPolicy()
    ok, reasons = verify_policy(empty, b, c)
    assert ok is False
    assert any("empty" in r for r in reasons)


def test_verify_policy_detects_overfit_rule(tmp_path: Path) -> None:
    """A hand-crafted rule that rejects baseline behaviour must trip
    the verify_policy gate. Guards against future changes silently
    accepting overfit rules."""
    b, c = _bundled_fixtures()
    bad_policy = SynthesizedPolicy(
        rules=[
            {
                "id": "overfit-test",
                "kind": "no_call",
                "params": {"tool": "search_files"},  # baseline calls this!
                "severity": "error",
                "description": "deliberately overfit for the test",
            }
        ]
    )
    ok, reasons = verify_policy(bad_policy, b, c)
    assert ok is False
    assert any("baseline" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_autopr_prints_yaml_to_stdout_by_default(tmp_path: Path) -> None:
    """`shadow autopr <baseline> <candidate>` with no --output prints
    the generated YAML to stdout — so callers can pipe it."""
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "b.agentlog"
    c_path = tmp_path / "c.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())

    result = runner.invoke(app, ["autopr", str(b_path), str(c_path)])
    assert result.exit_code == 0, result.output
    assert "apiVersion: shadow.dev/v1alpha1" in result.output


def test_cli_autopr_writes_to_file_when_output_passed(tmp_path: Path) -> None:
    """`--output FILE` writes the YAML to disk and reports what landed."""
    root = resources.files(_qs_data) / "fixtures"
    b_path = tmp_path / "b.agentlog"
    c_path = tmp_path / "c.agentlog"
    b_path.write_bytes(root.joinpath("baseline.agentlog").read_bytes())
    c_path.write_bytes(root.joinpath("candidate.agentlog").read_bytes())
    out = tmp_path / "regressions" / "policy.yaml"

    result = runner.invoke(app, ["autopr", str(b_path), str(c_path), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    parsed = yaml.safe_load(out.read_text())
    assert "rules" in parsed
    assert parsed["rules"], "wrote a policy with zero rules"


def test_cli_autopr_with_clean_traces_warns_no_rules_but_exits_zero(tmp_path: Path) -> None:
    """When the synthesizer finds nothing actionable, the command must
    still succeed (exit 0) with a clear warning. It's informational,
    not an error — the user asked us to try, we tried, nothing landed."""
    baseline = tmp_path / "b.agentlog"
    candidate = tmp_path / "c.agentlog"
    _record_with_stop_reason(baseline, stop_reason="end_turn")
    _record_with_stop_reason(candidate, stop_reason="end_turn")

    result = runner.invoke(app, ["autopr", str(baseline), str(candidate)])
    assert result.exit_code == 0, result.output
    assert "no rules synthesised" in result.output


def test_cli_autopr_missing_baseline_emits_hint(tmp_path: Path) -> None:
    """The shared `_fail()` hint for FileNotFoundError must surface so
    a fresh user knows the recovery path."""
    candidate = tmp_path / "c.agentlog"
    _record_with_stop_reason(candidate, stop_reason="end_turn")
    result = runner.invoke(app, ["autopr", str(tmp_path / "missing.agentlog"), str(candidate)])
    assert result.exit_code == 1
    # The hint added in app.py:_fail() points at `shadow demo` for first contact.
    assert "shadow demo" in result.output
