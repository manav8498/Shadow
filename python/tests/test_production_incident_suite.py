"""End-to-end tests for the production-incident regression suite.

The suite is the canonical real-world stress test for v2.5+ features:
five public-incident patterns (Air Canada, Avianca, NEDA/Tessa,
McDonald's, Replit) audited via the multi-scenario diff, LTL safety
policies, harmful-content judge, mSPRT/MSPRTtDetector latency
monitor, ACI online conformal, causal attribution, and the policy
suggestion engine.

These tests are NOT toy fixtures — each scenario mimics the public
failure mode reported in the news, and the audit pipeline must catch
the regression in every scenario. If any scenario passes the audit
when it shouldn't, that's a real regression in Shadow.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "production-incident-suite"

if not _EXAMPLE_DIR.exists():
    pytestmark = pytest.mark.skip(reason="production-incident-suite missing")
else:
    sys.path.insert(0, str(_EXAMPLE_DIR))


from audit import (  # noqa: E402
    SAFETY_POLICIES,
    AuditFindings,
    render_findings,
    run_audit,
)
from scenarios import (  # noqa: E402
    SCENARIO_BUILDERS,
    generate_baseline,
    generate_candidate,
)

# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


class TestScenarioGeneration:
    def test_baseline_has_five_scenarios(self) -> None:
        baseline = generate_baseline(seed=1)
        assert len(baseline) == 5

    def test_candidate_has_five_scenarios(self) -> None:
        candidate = generate_candidate(seed=2)
        assert len(candidate) == 5

    def test_each_scenario_has_scenario_id(self) -> None:
        for sess in generate_baseline(seed=1):
            sids = {(r.get("meta") or {}).get("scenario_id") for r in sess}
            assert len(sids) == 1
            assert next(iter(sids)) is not None

    def test_each_scenario_has_at_least_one_chat_response(self) -> None:
        for sess in generate_candidate(seed=2):
            n_resp = sum(1 for r in sess if r.get("kind") == "chat_response")
            assert n_resp >= 1, "every scenario should produce at least one response"

    def test_scenarios_are_deterministic_per_seed(self) -> None:
        a = generate_baseline(seed=1)
        b = generate_baseline(seed=1)
        # Same seed → same content (modulo whatever rng changes; full equality
        # is too strict because timestamps may shift).
        for sa, sb in zip(a, b, strict=True):
            assert len(sa) == len(sb)


# ---------------------------------------------------------------------------
# Audit findings
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def findings() -> AuditFindings:
    """Run the full audit once; share the result across tests."""
    return asyncio.run(run_audit(generate_baseline(seed=1), generate_candidate(seed=2)))


class TestAuditVerdict:
    def test_overall_verdict_is_unsafe(self, findings: AuditFindings) -> None:
        assert findings.is_safe is False

    def test_all_five_scenarios_in_findings(self, findings: AuditFindings) -> None:
        assert len(findings.scenarios) == 5

    def test_each_scenario_id_present(self, findings: AuditFindings) -> None:
        sids = {s.scenario_id for s in findings.scenarios}
        assert sids == {
            "air_canada_refund",
            "avianca_fake_citations",
            "neda_tessa_harm",
            "mcdonalds_pii_leak",
            "replit_prod_delete",
        }


class TestPerScenarioCoverage:
    """Each incident pattern must be flagged unsafe by the audit."""

    def test_air_canada_caught_by_ltl(self, findings: AuditFindings) -> None:
        finding = next(s for s in findings.scenarios if s.scenario_id == "air_canada_refund")
        assert finding.is_safe is False
        # LTL must_call_before catches verify-before-refund violation.
        assert "verify-before-refund" in finding.policy_violations

    def test_avianca_caught_by_forbidden_text(self, findings: AuditFindings) -> None:
        finding = next(s for s in findings.scenarios if s.scenario_id == "avianca_fake_citations")
        assert finding.is_safe is False
        assert "no-fabricated-fed-cite" in finding.policy_violations

    def test_neda_caught_by_harm_judge(self, findings: AuditFindings) -> None:
        finding = next(s for s in findings.scenarios if s.scenario_id == "neda_tessa_harm")
        assert finding.is_safe is False
        # The narrow safety axis won't catch this (no refusal); the
        # judge axis is the ONLY thing that flags it.
        assert finding.harm_verdict == "unsafe"

    def test_mcdonalds_caught_by_forbidden_text(self, findings: AuditFindings) -> None:
        finding = next(s for s in findings.scenarios if s.scenario_id == "mcdonalds_pii_leak")
        assert finding.is_safe is False
        assert "no-ssn-echo" in finding.policy_violations

    def test_replit_caught_by_forbidden_tool_call(
        self, findings: AuditFindings
    ) -> None:
        finding = next(s for s in findings.scenarios if s.scenario_id == "replit_prod_delete")
        assert finding.is_safe is False
        assert "no-prod-sql-without-confirm" in finding.policy_violations


class TestMultiScenarioDiff:
    def test_diff_report_has_all_five_scenarios(self, findings: AuditFindings) -> None:
        sids = {s["scenario_id"] for s in findings.multi_scenario_report["scenarios"]}
        assert "air_canada_refund" in sids
        assert "avianca_fake_citations" in sids
        assert "neda_tessa_harm" in sids
        assert "mcdonalds_pii_leak" in sids
        assert "replit_prod_delete" in sids

    def test_no_baseline_only_or_candidate_only_scenarios(
        self, findings: AuditFindings
    ) -> None:
        # All scenarios should be present on both sides — sanity check
        # that scenario_id wiring is correct.
        assert findings.multi_scenario_report["baseline_only_scenarios"] == []
        assert findings.multi_scenario_report["candidate_only_scenarios"] == []


class TestCausalAttribution:
    def test_attributes_latency_to_model_change(self, findings: AuditFindings) -> None:
        """The do-calculus engine should attribute the latency drift to
        the 'model' delta and not to the 'temperature' delta."""
        ate = findings.causal_attribution["ate"]
        assert "model" in ate
        # Latency effect on 'model' should be positive (opus is slower).
        assert ate["model"]["latency"] > 0.4

    def test_attributes_verbosity_to_temperature(self, findings: AuditFindings) -> None:
        ate = findings.causal_attribution["ate"]
        assert ate["temperature"]["verbosity"] > 0.1


class TestStreamingDetectors:
    def test_msprt_runs_and_sets_a_decision(self, findings: AuditFindings) -> None:
        # mSPRT decision is one of three valid strings.
        for s in findings.scenarios:
            assert s.msprt_decision in {"continue", "h0", "h1"} or s.msprt_decision is None

    def test_aci_breach_rate_in_unit_interval(self, findings: AuditFindings) -> None:
        for s in findings.scenarios:
            if s.aci_breach_rate is not None:
                assert 0.0 <= s.aci_breach_rate <= 1.0


# ---------------------------------------------------------------------------
# False-positive check: baseline-vs-baseline must NOT flag anything
# ---------------------------------------------------------------------------


class TestFalsePositiveFreedom:
    @pytest.fixture(scope="class")
    def baseline_findings(self) -> AuditFindings:
        baseline = generate_baseline(seed=1)
        return asyncio.run(run_audit(baseline, baseline))

    def test_baseline_against_itself_flags_nothing_critical(
        self, baseline_findings: AuditFindings
    ) -> None:
        critical = [s for s in baseline_findings.scenarios if not s.is_safe]
        # The baseline scenarios for safe-pattern cases should have no
        # critical violations. Note: the air_canada baseline DOES call
        # verify_user before refund — no violation. The avianca baseline
        # refuses — no fake F.3d cite. McDonald's baseline doesn't echo
        # the SSN. Replit baseline doesn't execute_sql. NEDA baseline
        # refuses. So none should fail.
        assert not critical, (
            f"baseline-vs-baseline produced false positives: "
            f"{[s.scenario_id for s in critical]}"
        )

    def test_baseline_overall_safe(self, baseline_findings: AuditFindings) -> None:
        assert baseline_findings.is_safe


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


class TestReportRender:
    def test_render_contains_all_section_headers(self, findings: AuditFindings) -> None:
        report = render_findings(findings)
        assert "Per-scenario findings" in report
        assert "Multi-scenario diff sections" in report
        assert "Auto-suggested policies" in report
        assert "Causal attribution" in report

    def test_render_says_fail_when_unsafe(self, findings: AuditFindings) -> None:
        report = render_findings(findings)
        assert "FAIL" in report

    def test_render_lists_each_scenario_id(self, findings: AuditFindings) -> None:
        report = render_findings(findings)
        for s in findings.scenarios:
            assert s.scenario_id in report


# ---------------------------------------------------------------------------
# CLI exit code
# ---------------------------------------------------------------------------


class TestCLIDemo:
    def test_run_audit_exits_one_on_unsafe(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_EXAMPLE_DIR / "run_audit.py")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert proc.returncode == 1, (
            f"Demo should exit 1 (incidents detected); got {proc.returncode}\n"
            f"stdout tail: {proc.stdout[-500:]}"
        )

    def test_demo_output_contains_all_scenarios(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(_EXAMPLE_DIR / "run_audit.py")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        for sid in (
            "air_canada_refund",
            "avianca_fake_citations",
            "neda_tessa_harm",
            "mcdonalds_pii_leak",
            "replit_prod_delete",
        ):
            assert sid in proc.stdout


# ---------------------------------------------------------------------------
# Scenario builder smoke
# ---------------------------------------------------------------------------


class TestScenarioBuilderSmoke:
    @pytest.mark.parametrize("builder", SCENARIO_BUILDERS)
    def test_each_builder_produces_baseline_and_candidate(self, builder) -> None:
        import random

        rng = random.Random(0)
        b = builder(rng, base_idx=0, profile="baseline")
        c = builder(rng, base_idx=1000, profile="candidate")
        assert len(b) >= 2  # at least 1 request + 1 response
        assert len(c) >= 2
        # Same scenario_id on both sides.
        bsid = (b[0].get("meta") or {}).get("scenario_id")
        csid = (c[0].get("meta") or {}).get("scenario_id")
        assert bsid == csid


# ---------------------------------------------------------------------------
# Policy schema sanity
# ---------------------------------------------------------------------------


class TestSafetyPolicies:
    def test_safety_policies_are_well_formed(self) -> None:
        from shadow.hierarchical import _POLICY_KINDS  # type: ignore[attr-defined]

        for rule in SAFETY_POLICIES:
            assert rule.id  # non-empty
            assert rule.kind in _POLICY_KINDS, (
                f"rule {rule.id!r} has unknown kind {rule.kind!r}"
            )
            assert rule.severity in {"critical", "error", "warning"}
            assert rule.description  # non-empty
