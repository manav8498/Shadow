"""Tests for the shadow.diagnose_pr.dashboard FastAPI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadow.diagnose_pr.dashboard import build_app, render_report_html


def _sample_report() -> dict:
    return {
        "schema_version": "diagnose-pr/v0.1",
        "verdict": "stop",
        "total_traces": 3,
        "affected_traces": 3,
        "blast_radius": 1.0,
        "dominant_cause": {
            "delta_id": "prompt.system",
            "axis": "trajectory",
            "ate": 0.6,
            "ci_low": 0.6,
            "ci_high": 0.6,
            "e_value": 6.7,
            "confidence": 1.0,
        },
        "top_causes": [
            {
                "delta_id": "prompt.system",
                "axis": "trajectory",
                "ate": 0.6,
                "ci_low": 0.6,
                "ci_high": 0.6,
                "e_value": 6.7,
                "confidence": 1.0,
            }
        ],
        "trace_diagnoses": [
            {
                "trace_id": "abc12345" * 4,
                "affected": True,
                "risk": 0.0,
                "worst_axis": "trajectory",
                "first_divergence": None,
                "policy_violations": [{"rule_id": "x"}],
            }
        ],
        "affected_trace_ids": ["abc12345" * 4],
        "new_policy_violations": 1,
        "worst_policy_rule": "confirm-before-refund",
        "suggested_fix": "Restore the prompt instruction.",
        "flags": ["low_power", "synthetic_mock"],
    }


def test_render_report_html_includes_verdict_and_cause() -> None:
    html = render_report_html(_sample_report(), source_path="/tmp/r.json")
    assert "STOP" in html
    assert "prompt.system" in html
    assert "confirm-before-refund" in html
    assert "+0.60" in html  # ATE
    assert "[0.60, 0.60]" in html  # CI
    assert "Restore the prompt instruction" in html


def test_render_report_html_handles_minimal_ship_report() -> None:
    minimal = {
        "schema_version": "diagnose-pr/v0.1",
        "verdict": "ship",
        "total_traces": 0,
        "affected_traces": 0,
        "blast_radius": 0.0,
        "dominant_cause": None,
        "top_causes": [],
        "trace_diagnoses": [],
        "affected_trace_ids": [],
        "new_policy_violations": 0,
        "worst_policy_rule": None,
        "suggested_fix": None,
        "flags": [],
    }
    html = render_report_html(minimal)
    assert "SHIP" in html
    assert "verdict-ship" in html
    # No dominant cause section when None
    assert "Dominant cause" not in html


def test_render_html_escapes_user_input() -> None:
    """Anything from the report that could carry user-controlled
    text (delta_id, fix text, policy rule) must be HTML-escaped."""
    nasty = _sample_report()
    nasty["dominant_cause"]["delta_id"] = "<script>alert(1)</script>"
    nasty["suggested_fix"] = "<img src=x onerror=alert(1)>"
    html = render_report_html(nasty)
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_dashboard_app_serves_html_at_root(tmp_path: Path) -> None:
    """Use FastAPI's TestClient to verify the route returns 200
    and the HTML contains the verdict."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_sample_report()))
    app = build_app(report_path)
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert "STOP" in res.text
    assert "text/html" in res.headers["content-type"]


def test_dashboard_serves_raw_json(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_sample_report()))
    app = build_app(report_path)
    client = TestClient(app)

    res = client.get("/report.json")
    assert res.status_code == 200
    parsed = res.json()
    assert parsed["verdict"] == "stop"


def test_dashboard_healthz(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    report_path = tmp_path / "r.json"
    report_path.write_text(json.dumps(_sample_report()))
    client = TestClient(build_app(report_path))
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_dashboard_404s_when_report_file_missing(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    missing = tmp_path / "does_not_exist.json"
    client = TestClient(build_app(missing))
    res = client.get("/")
    assert res.status_code == 404
