"""End-to-end tests for OTel ↔ diagnose-pr (Phase 5 acceptance).

Two contracts pinned:

  1. Round-trip preservation: native -> OTel -> native gives the
     same per-pair 9-axis diff outcome (severities + first_divergence
     match). Regression test for the message/content/latency loss
     that the Phase 5 audit found.

  2. Imported OTel traces produce the same diagnose-pr verdict +
     dominant cause + affected count as the native traces. Regression
     test for the trace_id collision (OTel-imported traces all
     shared the metadata content hash before Phase 5 stamped envelope
     `meta.trace_id` from the OTel `traceId`).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from shadow import _core
from shadow.cli.app import app
from shadow.diagnose_pr.diffing import diff_pair
from shadow.importers.otel import otel_to_agentlog
from shadow.otel import agentlog_to_otel

_DEMO = Path(__file__).resolve().parent.parent.parent / "examples" / "refund-causal-diagnosis"


def test_otel_roundtrip_preserves_per_pair_diff_outcome() -> None:
    """Native diff and OTel-roundtripped diff must produce identical
    per-axis severities + matching first_divergence on the refund demo."""
    nb = _core.parse_agentlog((_DEMO / "baseline_traces" / "s1.agentlog").read_bytes())
    nc = _core.parse_agentlog((_DEMO / "candidate_traces" / "s1.agentlog").read_bytes())
    rb = otel_to_agentlog(agentlog_to_otel(nb))
    rc = otel_to_agentlog(agentlog_to_otel(nc))

    native = diff_pair(nb, nc)
    rt = diff_pair(rb, rc)

    native_axes = {row["axis"]: row["severity"] for row in native["rows"]}
    rt_axes = {row["axis"]: row["severity"] for row in rt["rows"]}
    assert native_axes == rt_axes, (
        f"OTel round-trip changed per-axis severities. "
        f"native={native_axes} vs roundtripped={rt_axes}"
    )
    assert (native.get("first_divergence") is None) == (rt.get("first_divergence") is None)
    if native.get("first_divergence") is not None:
        assert native["first_divergence"]["primary_axis"] == rt["first_divergence"]["primary_axis"]


def test_otel_imported_traces_have_unique_trace_ids(tmp_path: Path) -> None:
    """Regression: pre-Phase-5, the OTel importer didn't write
    envelope meta.trace_id, so multiple OTel-imported traces with
    byte-identical metadata payloads collapsed to one trace_id (the
    metadata content hash). Ensures the OTel traceId is now stamped
    into envelope meta.trace_id so each trace stays distinct."""
    from shadow.diagnose_pr.loaders import load_traces

    runner = CliRunner()
    side_dir = tmp_path / "baseline"
    side_dir.mkdir()
    # Convert all 3 demo baseline traces to OTel JSON, then re-import.
    for f in (_DEMO / "baseline_traces").glob("*.agentlog"):
        records = _core.parse_agentlog(f.read_bytes())
        otel = agentlog_to_otel(records)
        json_path = side_dir / (f.stem + ".json")
        json_path.write_text(json.dumps(otel))
        agentlog_path = side_dir / (f.stem + ".agentlog")
        result = runner.invoke(
            app,
            [
                "import",
                "--format",
                "otel-genai",
                str(json_path),
                "--output",
                str(agentlog_path),
            ],
        )
        assert result.exit_code == 0, result.stdout
        json_path.unlink()  # leave only the .agentlog files for load_traces

    loaded = load_traces([side_dir])
    assert len(loaded) == 3
    unique_ids = {t.trace_id for t in loaded}
    assert len(unique_ids) == 3, (
        f"3 OTel-imported traces collapsed to {len(unique_ids)} unique trace_id(s); "
        "envelope meta.trace_id is not being stamped from OTel traceId."
    )


def test_otel_imported_corpus_diagnose_pr_matches_native_verdict(tmp_path: Path) -> None:
    """The Phase 5 acceptance: 'imported trace can be used by
    diagnose-pr'. Stronger contract: same verdict + same affected
    count as the native run."""
    runner = CliRunner()

    # Native run
    native_out = tmp_path / "native.json"
    res_n = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(_DEMO / "baseline_traces"),
            "--candidate-traces",
            str(_DEMO / "candidate_traces"),
            "--baseline-config",
            str(_DEMO / "baseline.yaml"),
            "--candidate-config",
            str(_DEMO / "candidate.yaml"),
            "--policy",
            str(_DEMO / "policy.yaml"),
            "--out",
            str(native_out),
            "--backend",
            "mock",
        ],
    )
    assert res_n.exit_code == 0, res_n.stdout
    native = json.loads(native_out.read_text())

    # OTel-roundtripped corpus
    otel_dir = tmp_path / "otel"
    for side in ("baseline_traces", "candidate_traces"):
        sd = otel_dir / side
        sd.mkdir(parents=True, exist_ok=True)
        for f in (_DEMO / side).glob("*.agentlog"):
            records = _core.parse_agentlog(f.read_bytes())
            otel = agentlog_to_otel(records)
            jp = sd / (f.stem + ".json")
            jp.write_text(json.dumps(otel))
            ap = sd / (f.stem + ".agentlog")
            result = runner.invoke(
                app,
                ["import", "--format", "otel-genai", str(jp), "--output", str(ap)],
            )
            assert result.exit_code == 0
            jp.unlink()

    rt_out = tmp_path / "rt.json"
    res_rt = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(otel_dir / "baseline_traces"),
            "--candidate-traces",
            str(otel_dir / "candidate_traces"),
            "--baseline-config",
            str(_DEMO / "baseline.yaml"),
            "--candidate-config",
            str(_DEMO / "candidate.yaml"),
            "--policy",
            str(_DEMO / "policy.yaml"),
            "--out",
            str(rt_out),
            "--backend",
            "mock",
        ],
    )
    assert res_rt.exit_code == 0, res_rt.stdout
    rt = json.loads(rt_out.read_text())

    assert native["verdict"] == rt["verdict"], (
        f"OTel-roundtripped corpus produced verdict={rt['verdict']} "
        f"vs native={native['verdict']}"
    )
    assert native["affected_traces"] == rt["affected_traces"], (
        f"OTel-roundtripped corpus produced affected={rt['affected_traces']} "
        f"vs native={native['affected_traces']}"
    )
    assert native["total_traces"] == rt["total_traces"]


def test_export_otel_genai_alias_works(tmp_path: Path) -> None:
    """The spec calls for `shadow export --format otel-genai`; that
    must be a recognized alias for `--format otel`."""
    runner = CliRunner()
    out = tmp_path / "out.json"
    result = runner.invoke(
        app,
        [
            "export",
            str(_DEMO / "baseline_traces" / "s1.agentlog"),
            "--format",
            "otel-genai",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    data = json.loads(out.read_text())
    assert "resourceSpans" in data
