"""Tests for the long-form TF-IDF semantic-axis hint.

External-review finding (mysterymanOO7 retest): comparing GPT-4.1 vs
GPT-5 long-form research outputs with the default TF-IDF semantic
backend over-alarmed at "severe" even after correct task alignment;
re-running with `--semantic` embeddings dropped it to "moderate". The
fix is to surface a hint in both terminal + markdown renderers when:
1. the report was produced WITHOUT the embeddings backend, AND
2. the semantic axis flagged moderate/severe, AND
3. the response_length axis median suggests long-form output.

These tests lock in that the hint fires under those exact conditions
and stays quiet otherwise.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

from rich.console import Console

from shadow.report.markdown import render_markdown
from shadow.report.terminal import render_terminal


def _row(axis: str, **kw: Any) -> dict[str, Any]:
    base = {
        "axis": axis,
        "baseline_median": 0.0,
        "candidate_median": 0.0,
        "delta": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "severity": "none",
        "n": 10,
        "flags": [],
    }
    base.update(kw)
    return base


def _long_form_severe_semantic_report(
    semantic_backend: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": [
            _row("semantic", severity="severe", baseline_median=1.0, candidate_median=0.4),
            _row("response_length", baseline_median=800.0, candidate_median=950.0),
            _row("trajectory"),
        ],
        "baseline_trace_id": "sha256:aaa",
        "candidate_trace_id": "sha256:bbb",
        "pair_count": 10,
    }
    if semantic_backend is not None:
        report["semantic_backend"] = semantic_backend
    return report


def _short_response_severe_semantic_report() -> dict[str, Any]:
    return {
        "rows": [
            _row("semantic", severity="severe", baseline_median=1.0, candidate_median=0.4),
            _row("response_length", baseline_median=40.0, candidate_median=42.0),
            _row("trajectory"),
        ],
        "baseline_trace_id": "sha256:aaa",
        "candidate_trace_id": "sha256:bbb",
        "pair_count": 10,
    }


def test_terminal_hints_at_embeddings_when_long_form_tfidf_alarms() -> None:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    render_terminal(_long_form_severe_semantic_report(), console=console)
    out = buf.getvalue()
    assert "--semantic" in out
    assert "long-form" in out.lower() or "paraphrase" in out.lower()


def test_terminal_silent_when_embeddings_already_used() -> None:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    render_terminal(
        _long_form_severe_semantic_report(semantic_backend="embeddings"),
        console=console,
    )
    out = buf.getvalue()
    assert "--semantic" not in out


def test_terminal_silent_on_short_responses_even_when_severe() -> None:
    """Severe semantic on short responses is more likely to be a real
    regression (vocab divergence on short text is harder to fake). The
    embeddings hint would only add noise — keep it quiet.
    """
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    render_terminal(_short_response_severe_semantic_report(), console=console)
    out = buf.getvalue()
    assert "long-form" not in out.lower()


def test_markdown_hints_at_embeddings_when_long_form_tfidf_alarms() -> None:
    md = render_markdown(_long_form_severe_semantic_report())
    assert "--semantic" in md
    assert "shadow-diff[embeddings]" in md


def test_markdown_silent_when_embeddings_already_used() -> None:
    md = render_markdown(_long_form_severe_semantic_report(semantic_backend="embeddings"))
    assert "shadow-diff[embeddings]" not in md
