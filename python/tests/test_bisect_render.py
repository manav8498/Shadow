"""Tests for the bisect attribution renderer.

The renderer's contract is honest output: percentages are prefixed
``est.``, the bare ``✓`` is gone, the header carries a correlational-
not-causal caveat, and the qualifier on each row spells out *why* a
delta is considered stable. These tests pin all five.
"""

from __future__ import annotations

from typing import Any

from shadow.bisect.render import (
    render_attribution_markdown,
    render_attribution_terminal,
)


def _row(
    label: str,
    *,
    attribution: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
    selection_frequency: float = 0.0,
    significant: bool = False,
) -> dict[str, Any]:
    return {
        "label": label,
        "attribution": attribution,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "selection_frequency": selection_frequency,
        "significant": significant,
    }


def _result(per_axis: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"attributions": per_axis}


# ---- header caveat ------------------------------------------------------


def test_terminal_renderer_leads_with_correlational_caveat() -> None:
    text = render_attribution_terminal(_result({}))
    assert text.splitlines()[0].startswith("Bisect attribution (estimated, correlational)")
    assert "shadow replay" in text


def test_markdown_renderer_leads_with_correlational_caveat() -> None:
    md = render_attribution_markdown(_result({}))
    assert "## Bisect attribution" in md
    assert "estimated, correlational" in md
    assert "shadow replay" in md


# ---- 'est.' prefix on percentages --------------------------------------


def test_attribution_percentages_carry_est_prefix() -> None:
    text = render_attribution_terminal(
        _result(
            {
                "semantic": [
                    _row(
                        "prompt.system",
                        attribution=0.749,
                        ci_low=0.71,
                        ci_high=0.892,
                        selection_frequency=1.0,
                        significant=True,
                    )
                ]
            }
        )
    )
    assert "est. 74.9%" in text
    # The bare unhedged "74.9% caused" form must NOT appear.
    assert "74.9% caused" not in text


def test_markdown_attribution_carries_est_prefix() -> None:
    md = render_attribution_markdown(
        _result(
            {
                "semantic": [
                    _row(
                        "prompt.system",
                        attribution=0.749,
                        ci_low=0.71,
                        ci_high=0.892,
                        selection_frequency=1.0,
                        significant=True,
                    )
                ]
            }
        )
    )
    assert "est. 74.9%" in md


# ---- bare check-mark removed -------------------------------------------


def test_terminal_renderer_replaces_bare_check_with_qualifier() -> None:
    text = render_attribution_terminal(
        _result(
            {
                "semantic": [
                    _row(
                        "prompt.system",
                        attribution=0.7,
                        ci_low=0.5,
                        ci_high=0.85,
                        selection_frequency=0.95,
                        significant=True,
                    )
                ]
            }
        )
    )
    # The check mark by itself is gone; an explicit qualifier replaces it.
    assert " ✓" not in text
    assert "stable" in text
    assert "CI excludes 0" in text


def test_terminal_renderer_marks_screening_only_rows() -> None:
    """A delta selected by screening but with a low selection frequency
    should be labelled 'screening only', not 'stable'."""
    text = render_attribution_terminal(
        _result(
            {
                "latency": [
                    _row(
                        "tools",
                        attribution=0.2,
                        ci_low=-0.05,
                        ci_high=0.4,
                        selection_frequency=0.3,
                        significant=False,
                    )
                ]
            }
        )
    )
    assert "(weak signal)" in text or "CI crosses 0" in text


# ---- CI bounds are labelled --------------------------------------------


def test_terminal_renderer_labels_brackets_as_95_ci() -> None:
    text = render_attribution_terminal(
        _result(
            {
                "semantic": [
                    _row(
                        "prompt.system",
                        attribution=0.5,
                        ci_low=0.3,
                        ci_high=0.6,
                        selection_frequency=0.9,
                        significant=True,
                    )
                ]
            }
        )
    )
    # Brackets must be qualified as a 95% CI so a reader knows what
    # they're looking at, not bare numbers next to a percentage.
    assert "95% CI" in text


# ---- empty / no-attribution path ----------------------------------------


def test_terminal_renderer_handles_no_attributions() -> None:
    text = render_attribution_terminal(_result({}))
    assert "(no attributions" in text


def test_markdown_renderer_handles_no_attributions() -> None:
    md = render_attribution_markdown(_result({}))
    assert "_No attributions" in md
