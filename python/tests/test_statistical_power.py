"""Tests for the tiered statistical-power classifier.

External-review-driven: a fourth reviewer pass flagged that the
binary `low_statistical_power` flag from v3.2.2 only fired at n<5,
which is the cliff edge. Most reviewers expect tiered guidance
across n<5 / n<30 / n<100 boundaries.
"""

from __future__ import annotations

import pytest

from shadow.report.statistical_power import (
    classify_power,
    power_blurb,
    recommended_sample_size,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "low"),
        (1, "low"),
        (4, "low"),
        (5, "moderate"),
        (10, "moderate"),
        (29, "moderate"),
        (30, "adequate"),
        (50, "adequate"),
        (99, "adequate"),
        (100, "robust"),
        (500, "robust"),
        (10_000, "robust"),
    ],
)
def test_classify_power_boundaries(n: int, expected: str) -> None:
    """Pin the exact thresholds. Off-by-one on a tier boundary would
    silently shift every report's classification."""
    assert classify_power(n) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 30),
        (4, 30),
        (5, 100),
        (29, 100),
        (30, 100),  # still recommend 100 even when adequate
        (99, 100),
        (100, None),
        (1_000, None),
    ],
)
def test_recommended_sample_size_returns_next_tier(n: int, expected: int | None) -> None:
    """`recommended_sample_size` returns the target the user should
    aim for to hit the next tier. Returns None once they're robust."""
    assert recommended_sample_size(n) == expected


def test_power_blurb_low_mentions_directional_and_target() -> None:
    out = power_blurb(3)
    assert "low" in out
    assert "directional" in out.lower()
    assert "30" in out  # recommended next-tier target


def test_power_blurb_moderate_mentions_unstable() -> None:
    out = power_blurb(15)
    assert "moderate" in out
    assert "unstable" in out.lower()
    assert "100" in out


def test_power_blurb_adequate_recommends_100_for_enterprise() -> None:
    out = power_blurb(50)
    assert "adequate" in out
    assert "100" in out


def test_power_blurb_robust_carries_no_recommendation_clause() -> None:
    out = power_blurb(500)
    assert "robust" in out
    # No "record X more" / "recommends Y+" language at the robust tier.
    assert "+" not in out
