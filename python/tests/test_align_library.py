"""Tests for the `shadow.align` standalone library (Phase 6).

Pin the public API contract: 5 functions, stable dataclasses, no
hidden coupling to the rest of `shadow`. External tools rely on
this surface.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Public-API import contract
# ---------------------------------------------------------------------------


def test_public_api_imports_from_top_level() -> None:
    """The five spec'd functions + their dataclasses must be
    importable directly from `shadow.align`."""
    from shadow.align import (
        AlignedTurn,
        Alignment,
        ArgDelta,
        Divergence,
        align_traces,
        first_divergence,
        tool_arg_delta,
        top_k_divergences,
        trajectory_distance,
    )

    # Smoke: types are real, not None
    assert callable(align_traces)
    assert callable(first_divergence)
    assert callable(top_k_divergences)
    assert callable(trajectory_distance)
    assert callable(tool_arg_delta)
    assert AlignedTurn is not None and Alignment is not None
    assert ArgDelta is not None and Divergence is not None


# ---------------------------------------------------------------------------
# trajectory_distance — pure Python, no deps
# ---------------------------------------------------------------------------


class TestTrajectoryDistance:
    def test_equal_sequences_return_zero(self) -> None:
        from shadow.align import trajectory_distance

        assert trajectory_distance(["a", "b", "c"], ["a", "b", "c"]) == 0.0

    def test_completely_disjoint_sequences_return_one(self) -> None:
        from shadow.align import trajectory_distance

        assert trajectory_distance(["a", "b"], ["x", "y"]) == 1.0

    def test_one_substitution_in_three_is_one_third(self) -> None:
        from shadow.align import trajectory_distance

        # Levenshtein dist 1 / max(3,3) = 0.333...
        d = trajectory_distance(["a", "b", "c"], ["a", "x", "c"])
        assert d == pytest.approx(1 / 3, abs=0.01)

    def test_extra_tool_at_end_counted_as_one_insertion(self) -> None:
        from shadow.align import trajectory_distance

        d = trajectory_distance(["a", "b"], ["a", "b", "c"])
        assert d == pytest.approx(1 / 3, abs=0.01)

    def test_both_empty_returns_zero(self) -> None:
        from shadow.align import trajectory_distance

        assert trajectory_distance([], []) == 0.0

    def test_one_empty_returns_one(self) -> None:
        from shadow.align import trajectory_distance

        assert trajectory_distance(["a"], []) == 1.0


# ---------------------------------------------------------------------------
# tool_arg_delta — pure Python structural diff
# ---------------------------------------------------------------------------


class TestToolArgDelta:
    def test_equal_dicts_no_deltas(self) -> None:
        from shadow.align import tool_arg_delta

        assert tool_arg_delta({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []

    def test_added_key(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta({"a": 1}, {"a": 1, "b": 2})
        assert len(deltas) == 1
        assert deltas[0].path == "/b"
        assert deltas[0].kind == "added"
        assert deltas[0].new == 2

    def test_removed_key(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta({"a": 1, "b": 2}, {"a": 1})
        assert deltas[0].kind == "removed"
        assert deltas[0].old == 2

    def test_changed_value(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta({"x": 1}, {"x": 2})
        assert deltas[0].kind == "changed"
        assert deltas[0].old == 1
        assert deltas[0].new == 2

    def test_type_changed(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta({"x": 1}, {"x": "1"})
        assert deltas[0].kind == "type_changed"

    def test_nested_dict_changes_get_nested_paths(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta(
            {"outer": {"inner": "old"}},
            {"outer": {"inner": "new"}},
        )
        assert len(deltas) == 1
        assert deltas[0].path == "/outer/inner"
        assert deltas[0].kind == "changed"

    def test_list_with_appended_item(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta({"items": [1, 2]}, {"items": [1, 2, 3]})
        assert len(deltas) == 1
        assert deltas[0].path == "/items/2"
        assert deltas[0].kind == "added"
        assert deltas[0].new == 3

    def test_list_index_change(self) -> None:
        from shadow.align import tool_arg_delta

        deltas = tool_arg_delta([1, 2, 3], [1, 9, 3])
        assert len(deltas) == 1
        assert deltas[0].path == "/1"
        assert deltas[0].old == 2 and deltas[0].new == 9


# ---------------------------------------------------------------------------
# first_divergence + top_k_divergences + align_traces — Rust-backed
# ---------------------------------------------------------------------------


def _quickstart_pair() -> tuple[list, list]:
    from importlib import resources

    import shadow.quickstart_data as q
    from shadow import _core

    root = resources.files(q) / "fixtures"
    base = _core.parse_agentlog(root.joinpath("baseline.agentlog").read_bytes())
    cand = _core.parse_agentlog(root.joinpath("candidate.agentlog").read_bytes())
    return base, cand


class TestFirstDivergence:
    def test_identical_traces_return_none(self) -> None:
        from shadow.align import first_divergence

        b, _ = _quickstart_pair()
        # Pass baseline as both sides.
        assert first_divergence(b, b) is None

    def test_real_pair_returns_divergence_with_required_fields(self) -> None:
        from shadow.align import Divergence, first_divergence

        b, c = _quickstart_pair()
        fd = first_divergence(b, c)
        assert fd is not None
        assert isinstance(fd, Divergence)
        assert fd.kind  # non-empty string
        assert fd.primary_axis
        assert fd.explanation
        assert 0.0 <= fd.confidence <= 1.0


class TestTopKDivergences:
    def test_identical_returns_empty(self) -> None:
        from shadow.align import top_k_divergences

        b, _ = _quickstart_pair()
        assert top_k_divergences(b, b) == []

    def test_real_pair_returns_at_most_k(self) -> None:
        from shadow.align import top_k_divergences

        b, c = _quickstart_pair()
        out = top_k_divergences(b, c, k=3)
        assert len(out) <= 3

    def test_k_zero_or_negative_rejected(self) -> None:
        from shadow.align import top_k_divergences

        b, c = _quickstart_pair()
        with pytest.raises(ValueError):
            top_k_divergences(b, c, k=0)


class TestAlignTraces:
    def test_real_pair_returns_alignment_with_paired_turns(self) -> None:
        from shadow.align import Alignment, align_traces

        b, c = _quickstart_pair()
        aln = align_traces(b, c)
        assert isinstance(aln, Alignment)
        assert aln.total_cost >= 0
        # At least one paired turn for a non-empty trace.
        assert len(aln.turns) >= 1


# ---------------------------------------------------------------------------
# Type-shape sanity — these check that bad inputs raise instead of
# silently producing meaningless output.
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_list_baseline_raises_type_error(self) -> None:
        from shadow.align import first_divergence

        with pytest.raises(TypeError, match="baseline must be a list"):
            first_divergence("not a list", [])  # type: ignore[arg-type]

    def test_non_list_candidate_raises_type_error(self) -> None:
        from shadow.align import top_k_divergences

        with pytest.raises(TypeError, match="candidate must be a list"):
            top_k_divergences([], 42)  # type: ignore[arg-type]
