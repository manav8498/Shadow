"""Tests for shadow.bisect (Phase 4)."""

from __future__ import annotations

import numpy as np

from shadow.bisect import Delta, diff_configs, full_factorial, plackett_burman
from shadow.bisect.attribution import AXIS_NAMES, rank_attributions
from shadow.bisect.design import is_orthogonal
from shadow.bisect.runner import choose_design, score_corners_synthetic


def test_diff_configs_finds_leaf_differences() -> None:
    a = {"model": "claude-opus-4-7", "params": {"temperature": 0.2}}
    b = {"model": "claude-opus-4-7", "params": {"temperature": 0.7, "top_p": 0.9}}
    deltas = diff_configs(a, b)
    # Two differing leaves: params.temperature and params.top_p.
    paths = {d.path for d in deltas}
    assert paths == {"params.temperature", "params.top_p"}
    by_path = {d.path: d for d in deltas}
    assert by_path["params.temperature"].old_value == 0.2
    assert by_path["params.temperature"].new_value == 0.7
    assert by_path["params.top_p"].old_value is None  # added in b
    assert by_path["params.top_p"].new_value == 0.9


def test_diff_configs_identical_returns_empty() -> None:
    a = {"model": "x", "params": {"t": 0.2}}
    assert diff_configs(a, a) == []


def test_diff_configs_coalesces_tool_schema_changes_to_one_delta_per_tool() -> None:
    """REGRESSION TEST: the v0.1 walker produced ~50 atomic deltas for a
    typical 4-tool config edit because every schema-leaf counted as one
    delta. This made LASSO-over-corners infeasible (too many features,
    too few runs). The coalescer collapses to one delta per tool-NAME."""
    a = {
        "tools": [
            {
                "name": "fetch",
                "description": "Get a thing",
                "input_schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
            {
                "name": "verify",
                "description": "Double-check a thing",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
        ]
    }
    # b: deletes `verify` entirely, AND edits `fetch`'s description + schema.
    b = {
        "tools": [
            {
                "name": "fetch",
                "description": "Fetch a thing",  # changed
                "input_schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["id"],
                },
            },
        ]
    }
    deltas = diff_configs(a, b)
    paths = sorted(d.path for d in deltas)
    # Exactly 2 deltas: one per affected tool-name.
    assert paths == ["tools.fetch", "tools.verify"]
    by_path = {d.path: d for d in deltas}
    # fetch was modified
    assert by_path["tools.fetch"].old_value != by_path["tools.fetch"].new_value
    # verify was deleted
    assert by_path["tools.verify"].new_value is None
    # Legacy mode still exposes the leaf-level explosion for debugging.
    leaf_deltas = diff_configs(a, b, coalesce=False)
    assert len(leaf_deltas) > len(deltas)


def test_diff_configs_coalesced_kind_is_tools() -> None:
    a = {"tools": [{"name": "x", "description": "a"}]}
    b = {"tools": [{"name": "x", "description": "b"}]}
    [d] = diff_configs(a, b)
    assert d.path == "tools.x"
    assert d.kind == "tools"


def test_bisect_attribution_schema_is_uniform_across_modes(tmp_path) -> None:
    """REGRESSION TEST: heuristic and placeholder modes used to emit
    different attribution row shapes than live-replay mode, causing
    downstream KeyErrors. Every mode must now emit the SAME keys."""
    import yaml

    from shadow import _core
    from shadow.bisect.runner import run_bisect

    expected_keys = {
        "delta",
        "weight",
        "ci95_low",
        "ci95_high",
        "significant",
        "selection_frequency",
    }

    a_cfg = {
        "model_id": "claude-sonnet-4-6",
        "params": {"temperature": 0.0},
        "prompt": {"system": "careful"},
        "tools": [{"name": "x", "description": "a"}],
    }
    b_cfg = {
        "model_id": "claude-sonnet-4-6",
        "params": {"temperature": 0.5},
        "prompt": {"system": "quick"},
        "tools": [{"name": "x", "description": "b"}],
    }
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(a_cfg))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(b_cfg))

    # Empty agentlog files (just metadata) for both traces.
    empty = _core.write_agentlog(
        [
            {
                "version": _core.SPEC_VERSION,
                "id": _core.content_id({"sdk": {"name": "shadow"}}),
                "kind": "metadata",
                "ts": "2026-04-23T00:00:00Z",
                "parent": None,
                "payload": {"sdk": {"name": "shadow"}},
                "meta": {"trace_id": "t"},
            }
        ]
    )
    (tmp_path / "a.agentlog").write_bytes(empty)
    (tmp_path / "b.agentlog").write_bytes(empty)

    # Heuristic mode (candidate_traces supplied, no backend).
    heu = run_bisect(
        tmp_path / "a.yaml",
        tmp_path / "b.yaml",
        tmp_path / "a.agentlog",
        candidate_traces=tmp_path / "b.agentlog",
    )
    assert heu["mode"] == "heuristic_kind_allocator"
    for axis_rows in heu["attributions"].values():
        for row in axis_rows:
            assert expected_keys <= set(
                row.keys()
            ), f"heuristic row missing keys: {expected_keys - set(row.keys())}"

    # Placeholder-zero mode (neither backend nor candidate_traces).
    plc = run_bisect(
        tmp_path / "a.yaml",
        tmp_path / "b.yaml",
        tmp_path / "a.agentlog",
    )
    assert plc["mode"] == "lasso_placeholder_zero"
    for axis_rows in plc["attributions"].values():
        for row in axis_rows:
            assert expected_keys <= set(
                row.keys()
            ), f"placeholder row missing keys: {expected_keys - set(row.keys())}"


def test_full_factorial_k2_matches_manual_table() -> None:
    # Expected 4 rows x 2 cols: (-1,-1), (1,-1), (-1,1), (1,1).
    expected = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.int8)
    assert np.array_equal(full_factorial(2), expected)


def test_full_factorial_k_up_to_6_has_right_shape() -> None:
    for k in range(1, 7):
        d = full_factorial(k)
        assert d.shape == (1 << k, k)
        # Orthogonality: X^T X = runs * I
        assert is_orthogonal(d)


def test_full_factorial_rejects_k_gt_6() -> None:
    import pytest

    with pytest.raises(ValueError, match="capped at k=6"):
        full_factorial(7)


def test_plackett_burman_k8_has_12_runs_and_is_orthogonal() -> None:
    d = plackett_burman(8)
    assert d.shape == (12, 8)
    assert is_orthogonal(d)


def test_plackett_burman_k7_has_8_runs() -> None:
    d = plackett_burman(7)
    assert d.shape == (8, 7)
    assert is_orthogonal(d)


def test_choose_design_picks_factorial_for_small_k() -> None:
    d = choose_design(3)
    assert d.shape == (8, 3)


def test_choose_design_picks_pb_for_large_k() -> None:
    d = choose_design(8)
    assert d.shape == (12, 8)


def test_ground_truth_lasso_recovers_the_driver() -> None:
    """Synthetic scenario with 8 deltas; only delta #2 moves 'trajectory'.

    The LASSO attribution must give delta #2 the dominant share of the
    trajectory axis (≥0.9) and give every other delta at most 0.05.
    """
    k = 8
    design = plackett_burman(k)
    # Only delta #2 drives the trajectory axis (axis index 1).
    axis_weights = {2: {1: 1.0}}
    divergence = score_corners_synthetic(design, axis_weights)
    labels = [f"delta_{i}" for i in range(k)]
    attributions = rank_attributions(design, divergence, labels, alpha=0.01)

    # Look up trajectory-axis attributions.
    trajectory = dict(attributions["trajectory"])
    # Deltas that shouldn't move trajectory must get ≤ 0.05 weight.
    for i in range(k):
        if i == 2:
            continue
        assert (
            trajectory[f"delta_{i}"] <= 0.05
        ), f"non-driver delta_{i} got weight {trajectory[f'delta_{i}']:.3f}"
    # The actual driver gets ≥ 0.9.
    assert trajectory["delta_2"] >= 0.9

    # All other axes should be all-zero attributions (no signal).
    for axis in AXIS_NAMES:
        if axis == "trajectory":
            continue
        axis_attr = dict(attributions[axis])
        for i in range(k):
            assert axis_attr[f"delta_{i}"] == 0.0


def test_delta_kind_extracts_top_level_prefix() -> None:
    d = Delta(path="params.temperature", old_value=0.2, new_value=0.7)
    assert d.kind == "params"
