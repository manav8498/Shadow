"""Ground-truth tests for shadow.causal.attribution.

The headline test: construct a synthetic scenario with 3 real deltas
and 5 noise deltas, where only the real deltas affect the divergence.
The causal attribution algorithm should rank the real deltas above
the noise deltas — the same property a NeurIPS / ICML reviewer would
demand of a causal-attribution paper.
"""

from __future__ import annotations

from typing import Any

import pytest

from shadow.causal import (
    CausalAttribution,
    InterventionResult,
    causal_attribution,
)

# ---------------------------------------------------------------------------
# Synthetic ground-truth replay function
# ---------------------------------------------------------------------------


def _ground_truth_replay(config: dict[str, Any]) -> dict[str, float]:
    """A deterministic 'replay' that depends ONLY on three named deltas.

    - delta_temperature (real): affects 'verbosity' and 'safety' axes.
    - delta_tool_schema (real): affects 'trajectory' axis.
    - delta_system_prompt (real): affects 'semantic' and 'safety' axes.

    All other deltas in the config are ignored — they don't move the
    output. A correct causal attribution algorithm must rank the
    three real deltas above the noise deltas.
    """
    div: dict[str, float] = {
        "semantic": 0.0,
        "trajectory": 0.0,
        "safety": 0.0,
        "verbosity": 0.0,
        "latency": 0.0,
    }
    if config.get("delta_temperature") == "high":
        div["verbosity"] += 0.4
        div["safety"] += 0.2
    if config.get("delta_tool_schema") == "extended":
        div["trajectory"] += 0.6
    if config.get("delta_system_prompt") == "loose":
        div["semantic"] += 0.5
        div["safety"] += 0.1
    return div


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


class TestAPI:
    def test_returns_causal_attribution(self):
        baseline = {"delta_temperature": "low"}
        candidate = {"delta_temperature": "high"}
        result = causal_attribution(
            baseline_config=baseline,
            candidate_config=candidate,
            replay_fn=_ground_truth_replay,
        )
        assert isinstance(result, CausalAttribution)
        assert "delta_temperature" in result.ate

    def test_no_differing_keys_raises(self):
        with pytest.raises(ValueError, match="no differing keys"):
            causal_attribution(
                baseline_config={"a": 1},
                candidate_config={"a": 1},
                replay_fn=_ground_truth_replay,
            )

    def test_invalid_n_replays_raises(self):
        with pytest.raises(ValueError, match="n_replays"):
            causal_attribution(
                baseline_config={"a": 1},
                candidate_config={"a": 2},
                replay_fn=_ground_truth_replay,
                n_replays=0,
            )

    def test_intervention_result_dataclass(self):
        iv = InterventionResult(delta="x", divergence={"axis": 0.5})
        assert iv.delta == "x"
        d = iv.to_dict()
        assert d["delta"] == "x"
        assert d["divergence"] == {"axis": 0.5}


# ---------------------------------------------------------------------------
# Ground-truth attribution
# ---------------------------------------------------------------------------


class TestGroundTruthAttribution:
    """The headline test: 3 real deltas + 5 noise deltas → algorithm
    correctly identifies the 3 real ones."""

    def _build_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        baseline = {
            "delta_temperature": "low",
            "delta_tool_schema": "basic",
            "delta_system_prompt": "strict",
            # Five noise deltas with no causal effect on the output.
            "noise_seed": 42,
            "noise_max_tokens": 1024,
            "noise_top_p": 0.9,
            "noise_logger": "info",
            "noise_session_tag": "baseline",
        }
        candidate = {
            "delta_temperature": "high",
            "delta_tool_schema": "extended",
            "delta_system_prompt": "loose",
            "noise_seed": 99,
            "noise_max_tokens": 2048,
            "noise_top_p": 0.95,
            "noise_logger": "debug",
            "noise_session_tag": "candidate",
        }
        return baseline, candidate

    def test_real_deltas_identified_by_attribution(self):
        baseline, candidate = self._build_configs()
        result = causal_attribution(
            baseline_config=baseline,
            candidate_config=candidate,
            replay_fn=_ground_truth_replay,
        )

        # The three real deltas must have non-zero ATE on at least one
        # axis; the five noise deltas must have ATE ≈ 0 across all axes.
        real_deltas = {
            "delta_temperature",
            "delta_tool_schema",
            "delta_system_prompt",
        }
        noise_deltas = {
            "noise_seed",
            "noise_max_tokens",
            "noise_top_p",
            "noise_logger",
            "noise_session_tag",
        }

        for d in real_deltas:
            max_abs_ate = max(abs(v) for v in result.ate[d].values())
            assert max_abs_ate >= 0.1, (
                f"real delta {d!r} has ATE {result.ate[d]} — should be ≥0.1 on some axis"
            )

        for d in noise_deltas:
            max_abs_ate = max(abs(v) for v in result.ate[d].values())
            assert max_abs_ate < 0.01, (
                f"noise delta {d!r} has ATE {result.ate[d]} — should be ≈0 (got {max_abs_ate})"
            )

    def test_per_axis_attribution_correctly_decomposed(self):
        """Each delta's effect should land on the axes it actually moves."""
        baseline, candidate = self._build_configs()
        result = causal_attribution(
            baseline_config=baseline,
            candidate_config=candidate,
            replay_fn=_ground_truth_replay,
        )

        # delta_temperature affects verbosity and safety, not trajectory.
        temp_ate = result.ate["delta_temperature"]
        assert abs(temp_ate["verbosity"]) > 0.1
        assert abs(temp_ate["safety"]) > 0.1
        assert abs(temp_ate.get("trajectory", 0.0)) < 0.01

        # delta_tool_schema affects trajectory only.
        tool_ate = result.ate["delta_tool_schema"]
        assert abs(tool_ate["trajectory"]) > 0.5
        assert abs(tool_ate.get("verbosity", 0.0)) < 0.01

        # delta_system_prompt affects semantic and (a bit) safety.
        prompt_ate = result.ate["delta_system_prompt"]
        assert abs(prompt_ate["semantic"]) > 0.4
        assert abs(prompt_ate.get("trajectory", 0.0)) < 0.01

    def test_top_k_ranks_real_above_noise(self):
        baseline, candidate = self._build_configs()
        result = causal_attribution(
            baseline_config=baseline,
            candidate_config=candidate,
            replay_fn=_ground_truth_replay,
        )
        # On the trajectory axis only delta_tool_schema has effect.
        top_traj = result.top("trajectory", k=3)
        assert top_traj[0][0] == "delta_tool_schema"
        # Remaining entries should all be near-zero noise.
        for _, ate in top_traj[1:]:
            assert abs(ate) < 0.01


# ---------------------------------------------------------------------------
# n_replays averaging
# ---------------------------------------------------------------------------


class TestReplayAveraging:
    def test_n_replays_averages_noisy_replays(self):
        """With a noisy replay function, n_replays > 1 should reduce
        variance in the ATE estimate."""
        import random

        rng = random.Random(0)

        def noisy_replay(config: dict[str, Any]) -> dict[str, float]:
            base = _ground_truth_replay(config)
            # Inject noise on every axis.
            return {k: v + rng.gauss(0, 0.05) for k, v in base.items()}

        baseline = {"delta_temperature": "low", "noise_a": 1}
        candidate = {"delta_temperature": "high", "noise_a": 2}

        result = causal_attribution(
            baseline_config=baseline,
            candidate_config=candidate,
            replay_fn=noisy_replay,
            n_replays=10,
        )
        # Noise delta should still be near-zero; real delta has its
        # known effect (within tolerance).
        assert abs(result.ate["noise_a"]["verbosity"]) < 0.10
        assert abs(result.ate["delta_temperature"]["verbosity"]) > 0.20
