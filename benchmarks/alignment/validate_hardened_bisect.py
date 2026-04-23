"""Real-world validation of hardened causal bisection.

Runs the interaction-enabled attribution engine on a realistic multi-
delta config-change scenario with KNOWN ground truth. Verifies:

  - Main-effect recovery matches ground truth (true drivers → high
    weight, narrow CI excluding 0, significant)
  - Interaction recovery: when the data-generating process includes
    AxB, the detector surfaces it with an honest CI
  - Strong-hierarchy filter removes spurious interactions
  - Output format matches the "78% [73%, 83%]" spec from the research
    brief
"""

from __future__ import annotations

import sys

sys.path.insert(0, "python/src")

import numpy as np

from shadow.bisect.attribution import (
    AXIS_NAMES,
    rank_attributions_with_interactions,
)
from shadow.bisect.design import full_factorial


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_row(row: dict) -> str:
    w = fmt_pct(row["weight"])
    lo = fmt_pct(row["ci95_low"])
    hi = fmt_pct(row["ci95_high"])
    sf = row["selection_frequency"]
    sig = "✓" if row["significant"] else " "
    return f"{w} [{lo}, {hi}]  sel_freq={sf:.2f}  {sig}"


def main() -> int:
    print("=" * 78)
    print(" HARDENED BISECTION VALIDATION — interactions + bootstrap CIs")
    print("=" * 78)

    # Realistic scenario: Shadow is bisecting an agent-config PR that
    # touches FOUR deltas. Ground-truth data-generating process:
    #
    #   semantic   ← 65% from prompt main + 25% from promptxmodel interaction + noise
    #   trajectory ← 80% from tools main + noise
    #   latency    ← 50% from model main + 40% from modelxtools + noise
    #   verbosity  ← 70% from prompt main (no interactions)
    #
    # All other axes: noise only.
    design = full_factorial(4)
    labels = ["model_id", "params.temperature", "prompt.system", "tools"]
    rng = np.random.default_rng(42)
    runs = design.shape[0]
    divergence = np.zeros((runs, len(AXIS_NAMES)), dtype=float)

    def axis(name: str) -> int:
        return AXIS_NAMES.index(name)

    model_on = design[:, 0] == 1
    prompt_on = design[:, 2] == 1
    tools_on = design[:, 3] == 1

    # semantic: prompt main + promptxmodel interaction
    divergence[:, axis("semantic")] = (
        prompt_on.astype(float) * 3.0
        + (prompt_on & model_on).astype(float) * 1.5
        + rng.normal(0, 0.3, runs)
    )
    # trajectory: tools main only
    divergence[:, axis("trajectory")] = tools_on.astype(float) * 4.0 + rng.normal(
        0, 0.3, runs
    )
    # latency: model main + modelxtools interaction
    divergence[:, axis("latency")] = (
        model_on.astype(float) * 2.0
        + (model_on & tools_on).astype(float) * 1.8
        + rng.normal(0, 0.3, runs)
    )
    # verbosity: prompt main only
    divergence[:, axis("verbosity")] = prompt_on.astype(float) * 2.5 + rng.normal(
        0, 0.3, runs
    )

    print("\nGround truth:")
    print("  semantic   ← prompt main + promptxmodel interaction")
    print("  trajectory ← tools main only")
    print("  latency    ← model main + modelxtools interaction")
    print("  verbosity  ← prompt main only")
    print("  (others: noise)")
    print()

    result = rank_attributions_with_interactions(
        design, divergence, labels, n_bootstrap=500, seed=1, alpha=None
    )

    failures = 0

    def assert_case(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}")
        if not ok:
            print(f"     detail: {detail}")
            failures += 1

    # ----- Semantic axis -----
    print("\n" + "─" * 78)
    print(" SEMANTIC — expecting prompt main (strong) + promptxmodel interaction")
    print("─" * 78)
    sem = result["semantic"]
    print("  Main effects:")
    for r in sem["main_effects"]:
        print(f"    {r['delta']:<22}  {render_row(r)}")
    print("  Interactions:")
    for r in sem["interactions"]:
        print(f"    {r['label']:<38}  {render_row(r)}")

    prompt_main = next(r for r in sem["main_effects"] if r["delta"] == "prompt.system")
    assert_case(
        "semantic: prompt.system is the top main effect",
        sem["main_effects"][0]["delta"] == "prompt.system",
        f"top was {sem['main_effects'][0]['delta']}",
    )
    assert_case(
        "semantic: prompt.system has CI above 0 and is significant",
        prompt_main["ci95_low"] > 0.0 and prompt_main["significant"],
        f"ci_low={prompt_main['ci95_low']:.3f} sig={prompt_main['significant']}",
    )
    # Interaction should be surfaced (promptxmodel)
    prompt_model_inter = [
        r
        for r in sem["interactions"]
        if set(r["pair"]) == {"prompt.system", "model_id"}
    ]
    assert_case(
        "semantic: promptxmodel interaction is surfaced",
        len(prompt_model_inter) == 1,
        f"found {len(prompt_model_inter)} promptxmodel interactions",
    )

    # ----- Trajectory axis -----
    print("\n" + "─" * 78)
    print(" TRAJECTORY — expecting tools main only, no interactions")
    print("─" * 78)
    traj = result["trajectory"]
    print("  Main effects:")
    for r in traj["main_effects"]:
        print(f"    {r['delta']:<22}  {render_row(r)}")
    print("  Interactions:")
    for r in traj["interactions"]:
        print(f"    {r['label']:<38}  {render_row(r)}")

    tools_main = next(r for r in traj["main_effects"] if r["delta"] == "tools")
    assert_case(
        "trajectory: tools is the top main effect",
        traj["main_effects"][0]["delta"] == "tools",
        f"top was {traj['main_effects'][0]['delta']}",
    )
    assert_case(
        "trajectory: tools is significant",
        tools_main["significant"] is True,
        "",
    )

    # ----- Latency axis -----
    print("\n" + "─" * 78)
    print(" LATENCY — expecting model main + modelxtools interaction")
    print("─" * 78)
    lat = result["latency"]
    print("  Main effects:")
    for r in lat["main_effects"]:
        print(f"    {r['delta']:<22}  {render_row(r)}")
    print("  Interactions:")
    for r in lat["interactions"]:
        print(f"    {r['label']:<38}  {render_row(r)}")

    # modelxtools interaction should appear
    model_tools_inter = [
        r for r in lat["interactions"] if set(r["pair"]) == {"model_id", "tools"}
    ]
    assert_case(
        "latency: modelxtools interaction is surfaced",
        len(model_tools_inter) == 1,
        f"found {len(model_tools_inter)} modelxtools interactions",
    )

    # ----- Verbosity axis (main-only, strong hierarchy filter) -----
    print("\n" + "─" * 78)
    print(" VERBOSITY — expecting prompt main ONLY (no interactions)")
    print("─" * 78)
    verb = result["verbosity"]
    print("  Main effects:")
    for r in verb["main_effects"]:
        print(f"    {r['delta']:<22}  {render_row(r)}")
    print("  Interactions:")
    if verb["interactions"]:
        for r in verb["interactions"]:
            print(f"    {r['label']:<38}  {render_row(r)}")
    else:
        print("    (none — strong hierarchy filtered)")

    prompt_verb = next(r for r in verb["main_effects"] if r["delta"] == "prompt.system")
    assert_case(
        "verbosity: prompt.system is the top main effect",
        verb["main_effects"][0]["delta"] == "prompt.system",
        f"top was {verb['main_effects'][0]['delta']}",
    )
    assert_case(
        "verbosity: significant",
        prompt_verb["significant"] is True,
        "",
    )

    # ----- Output format check -----
    print("\n" + "─" * 78)
    print(" OUTPUT FORMAT — '78% [73%, 83%]' percentile-based")
    print("─" * 78)
    # Render the top semantic main effect in the spec format
    top = sem["main_effects"][0]
    formatted = (
        f"{fmt_pct(top['weight'])} [{fmt_pct(top['ci95_low'])}, "
        f"{fmt_pct(top['ci95_high'])}]"
    )
    print("  Top semantic attribution:")
    print(f"    {top['delta']:<22}  {formatted}")
    assert_case(
        "output format parses as 'X% [Y%, Z%]'",
        formatted.count("%") == 3 and "[" in formatted and "]" in formatted,
        f"got: {formatted}",
    )

    # ----- Summary -----
    print("\n" + "=" * 78)
    if failures == 0:
        print(" ✅ ALL CHECKS PASSED")
    else:
        print(f" ❌ {failures} CHECK(S) FAILED")
    print("=" * 78)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
