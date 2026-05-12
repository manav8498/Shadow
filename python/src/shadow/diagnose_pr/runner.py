"""Pure-Python entry point for `shadow diagnose-pr`.

Extracted from `shadow.cli.app.diagnose_pr_cmd` so the same logic
can be invoked directly by `shadow gate-pr` (and any future
programmatic caller) without going through the Typer command
machinery.

The CLI command is now a thin wrapper that:
  1. Parses Typer options.
  2. Calls `run_diagnose_pr(opts)`.
  3. Writes the JSON + markdown.
  4. Maps `--fail-on` to an exit code.

This module never touches argv, stdout, or sys.exit — it returns
the assembled `DiagnosePrReport` plus optional rendered markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shadow.diagnose_pr.attribution import (
    causal_from_replay,
    pick_dominant,
    simple_attribution,
    suggested_fix_for,
)
from shadow.diagnose_pr.deltas import extract_deltas
from shadow.diagnose_pr.diffing import diff_pair, is_affected, worst_axis_for
from shadow.diagnose_pr.loaders import LoadedTrace, load_config, load_traces
from shadow.diagnose_pr.models import DiagnosePrReport, TraceDiagnosis
from shadow.diagnose_pr.policy import evaluate_policy
from shadow.diagnose_pr.report import build_report
from shadow.diagnose_pr.risk import is_dangerous_violation


@dataclass(frozen=True)
class DiagnoseOptions:
    """Pure-data inputs to `run_diagnose_pr`. Mirrors the CLI flags
    one-for-one so the CLI command body stays trivial."""

    traces: list[Path]
    candidate_traces: list[Path] | None
    baseline_config: Path
    candidate_config: Path
    policy: Path | None = None
    changed_files: list[str] | None = None
    max_traces: int = 200
    backend: str = "recorded"
    n_bootstrap: int = 500
    pricing: dict[str, tuple[float, float]] | None = None
    max_cost_usd: float | None = None  # live-backend safety cap (Gap 7)
    # Optional baseline ref (e.g. `origin/main` or a PR base SHA) used
    # to hunk-blame prompt files via `git diff <ref>...HEAD`. When set
    # along with --changed-files, prompt deltas carry file:line +
    # removed/added text in the report and PR comment.
    baseline_ref: str | None = None
    repo_root: Path | None = None


@dataclass(frozen=True)
class DiagnoseResult:
    """Outcome of `run_diagnose_pr`. The `cost_usd` field is `None`
    for non-live backends (no real spend)."""

    report: DiagnosePrReport
    backend_used: str
    cost_usd: float | None = None
    cost_breakdown: list[dict[str, Any]] = field(default_factory=list)


_VALID_BACKENDS = frozenset({"recorded", "mock", "live"})


# Mock signal magnitudes per DeltaKind. Differentiated so multi-
# delta scenarios produce a clear dominant cause rather than tying
# at a uniform 0.5. NOT grounded in real-world impact data — for
# the mock backend's demo / testing purpose. PR comments built on
# top of this surface a "synthetic mock backend" disclosure (see
# render.py).
_MOCK_DELTA_SIGNAL_MAP: dict[str, tuple[str, float]] = {
    "prompt": ("trajectory", 0.6),
    "policy": ("safety", 0.7),
    "tool_schema": ("trajectory", 0.5),
    "retriever": ("semantic", 0.5),
    "model": ("verbosity", 0.4),
    "temperature": ("verbosity", 0.3),
    "unknown": ("semantic", 0.2),
}


def run_diagnose_pr(opts: DiagnoseOptions) -> DiagnoseResult:
    """Execute the full diagnose-pr pipeline and return the
    assembled report.

    Raises typed errors from `shadow.errors` on user-facing failure
    paths (missing config, malformed YAML, no filename overlap,
    invalid backend). Callers convert these to CLI error messages.
    """
    if opts.backend not in _VALID_BACKENDS:
        raise ValueError(
            f"--backend must be one of {sorted(_VALID_BACKENDS)}, got {opts.backend!r}"
        )

    baseline = load_config(opts.baseline_config)
    candidate = load_config(opts.candidate_config)
    loaded = load_traces(list(opts.traces))
    cand_loaded = (
        load_traces(list(opts.candidate_traces)) if opts.candidate_traces is not None else None
    )

    prompt_blame = None
    if opts.baseline_ref and opts.changed_files:
        from shadow.diagnose_pr.git_blame import blame_prompt_files

        repo_root = opts.repo_root or Path.cwd()
        prompt_blame = blame_prompt_files(
            repo_root=repo_root,
            baseline_ref=opts.baseline_ref,
            paths=list(opts.changed_files),
        )
    deltas = extract_deltas(
        baseline,
        candidate,
        changed_files=opts.changed_files,
        prompt_blame=prompt_blame,
    )

    if opts.max_traces > 0 and len(loaded) > opts.max_traces:
        from shadow.mine import mine as _mine

        records_only = [t.records for t in loaded]
        sampled = _mine(records_only, max_cases=opts.max_traces, per_cluster=1)
        sampled_trace_ids = {case.baseline_source for case in sampled.cases}
        loaded = [t for t in loaded if t.trace_id in sampled_trace_ids]
        if not loaded:
            raise RuntimeError("mining produced no usable cases — corpus may be malformed")

    cand_by_name: dict[str, list[dict[str, Any]]] = {}
    if cand_loaded is not None:
        cand_by_name = {t.path.name: t.records for t in cand_loaded}
        baseline_names = {t.path.name for t in loaded}
        if not (baseline_names & cand_by_name.keys()):
            raise RuntimeError(
                "no baseline trace filename matches any candidate trace filename — "
                "pair by exact filename (e.g. baseline/x.agentlog <-> candidate/x.agentlog)"
            )

    diagnoses, has_severe_axis, has_dangerous_violation, total_new_violations, worst_rule_id = (
        _per_trace_diff_and_policy(loaded, cand_by_name, opts.policy)
    )

    has_divergence = any(d.affected for d in diagnoses)
    top_causes, cost_usd, cost_breakdown = _attribute_causes(
        baseline=baseline,
        candidate=candidate,
        deltas=deltas,
        loaded=loaded,
        backend=opts.backend,
        n_bootstrap=opts.n_bootstrap,
        has_divergence=has_divergence,
        max_cost_usd=opts.max_cost_usd,
    )
    dominant = pick_dominant(top_causes)
    suggested_fix = suggested_fix_for(dominant, deltas=deltas)

    backend_flags: list[str] = []
    # Always disclose synthetic backend, not just when a dominant cause
    # is crowned. A buyer reading the PR comment must see that the
    # numbers below come from a deterministic heuristic before they
    # read the verdict, regardless of whether any cause was picked.
    if opts.backend == "mock":
        backend_flags.append("synthetic_mock")

    report = build_report(
        traces=loaded,
        deltas=deltas,
        diagnoses=diagnoses,
        new_policy_violations=total_new_violations,
        worst_policy_rule=worst_rule_id,
        has_dangerous_violation=has_dangerous_violation,
        has_severe_axis=has_severe_axis,
        top_causes=top_causes,
        dominant_cause=dominant,
        suggested_fix=suggested_fix,
        extra_flags=backend_flags,
    )

    return DiagnoseResult(
        report=report,
        backend_used=opts.backend,
        cost_usd=cost_usd,
        cost_breakdown=cost_breakdown,
    )


_PARALLEL_THRESHOLD = 16
"""Below this many paired traces, the threadpool overhead isn't
worth it — we run sequentially. Above it, fan out across cores."""


def _diff_one_pair(
    t: LoadedTrace,
    cand_records: list[dict[str, Any]] | None,
    policy: Path | None,
) -> tuple[TraceDiagnosis, bool, bool, int, str | None, int]:
    """Diff + policy check for one paired trace. Returns:
        (diagnosis, has_severe_axis_locally, has_dangerous_locally,
         new_violations, worst_rule_id_locally,
         worst_rule_severity_rank_locally)
    The aggregator merges these per-pair results."""
    if cand_records is None:
        return (
            TraceDiagnosis(
                trace_id=t.trace_id,
                affected=False,
                risk=0.0,
                worst_axis=None,
                first_divergence=None,
                policy_violations=[],
            ),
            False,
            False,
            0,
            None,
            -1,
        )

    diff_report = diff_pair(t.records, cand_records)
    affected = is_affected(diff_report)
    worst_axis = worst_axis_for(diff_report)
    first_div = diff_report.get("first_divergence")
    has_severe_local = any(
        row.get("severity") == "severe" for row in (diff_report.get("rows") or [])
    )

    policy_result = evaluate_policy(policy, t.records, cand_records)
    severity_order = {"info": 0, "warning": 1, "error": 2, "critical": 3}
    pol_severity_rank = (
        max(
            (severity_order.get(v.get("severity", ""), 0) for v in policy_result.regressions),
            default=-1,
        )
        if policy_result.worst_rule is not None
        else -1
    )
    has_dangerous_local = any(is_dangerous_violation(reg) for reg in policy_result.regressions)

    return (
        TraceDiagnosis(
            trace_id=t.trace_id,
            affected=affected,
            risk=0.0,
            worst_axis=worst_axis,
            first_divergence=first_div,
            policy_violations=policy_result.regressions,
        ),
        has_severe_local,
        has_dangerous_local,
        policy_result.new_violations,
        policy_result.worst_rule,
        pol_severity_rank,
    )


def _per_trace_diff_and_policy(
    loaded: list[LoadedTrace],
    cand_by_name: dict[str, list[dict[str, Any]]],
    policy: Path | None,
) -> tuple[list[TraceDiagnosis], bool, bool, int, str | None]:
    """Run the per-pair 9-axis diff + per-pair policy check across
    every paired trace. Returns the diagnoses list and four
    aggregate flags consumed by build_report + risk.classify_verdict.

    Above `_PARALLEL_THRESHOLD` traces, runs the per-pair work in
    a thread pool. Both `diff_pair` (Rust via PyO3) and
    `evaluate_policy` release the GIL during their heavy work
    (Rust differ does, regex-driven policy check spends most time
    in C-extension regex), so threading gives a real speedup on
    big corpora without process-level overhead.
    """
    pairs: list[tuple[LoadedTrace, list[dict[str, Any]] | None]] = [
        (t, cand_by_name.get(t.path.name)) for t in loaded
    ]

    if len(pairs) >= _PARALLEL_THRESHOLD:
        from concurrent.futures import ThreadPoolExecutor

        def _worker(
            args: tuple[LoadedTrace, list[dict[str, Any]] | None],
        ) -> tuple[TraceDiagnosis, bool, bool, int, str | None, int]:
            t, c = args
            return _diff_one_pair(t, c, policy)

        # Cap at 8 — diminishing returns beyond and most CI runners
        # have 2-4 cores. The Rust differ is the bottleneck.
        with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as ex:
            results = list(ex.map(_worker, pairs))
    else:
        results = [_diff_one_pair(t, c, policy) for t, c in pairs]

    diagnoses: list[TraceDiagnosis] = []
    has_severe_axis = False
    has_dangerous_violation = False
    total_new_violations = 0
    worst_rule_id: str | None = None
    worst_rule_severity_rank = -1

    for diag, severe, dangerous, n_violations, rule_id, rule_rank in results:
        diagnoses.append(diag)
        has_severe_axis = has_severe_axis or severe
        has_dangerous_violation = has_dangerous_violation or dangerous
        total_new_violations += n_violations
        if rule_id is not None and rule_rank > worst_rule_severity_rank:
            worst_rule_severity_rank = rule_rank
            worst_rule_id = rule_id

    return (
        diagnoses,
        has_severe_axis,
        has_dangerous_violation,
        total_new_violations,
        worst_rule_id,
    )


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _attribute_causes(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    deltas: list[Any],
    loaded: list[LoadedTrace],
    backend: str,
    n_bootstrap: int,
    has_divergence: bool,
    max_cost_usd: float | None,
) -> tuple[list[Any], float | None, list[dict[str, Any]]]:
    """Run cause attribution per the requested backend. Returns
    (top_causes, cost_usd, cost_breakdown). cost_usd is None for
    non-live backends; cost_breakdown is empty unless the live
    backend tracked spend per call."""
    if backend == "mock" and deltas:
        kind_by_path = {d.path: d.kind for d in deltas}
        flat_baseline = _flatten(baseline)
        flat_candidate = _flatten(candidate)

        def _mock_replay(config: dict[str, Any]) -> dict[str, float]:
            div = {
                "semantic": 0.0,
                "trajectory": 0.0,
                "safety": 0.0,
                "verbosity": 0.0,
                "latency": 0.0,
            }
            for path, cand_val in flat_candidate.items():
                if flat_baseline.get(path) == cand_val:
                    continue
                if config.get(path) == cand_val:
                    kind = kind_by_path.get(path, "unknown")
                    ax, magnitude = _MOCK_DELTA_SIGNAL_MAP.get(
                        kind, _MOCK_DELTA_SIGNAL_MAP["unknown"]
                    )
                    div[ax] += magnitude
            return div

        causes = causal_from_replay(
            baseline_config=flat_baseline,
            candidate_config=flat_candidate,
            replay_fn=_mock_replay,
            n_bootstrap=n_bootstrap,
            sensitivity=True,
            deltas=deltas,
        )
        return causes, None, []

    if backend == "live" and deltas:
        from shadow.diagnose_pr.live import build_live_replay_fn_per_corpus

        if not loaded:
            raise RuntimeError("--backend live requires at least one baseline trace")

        flat_baseline = _flatten(baseline)
        flat_candidate = _flatten(candidate)

        live_fn, cost_tracker = build_live_replay_fn_per_corpus(
            baseline_traces=loaded,
            max_cost_usd=max_cost_usd,
        )
        causes = causal_from_replay(
            baseline_config=flat_baseline,
            candidate_config=flat_candidate,
            replay_fn=live_fn,
            n_bootstrap=n_bootstrap,
            sensitivity=True,
            deltas=deltas,
        )
        return causes, cost_tracker.total_usd, list(cost_tracker.breakdown)

    return simple_attribution(deltas=deltas, has_divergence=has_divergence), None, []


__all__ = [
    "DiagnoseOptions",
    "DiagnoseResult",
    "run_diagnose_pr",
]
