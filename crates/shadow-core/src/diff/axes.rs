//! Shared types for the nine-axis behavioral diff (CLAUDE.md §4).

use serde::{Deserialize, Serialize};

/// The nine behavioral axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    /// Final-output semantic similarity (embedding + structural).
    Semantic,
    /// Tool-call trajectory divergence (edit distance).
    Trajectory,
    /// Refusal / safety-filter rate.
    Safety,
    /// Output-token count CDF.
    Verbosity,
    /// End-to-end latency CDF.
    Latency,
    /// Cost distribution (tokens × pricing).
    Cost,
    /// Reasoning-depth (thinking tokens + self-correction markers).
    Reasoning,
    /// LLM-judge score (user-supplied rubric).
    Judge,
    /// Schema / format conformance rate.
    Conformance,
}

impl Axis {
    /// Human-readable label for terminal and markdown renderers.
    pub fn label(&self) -> &'static str {
        match self {
            Axis::Semantic => "semantic similarity",
            Axis::Trajectory => "tool-call trajectory",
            Axis::Safety => "refusal / safety",
            Axis::Verbosity => "verbosity",
            Axis::Latency => "latency",
            Axis::Cost => "cost",
            Axis::Reasoning => "reasoning depth",
            Axis::Judge => "llm-judge score",
            Axis::Conformance => "format conformance",
        }
    }

    /// All nine axes, in report order.
    pub fn all() -> [Axis; 9] {
        [
            Axis::Semantic,
            Axis::Trajectory,
            Axis::Safety,
            Axis::Verbosity,
            Axis::Latency,
            Axis::Cost,
            Axis::Reasoning,
            Axis::Judge,
            Axis::Conformance,
        ]
    }
}

/// Severity classification of a per-axis delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// No meaningful difference (abs(delta) within CI noise).
    None,
    /// Small effect (detectable but within 10% relative).
    Minor,
    /// Notable (10–30% relative).
    Moderate,
    /// Large (>30% relative, or CI excludes zero by a wide margin).
    Severe,
}

/// Caveat flags attached to a per-axis result.
///
/// Flags explain *why the severity is what it is* — they surface the
/// statistical caveats that users would otherwise have to read the CI
/// and `n` column to spot. A severity of `Severe` with no flags is
/// strong; a severity of `Severe` with `LowPower` means "the trend
/// looks large but our sample was too small to be confident."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Flag {
    /// `n < 5` — we don't have enough paired observations to be
    /// confident in the CI. Bootstrap on tiny samples produces wide
    /// intervals; treat severities as directional, not definitive.
    LowPower,
    /// The 95% CI includes zero. At the 95% confidence level we cannot
    /// reject "no effect"; the observed delta may be noise. Severity
    /// is capped at `Minor` in this case regardless of `|delta|`.
    CiCrossesZero,
}

impl Flag {
    /// Short machine-readable label for terminal / markdown / JSON output.
    pub fn label(&self) -> &'static str {
        match self {
            Flag::LowPower => "low_power",
            Flag::CiCrossesZero => "ci_crosses_zero",
        }
    }
}

/// Compute the caveat flags for an axis given its CI bounds and `n`.
pub fn compute_flags(ci95_low: f64, ci95_high: f64, n: usize) -> Vec<Flag> {
    let mut flags = Vec::new();
    if n < 5 && n > 0 {
        flags.push(Flag::LowPower);
    }
    // NaN-safe CI-straddles-zero check. Strict: a bound of exactly 0.0
    // is a boundary artifact (rate axes saturated at 0 or 1), not
    // genuine uncertainty about direction.
    if ci95_low.is_finite() && ci95_high.is_finite() && ci95_low < -1e-9 && ci95_high > 1e-9 {
        flags.push(Flag::CiCrossesZero);
    }
    flags
}

impl Severity {
    /// Classify a delta given the axis's 95% CI bounds.
    ///
    /// Rules:
    /// - If the CI crosses zero and the midpoint delta is small → None
    /// - If the CI crosses zero with any larger delta → capped at Minor
    ///   (we cannot reject "no effect" at 95%)
    /// - else if abs(rel_delta) < 0.1 → Minor
    /// - else if abs(rel_delta) < 0.3 → Moderate
    /// - else Severe
    ///
    /// `baseline_median` may be zero; if so, Minor is returned when delta is
    /// non-zero (avoiding divide-by-zero).
    pub fn classify(delta: f64, baseline_median: f64, ci95_low: f64, ci95_high: f64) -> Severity {
        // Reject NaN/Inf inputs explicitly. Rust's NaN comparisons always
        // return false, which would silently fall through to the rel-based
        // branches and return Severe on corrupt data — worst possible outcome
        // for a diff report.
        if !(delta.is_finite()
            && baseline_median.is_finite()
            && ci95_low.is_finite()
            && ci95_high.is_finite())
        {
            return Severity::None;
        }
        if delta.abs() < 1e-9 {
            // Exactly (or near-exactly) zero delta → nothing moved.
            return Severity::None;
        }
        // CI "straddles zero" means we genuinely can't determine direction:
        // both bounds are on opposite sides of zero by a meaningful margin.
        // A bound of exactly 0.0 is a boundary artifact (rate-bounded axes,
        // integer-valued statistics, etc.), not uncertainty — so don't
        // downgrade on it.
        let ci_straddles_zero = ci95_low < -1e-9 && ci95_high > 1e-9;
        if ci_straddles_zero && delta.abs() < f64::max(baseline_median.abs() * 0.05, 1e-9) {
            return Severity::None;
        }
        let base = if baseline_median.abs() < 1e-9 {
            if delta.abs() < 1e-9 {
                Severity::None
            } else {
                Severity::Minor
            }
        } else {
            let rel = (delta / baseline_median).abs();
            if rel < 0.10 {
                Severity::Minor
            } else if rel < 0.30 {
                Severity::Moderate
            } else {
                Severity::Severe
            }
        };
        // Only downgrade when the delta is small relative to CI width —
        // i.e. the signal is weak compared to noise. A unanimous (|delta|
        // ≥ CI width) observation should NOT be downgraded just because
        // a bootstrap resample happened to cross zero.
        if ci_straddles_zero && base > Severity::Minor {
            let ci_width = (ci95_high - ci95_low).abs();
            let delta_dominates = ci_width < 1e-9 || delta.abs() >= ci_width;
            if !delta_dominates {
                return Severity::Minor;
            }
        }
        base
    }

    /// Classify a rate-like axis (values bounded in `[0, 1]`) by absolute
    /// magnitude of the delta. Used by [`crate::diff::safety`] and
    /// [`crate::diff::conformance`], where a shift from 0.0 → 0.33 is
    /// "1/3 of traffic flipped" — real, not noise.
    ///
    /// Thresholds:
    /// - CI crosses zero AND `|delta| < 1e-9` → None
    /// - CI crosses zero with any larger delta → capped at Minor
    /// - `|delta| < 0.05` → Minor
    /// - `|delta| < 0.15` → Moderate
    /// - else → Severe
    pub fn classify_rate(delta: f64, ci95_low: f64, ci95_high: f64) -> Severity {
        // NaN guard — see classify() above.
        if !(delta.is_finite() && ci95_low.is_finite() && ci95_high.is_finite()) {
            return Severity::None;
        }
        let abs = delta.abs();
        if abs < 1e-9 {
            return Severity::None;
        }
        // Strict straddling: a CI bound of exactly 0.0 is a boundary
        // artifact for rate-axes bounded in [0,1] (e.g. saturated trajectory
        // divergence where every pair has divergence 1.0 or 0.0), not
        // statistical uncertainty.
        let ci_straddles_zero = ci95_low < -1e-9 && ci95_high > 1e-9;
        let base = if abs < 0.05 {
            Severity::Minor
        } else if abs < 0.15 {
            Severity::Moderate
        } else {
            Severity::Severe
        };
        // Only downgrade when delta is small relative to CI width.
        // Unanimous +1.0 delta with CI=[0,1] should remain Severe — the
        // point estimate dominates the CI.
        if ci_straddles_zero && base > Severity::Minor {
            let ci_width = (ci95_high - ci95_low).abs();
            let delta_dominates = ci_width < 1e-9 || abs >= ci_width;
            if !delta_dominates {
                return Severity::Minor;
            }
        }
        base
    }

    /// Short string for reports: "none" / "minor" / "moderate" / "severe".
    pub fn label(&self) -> &'static str {
        match self {
            Severity::None => "none",
            Severity::Minor => "minor",
            Severity::Moderate => "moderate",
            Severity::Severe => "severe",
        }
    }
}

/// One axis's statistical result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisStat {
    /// Which axis this row describes.
    pub axis: Axis,
    /// Median of the baseline sample.
    pub baseline_median: f64,
    /// Median of the candidate sample.
    pub candidate_median: f64,
    /// `candidate_median - baseline_median`.
    pub delta: f64,
    /// Lower bound of the 95% bootstrap CI of the delta.
    pub ci95_low: f64,
    /// Upper bound of the 95% bootstrap CI of the delta.
    pub ci95_high: f64,
    /// Severity classification per [`Severity::classify`].
    pub severity: Severity,
    /// Number of paired observations the axis was computed from. Zero
    /// means the axis had nothing to measure (e.g. no tool calls in
    /// either side → Trajectory axis is `n=0`).
    pub n: usize,
    /// Caveat flags — e.g. `low_power` (n<5) or `ci_crosses_zero`.
    #[serde(default)]
    pub flags: Vec<Flag>,
}

impl AxisStat {
    /// Build a "no data" row — used when the axis had nothing to measure.
    pub fn empty(axis: Axis) -> Self {
        Self {
            axis,
            baseline_median: 0.0,
            candidate_median: 0.0,
            delta: 0.0,
            ci95_low: 0.0,
            ci95_high: 0.0,
            severity: Severity::None,
            n: 0,
            flags: Vec::new(),
        }
    }

    /// Build an axis row for a continuous-valued axis (latency, verbosity,
    /// cost, reasoning, ...). Severity uses the relative-delta thresholds
    /// via [`Severity::classify`]; flags come from [`compute_flags`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_value(
        axis: Axis,
        baseline_median: f64,
        candidate_median: f64,
        delta: f64,
        ci95_low: f64,
        ci95_high: f64,
        n: usize,
    ) -> Self {
        Self {
            axis,
            baseline_median,
            candidate_median,
            delta,
            ci95_low,
            ci95_high,
            severity: Severity::classify(delta, baseline_median, ci95_low, ci95_high),
            n,
            flags: compute_flags(ci95_low, ci95_high, n),
        }
    }

    /// Build an axis row for a rate-like axis (values in `[0, 1]`:
    /// safety, conformance). Uses [`Severity::classify_rate`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_rate(
        axis: Axis,
        baseline_median: f64,
        candidate_median: f64,
        delta: f64,
        ci95_low: f64,
        ci95_high: f64,
        n: usize,
    ) -> Self {
        Self {
            axis,
            baseline_median,
            candidate_median,
            delta,
            ci95_low,
            ci95_high,
            severity: Severity::classify_rate(delta, ci95_low, ci95_high),
            n,
            flags: compute_flags(ci95_low, ci95_high, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_all_has_nine_entries() {
        assert_eq!(Axis::all().len(), 9);
    }

    #[test]
    fn axis_labels_are_unique() {
        let mut labels: Vec<&str> = Axis::all().iter().map(|a| a.label()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), 9);
    }

    #[test]
    fn severity_ci_crossing_zero_with_tiny_delta_is_none() {
        // Noise around a baseline of 100 with CI spanning zero.
        assert_eq!(Severity::classify(1.0, 100.0, -10.0, 10.0), Severity::None);
    }

    #[test]
    fn severity_minor_moderate_severe_thresholds() {
        assert_eq!(Severity::classify(5.0, 100.0, 3.0, 7.0), Severity::Minor); // 5% rel
        assert_eq!(
            Severity::classify(20.0, 100.0, 15.0, 25.0),
            Severity::Moderate
        ); // 20% rel
        assert_eq!(
            Severity::classify(50.0, 100.0, 40.0, 60.0),
            Severity::Severe
        ); // 50% rel
    }

    #[test]
    fn severity_capped_at_minor_when_ci_crosses_zero_with_small_delta() {
        // CI wider than delta: the observation is dominated by noise, so
        // cap at Minor even though the rel-delta would be Severe.
        // delta=50, ci=[-200, +300] (width=500, delta/width=0.1 — noisy)
        assert_eq!(
            Severity::classify(50.0, 100.0, -200.0, 300.0),
            Severity::Minor
        );
    }

    #[test]
    fn severity_not_downgraded_when_delta_dominates_ci_even_if_ci_straddles_zero() {
        // Would-be Severe (100% rel). CI [-5, +95] strictly straddles zero
        // but delta (100) ≥ CI width (100). Delta dominates — stays Severe.
        // This catches the v0.1 bug where saturated axes with legitimate
        // large deltas got reported as Minor.
        assert_eq!(
            Severity::classify(100.0, 100.0, -5.0, 95.0),
            Severity::Severe
        );
    }

    #[test]
    fn severity_rate_unanimous_saturated_delta_stays_severe() {
        // REGRESSION TEST: trajectory axis saturated at +1.0 (every pair
        // had 100% divergence). Bootstrap CI can legitimately touch
        // ci_low=0.0 due to rate-bounded resampling, but this is a
        // boundary artifact, not uncertainty. Must NOT be capped at Minor.
        assert_eq!(Severity::classify_rate(1.0, 0.0, 1.0), Severity::Severe);
    }

    #[test]
    fn severity_rate_capped_when_ci_genuinely_straddles_and_delta_is_small() {
        // Small delta + wide CI straddling zero → truly ambiguous, cap Minor.
        // delta=0.2, CI=[-0.3, +0.7] (width 1.0, delta/width=0.2).
        assert_eq!(Severity::classify_rate(0.2, -0.3, 0.7), Severity::Minor);
    }

    #[test]
    fn compute_flags_detects_low_power_and_ci_crosses_zero() {
        // Genuine straddle: [-1, +1] with n=3 → both LowPower + CiCrossesZero
        assert_eq!(
            compute_flags(-1.0, 1.0, 3),
            vec![Flag::LowPower, Flag::CiCrossesZero]
        );
        assert_eq!(compute_flags(0.5, 1.0, 3), vec![Flag::LowPower]);
        assert_eq!(compute_flags(-1.0, 1.0, 50), vec![Flag::CiCrossesZero]);
        assert!(compute_flags(0.5, 1.0, 50).is_empty());
    }

    #[test]
    fn compute_flags_does_not_flag_boundary_touching_ci() {
        // Rate axis saturated at 0: ci=[0,1] is a boundary-touching CI, not
        // a straddle. Should NOT flag ci_crosses_zero.
        assert!(!compute_flags(0.0, 1.0, 50).contains(&Flag::CiCrossesZero));
        // Symmetric case: negative-saturated ci=[-1, 0].
        assert!(!compute_flags(-1.0, 0.0, 50).contains(&Flag::CiCrossesZero));
    }

    #[test]
    fn severity_classify_rejects_nan_inputs() {
        // NaN comparisons always false in Rust — guard required or we'd
        // silently return Severe on corrupt data.
        assert_eq!(
            Severity::classify(f64::NAN, 100.0, -10.0, 10.0),
            Severity::None
        );
        assert_eq!(
            Severity::classify(5.0, f64::NAN, -10.0, 10.0),
            Severity::None
        );
        assert_eq!(
            Severity::classify(5.0, 100.0, f64::NAN, 10.0),
            Severity::None
        );
        assert_eq!(
            Severity::classify(5.0, 100.0, -10.0, f64::INFINITY),
            Severity::None
        );
    }

    #[test]
    fn severity_classify_rate_rejects_nan_inputs() {
        assert_eq!(Severity::classify_rate(f64::NAN, 0.0, 1.0), Severity::None);
        assert_eq!(Severity::classify_rate(0.5, f64::NAN, 1.0), Severity::None);
    }

    #[test]
    fn compute_flags_ignores_nan_ci_bounds() {
        // NaN CI doesn't count as crossing zero (the inequality is undefined).
        let flags = compute_flags(f64::NAN, 1.0, 10);
        assert!(!flags.contains(&Flag::CiCrossesZero));
    }

    #[test]
    fn compute_flags_n_zero_means_no_low_power() {
        // n=0 is an "axis had nothing to measure" sentinel, not "tiny sample".
        assert!(compute_flags(0.0, 0.0, 0).is_empty());
    }

    #[test]
    fn severity_handles_zero_baseline() {
        assert_eq!(Severity::classify(0.0, 0.0, 0.0, 0.0), Severity::None);
        assert_eq!(Severity::classify(1.0, 0.0, 0.5, 1.5), Severity::Minor);
    }

    #[test]
    fn severity_labels_distinguish_four_levels() {
        let labels: Vec<&str> = [
            Severity::None,
            Severity::Minor,
            Severity::Moderate,
            Severity::Severe,
        ]
        .iter()
        .map(|s| s.label())
        .collect();
        assert_eq!(labels, vec!["none", "minor", "moderate", "severe"]);
    }

    #[test]
    fn axis_stat_empty_has_zero_n() {
        let s = AxisStat::empty(Axis::Latency);
        assert_eq!(s.n, 0);
        assert_eq!(s.severity, Severity::None);
    }
}
