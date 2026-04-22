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

impl Severity {
    /// Classify a delta given the axis's 95% CI bounds.
    ///
    /// Rules:
    /// - If the CI crosses zero and the midpoint delta is small → None
    /// - else if abs(rel_delta) < 0.1 → Minor
    /// - else if abs(rel_delta) < 0.3 → Moderate
    /// - else Severe
    ///
    /// `baseline_median` may be zero; if so, Minor is returned when delta is
    /// non-zero (avoiding divide-by-zero).
    pub fn classify(delta: f64, baseline_median: f64, ci95_low: f64, ci95_high: f64) -> Severity {
        let ci_crosses_zero = ci95_low <= 0.0 && ci95_high >= 0.0;
        if ci_crosses_zero && delta.abs() < f64::max(baseline_median.abs() * 0.05, 1e-9) {
            return Severity::None;
        }
        if baseline_median.abs() < 1e-9 {
            return if delta.abs() < 1e-9 {
                Severity::None
            } else {
                Severity::Minor
            };
        }
        let rel = (delta / baseline_median).abs();
        if rel < 0.10 {
            Severity::Minor
        } else if rel < 0.30 {
            Severity::Moderate
        } else {
            Severity::Severe
        }
    }

    /// Classify a rate-like axis (values bounded in `[0, 1]`) by absolute
    /// magnitude of the delta. Used by [`crate::diff::safety`] and
    /// [`crate::diff::conformance`], where a shift from 0.0 → 0.33 is
    /// "1/3 of traffic flipped" — real, not noise.
    ///
    /// Thresholds:
    /// - CI crosses zero AND `|delta| < 1e-9` → None
    /// - `|delta| < 0.05` → Minor
    /// - `|delta| < 0.15` → Moderate
    /// - else → Severe
    pub fn classify_rate(delta: f64, ci95_low: f64, ci95_high: f64) -> Severity {
        let abs = delta.abs();
        let ci_crosses_zero = ci95_low <= 0.0 && ci95_high >= 0.0;
        if abs < 1e-9 && ci_crosses_zero {
            return Severity::None;
        }
        if abs < 0.05 {
            Severity::Minor
        } else if abs < 0.15 {
            Severity::Moderate
        } else {
            Severity::Severe
        }
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
