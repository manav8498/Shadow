//! Rendering of [`DiffReport`] to markdown and terminal.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::diff::alignment::FirstDivergence;
use crate::diff::axes::{AxisStat, Severity};
use crate::diff::drill_down::PairDrilldown;
use crate::diff::recommendations::Recommendation;

/// Top-level diff result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffReport {
    /// Per-axis rows, in [`Axis::all`] order (nine entries).
    pub rows: Vec<AxisStat>,
    /// Content id of the baseline trace used to produce this report.
    pub baseline_trace_id: String,
    /// Content id of the candidate trace used to produce this report.
    pub candidate_trace_id: String,
    /// Number of paired responses the report is based on.
    pub pair_count: usize,
    /// The first turn at which the candidate meaningfully diverged from
    /// the baseline, with a classification (style / decision / structural).
    /// `None` when the two traces agree end-to-end.
    ///
    /// Preserved for backward compatibility; equivalent to the first
    /// element of [`Self::divergences`] sorted by alignment order (not
    /// by importance rank).
    #[serde(default)]
    pub first_divergence: Option<FirstDivergence>,
    /// Top-ranked divergences between the two traces, sorted by
    /// importance (Structural > Decision > Style by class, then by
    /// confidence within a class). Empty when the traces agree
    /// end-to-end. Capped at `DEFAULT_K` entries in standard reports;
    /// renderers typically show the top 3 inline and hide the rest in
    /// a collapsible details section.
    #[serde(default)]
    pub divergences: Vec<FirstDivergence>,
    /// Prescriptive fix recommendations derived from the divergence
    /// set and the axis rows. Sorted by severity (Error > Warning >
    /// Info), capped at 8 entries. Every recommendation names a
    /// specific action (Restore / Remove / Revert / Review / Verify)
    /// and the turn it targets. Empty when nothing is actionable.
    #[serde(default)]
    pub recommendations: Vec<Recommendation>,
    /// Top-K most-regressive response pairs with per-axis breakdown,
    /// ranked by an aggregate regression score. Surfaces *which*
    /// specific turns drove the aggregate axis deltas — without this,
    /// a reviewer looking at a PR with many paired traces has to
    /// hand-audit each pair. Empty when no pairs are in the report.
    /// Capped at `drill_down::DEFAULT_K` entries.
    #[serde(default)]
    pub drill_down: Vec<PairDrilldown>,
}

impl DiffReport {
    /// The highest severity observed across all axes.
    pub fn worst_severity(&self) -> Severity {
        self.rows
            .iter()
            .map(|r| r.severity)
            .max()
            .unwrap_or(Severity::None)
    }

    /// Render as a markdown table (one row per axis, columns as in
    /// the nine axes in README.md).
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "# Shadow diff — {pair_count} response pair{s}\n",
            pair_count = self.pair_count,
            s = if self.pair_count == 1 { "" } else { "s" }
        )
        .ok();
        writeln!(
            out,
            "| axis | baseline | candidate | delta | 95% CI | severity | flags |"
        )
        .ok();
        writeln!(
            out,
            "|------|---------:|----------:|------:|--------|----------|-------|"
        )
        .ok();
        for row in &self.rows {
            let flags = if row.flags.is_empty() {
                String::new()
            } else {
                row.flags
                    .iter()
                    .map(|f| f.label())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            writeln!(
                out,
                "| {axis} | {bm:.3} | {cm:.3} | {d:+.3} | [{lo:+.3}, {hi:+.3}] | {sev} | {flags} |",
                axis = row.axis.label(),
                bm = row.baseline_median,
                cm = row.candidate_median,
                d = row.delta,
                lo = row.ci95_low,
                hi = row.ci95_high,
                sev = row.severity.label(),
            )
            .ok();
        }
        writeln!(
            out,
            "\n**Worst severity:** `{}` &nbsp; · &nbsp; baseline `{}` &nbsp; · &nbsp; candidate `{}`",
            self.worst_severity().label(),
            short(&self.baseline_trace_id),
            short(&self.candidate_trace_id),
        )
        .ok();
        if let Some(fd) = &self.first_divergence {
            writeln!(
                out,
                "\n### First divergence\n\n**Turn** baseline `#{}` ↔ candidate `#{}` &nbsp; · &nbsp; **Kind** `{}` &nbsp; · &nbsp; **Axis** `{}` &nbsp; · &nbsp; **Confidence** {:.0}%\n\n> {}",
                fd.baseline_turn,
                fd.candidate_turn,
                fd.kind.label(),
                fd.primary_axis.label(),
                fd.confidence * 100.0,
                fd.explanation,
            )
            .ok();
        }
        if !self.drill_down.is_empty() {
            writeln!(out, "\n### Top regressive pairs").ok();
            let shown = self.drill_down.len().min(3);
            for row in &self.drill_down[..shown] {
                writeln!(
                    out,
                    "\n- **pair `#{i}`** &nbsp;·&nbsp; dominant: `{axis}` &nbsp;·&nbsp; score `{score:.2}`",
                    i = row.pair_index,
                    axis = row.dominant_axis.label(),
                    score = row.regression_score,
                )
                .ok();
                let mut contributions: Vec<_> = row.axis_scores.iter().collect();
                contributions.sort_by(|a, b| {
                    b.normalized_delta
                        .partial_cmp(&a.normalized_delta)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for score in contributions.iter().take(2) {
                    if score.normalized_delta < 0.05 {
                        break;
                    }
                    writeln!(
                        out,
                        "  - `{axis}`: {bv:.2} → {cv:.2} &nbsp;(delta `{d:+.2}`, norm `{n:.2}`)",
                        axis = score.axis.label(),
                        bv = score.baseline_value,
                        cv = score.candidate_value,
                        d = score.delta,
                        n = score.normalized_delta,
                    )
                    .ok();
                }
            }
            if self.drill_down.len() > shown {
                writeln!(
                    out,
                    "\n<details><summary>+ {} more regressive pair(s)</summary>\n",
                    self.drill_down.len() - shown
                )
                .ok();
                for row in &self.drill_down[shown..] {
                    writeln!(
                        out,
                        "- pair `#{i}` &nbsp;·&nbsp; `{axis}` &nbsp;·&nbsp; score `{score:.2}`",
                        i = row.pair_index,
                        axis = row.dominant_axis.label(),
                        score = row.regression_score,
                    )
                    .ok();
                }
                writeln!(out, "\n</details>").ok();
            }
        }
        out
    }

    /// Render a plain-text table suitable for `stdout`.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        writeln!(out, "Shadow diff — {} response pair(s)", self.pair_count).ok();
        writeln!(out, "baseline : {}", self.baseline_trace_id).ok();
        writeln!(out, "candidate: {}", self.candidate_trace_id).ok();
        writeln!(out).ok();
        writeln!(
            out,
            "{:<22} {:>10} {:>10} {:>10} {:>20} {:>10}  flags",
            "axis", "baseline", "candidate", "delta", "95% CI", "severity"
        )
        .ok();
        writeln!(out, "{}", "-".repeat(100)).ok();
        for row in &self.rows {
            let flags = if row.flags.is_empty() {
                String::new()
            } else {
                row.flags
                    .iter()
                    .map(|f| f.label())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            writeln!(
                out,
                "{axis:<22} {bm:>10.3} {cm:>10.3} {d:>+10.3} {ci:>20} {sev:>10}  {flags}",
                axis = row.axis.label(),
                bm = row.baseline_median,
                cm = row.candidate_median,
                d = row.delta,
                ci = format!("[{:+.2}, {:+.2}]", row.ci95_low, row.ci95_high),
                sev = row.severity.label(),
            )
            .ok();
        }
        writeln!(out, "\nworst severity: {}", self.worst_severity().label()).ok();
        if let Some(fd) = &self.first_divergence {
            writeln!(out).ok();
            writeln!(
                out,
                "first divergence: baseline turn #{}  ↔  candidate turn #{}",
                fd.baseline_turn, fd.candidate_turn,
            )
            .ok();
            writeln!(
                out,
                "  kind: {}  ·  axis: {}  ·  confidence: {:.0}%",
                fd.kind.label(),
                fd.primary_axis.label(),
                fd.confidence * 100.0,
            )
            .ok();
            writeln!(out, "  explanation: {}", fd.explanation).ok();
        }
        if !self.drill_down.is_empty() {
            writeln!(out).ok();
            let shown = self.drill_down.len().min(3);
            writeln!(
                out,
                "top regressive pairs ({shown} shown of {total}):",
                total = self.drill_down.len(),
            )
            .ok();
            for row in &self.drill_down[..shown] {
                writeln!(
                    out,
                    "  pair #{i}  ·  dominant axis: {axis}  ·  score: {score:.2}",
                    i = row.pair_index,
                    axis = row.dominant_axis.label(),
                    score = row.regression_score,
                )
                .ok();
                // Show the top 2 contributing axes inline.
                let mut contributions: Vec<_> = row.axis_scores.iter().collect();
                contributions.sort_by(|a, b| {
                    b.normalized_delta
                        .partial_cmp(&a.normalized_delta)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for score in contributions.iter().take(2) {
                    if score.normalized_delta < 0.05 {
                        break;
                    }
                    writeln!(
                        out,
                        "    {axis}: {bv:.2} → {cv:.2} (delta {d:+.2}, norm {n:.2})",
                        axis = score.axis.label(),
                        bv = score.baseline_value,
                        cv = score.candidate_value,
                        d = score.delta,
                        n = score.normalized_delta,
                    )
                    .ok();
                }
            }
            if self.drill_down.len() > shown {
                writeln!(out, "  +{} more", self.drill_down.len() - shown).ok();
            }
        }
        out
    }
}

fn short(id: &str) -> String {
    if id.len() > 16 {
        format!("{}…{}", &id[..12], &id[id.len() - 4..])
    } else {
        id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::axes::Axis;

    fn fake_report() -> DiffReport {
        let rows = Axis::all()
            .iter()
            .map(|a| AxisStat {
                axis: *a,
                baseline_median: 1.0,
                candidate_median: 1.1,
                delta: 0.1,
                ci95_low: 0.05,
                ci95_high: 0.15,
                severity: Severity::Minor,
                n: 10,
                flags: Vec::new(),
            })
            .collect();
        DiffReport {
            rows,
            baseline_trace_id:
                "sha256:aaaa0000bbbb1111cccc2222dddd3333eeee4444ffff5555aaaa6666bbbb".to_string(),
            candidate_trace_id:
                "sha256:0000aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777".to_string(),
            pair_count: 10,
            first_divergence: None,
            divergences: Vec::new(),
            recommendations: Vec::new(),
            drill_down: Vec::new(),
        }
    }

    #[test]
    fn markdown_has_nine_rows_plus_header() {
        let md = fake_report().to_markdown();
        assert!(md.contains("| axis |"));
        let row_count = md.lines().filter(|l| l.starts_with("| semantic")).count()
            + md.lines().filter(|l| l.starts_with("| tool-call")).count()
            + md.lines().filter(|l| l.starts_with("| refusal")).count()
            + md.lines().filter(|l| l.starts_with("| verbosity")).count()
            + md.lines().filter(|l| l.starts_with("| latency")).count()
            + md.lines().filter(|l| l.starts_with("| cost")).count()
            + md.lines().filter(|l| l.starts_with("| reasoning")).count()
            + md.lines().filter(|l| l.starts_with("| llm-judge")).count()
            + md.lines().filter(|l| l.starts_with("| format")).count();
        assert_eq!(row_count, 9);
    }

    #[test]
    fn terminal_renders_all_axes() {
        let txt = fake_report().to_terminal();
        for axis in Axis::all() {
            assert!(txt.contains(axis.label()), "missing axis {:?}", axis);
        }
    }

    #[test]
    fn worst_severity_picks_highest() {
        let mut r = fake_report();
        r.rows[3].severity = Severity::Severe;
        r.rows[0].severity = Severity::Moderate;
        assert_eq!(r.worst_severity(), Severity::Severe);
    }

    #[test]
    fn roundtrip_through_serde_json() {
        let r = fake_report();
        let wire = serde_json::to_string(&r).unwrap();
        let back: DiffReport = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, r);
    }
}
